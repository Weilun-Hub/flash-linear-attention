# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Opt-in CUDA regression coverage for KDA scratch and launch experiments."""

import importlib.util
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load_policy(relative):
    spec = importlib.util.spec_from_file_location('_policy', ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


POLICY = load_policy('fla/ops/kda/backward_tuning.py')
WY = load_policy('fla/ops/kda/backends/tilelang/tuning.py')
FLAGS = ('FLA_CONV_BWD_RECOMPUTE_SILU',
         'FLA_KDA_BWD_INTRA_TILE', 'FLA_KDA_RECOMPUTE_H_TILE')
GPU = pytest.mark.skipif(os.getenv('FLA_KDA_MEMORY_KERNEL_GPU_TEST') != '1', reason='explicit GPU opt-in')


@pytest.mark.parametrize('raw', ['', '32,1,3', '16,2,1', '32,4,2', '64,8,4', '32,4', '32,a,2', '8,4,2', '32,3,2', '32,4,5'])
def test_launch_policy(monkeypatch, raw):
    monkeypatch.setenv(FLAGS[1], raw)
    if raw in ('', '32,1,3', '16,2,1', '32,4,2', '64,8,4'):
        assert POLICY.launch_config(FLAGS[1], (16, 32, 64), (1, 2, 4, 8)) == (tuple(map(int, raw.split(','))) if raw else None)
    else:
        with pytest.raises(ValueError):
            POLICY.launch_config(FLAGS[1], (16, 32, 64), (1, 2, 4, 8))


@pytest.mark.parametrize('tile,k,v,expected', [
    ('', 128, 128, True), ('', 60, 128, False), ('', 128, 100, False), ('', 32, 32, True),
    ('32,64,4', 128, 128, True), ('32,64,4', 60, 128, False), ('32,64,4', 128, 32, False),
])
def test_wy_shape_dispatch(monkeypatch, tile, k, v, expected):
    monkeypatch.setenv('FLA_KDA_BWD_WY_TILE', tile)
    monkeypatch.setenv('FLA_KDA_BWD_WY_HEAD_FIRST', '0')
    assert WY.backward_shape_supported(k, v) is expected


def assert_parity(expected, actual):
    import torch

    assert len(expected) == len(actual)
    for ref, got in zip(expected, actual):
        if ref is None:
            assert got is None
            continue
        assert ref.shape == got.shape and ref.dtype == got.dtype
        assert torch.isfinite(ref).all() and torch.isfinite(got).all()
        diff = (ref.float() - got.float()).flatten()
        ratio = diff.square().mean().sqrt() / (ref.float().square().mean().sqrt() + 1e-8)
        assert diff.abs().max() <= 1e-6 or ratio < .006, ratio.item()


@pytest.fixture
def gpu(monkeypatch):
    import torch

    from fla.utils import IS_NVIDIA, device

    if not IS_NVIDIA:
        pytest.skip('NVIDIA kernel experiments')
    for flag in FLAGS:
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.delenv('FLA_KDA_BWD_WY_TILE', raising=False)
    monkeypatch.setenv('FLA_FLASH_KDA', '0')
    monkeypatch.setenv('FLA_INTRACARD_CP', '0')
    torch.manual_seed(42)
    def poison(allocate):
        def wrapped(*args, **kwargs):
            result = allocate(*args, **kwargs)
            if result.is_floating_point():
                with torch.no_grad():
                    result.fill_(float('nan'))
            return result
        return wrapped

    monkeypatch.setattr(torch, 'empty_like', poison(torch.empty_like))
    monkeypatch.setattr(torch, 'empty', poison(torch.empty))
    monkeypatch.setattr(torch.Tensor, 'new_empty', poison(torch.Tensor.new_empty))
    return torch, device


@GPU
@pytest.mark.parametrize('packed,safe,k,hv', [(False, False, 128, 2), (True, True, 128, 2),
                                           (True, False, 60, 4), (False, True, 100, 2)])
@pytest.mark.parametrize('tile', ['32,1,3', '16,2,2', '32,4,2', '64,4,3'])
def test_intra_gradients_and_storage(gpu, monkeypatch, packed, safe, k, hv, tile):
    torch, device = gpu
    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra

    b, t, h = (1 if packed else 2), 257, 2
    def rand(shape, dtype=torch.float32):
        return torch.randn(shape, dtype=dtype, device=device) * .05
    cu = torch.tensor([0, 1, 63, 128, 257], device=device) if packed else None
    kwargs = dict(q=rand((b, t, h, k), torch.bfloat16), k=rand((b, t, h, k), torch.bfloat16),
                  g=-rand((b, t, hv, k)).abs(), beta=rand((b, t, hv)).sigmoid(),
                  dAqk=rand((b, t, hv, 64)), dAkk=rand((b, t, hv, 64)),
                  dq=rand((b, t, hv, k)), dk=rand((b, t, hv, k)), dg=rand((b, t, hv, k)),
                  db=rand((b, t, hv)), cu_seqlens=cu, safe_gate=safe)
    original = {key: x.clone() for key, x in kwargs.items() if isinstance(x, torch.Tensor)}
    expected = chunk_kda_bwd_intra(**kwargs)
    monkeypatch.setenv(FLAGS[1], tile)
    actual = chunk_kda_bwd_intra(**kwargs)
    assert_parity(expected, actual)
    for name, out in zip(('dq', 'dk', 'dg'), (actual[0], actual[1], actual[3])):
        assert out.data_ptr() != kwargs[name].data_ptr()
    for name, before in original.items():
        torch.testing.assert_close(kwargs[name], before, rtol=0, atol=0)


@GPU
@pytest.mark.parametrize('packed,vf,k,v', [(False, False, 128, 128), (True, True, 128, 128),
                                        (True, False, 60, 96)])
@pytest.mark.parametrize('tile', ['16,2,2', '32,2,2', '64,2,3'])
def test_recurrent_reconstruction(gpu, monkeypatch, packed, vf, k, v, tile):
    torch, device = gpu
    from fla.ops.kda.backward_tuning import reconstruct_state

    b, t, h = (1 if packed else 2), 257, 2
    def rand(shape):
        return torch.randn(shape, dtype=torch.bfloat16, device=device) * .02
    cu = torch.tensor([0, 1, 63, 128, 257], device=device) if packed else None
    n = len(cu) - 1 if packed else b
    shape = (v, k) if vf else (k, v)
    kwargs = dict(k=rand((b, t, h, k)), w=rand((b, t, h, k)), u=rand((b, t, h, v)),
                  gk=-torch.rand((b, t, h, k), device=device),
                  initial_state=rand((n, h, *shape)).float(), output_final_state=True,
                  state_v_first=vf, cu_seqlens=cu)
    expected = reconstruct_state(**kwargs)
    monkeypatch.setenv(FLAGS[2], tile)
    assert_parity(expected, reconstruct_state(**kwargs))


@GPU
@pytest.mark.parametrize('packed,width,states,dtype', [(False, 4, False, 'bfloat16'), (True, 4, False, 'bfloat16'),
    (True, 3, False, 'float32'), (True, 8, False, 'bfloat16'), (True, 4, True, 'bfloat16'),
    (False, 9, False, 'bfloat16')])
def test_convolution_backward_and_fallback(gpu, monkeypatch, packed, width, states, dtype):
    torch, device = gpu
    import torch.nn.functional as F

    from fla.modules.conv.causal_conv1d import causal_conv1d

    b, t, d = (1 if packed else 2), 129, 37
    dtype = getattr(torch, dtype)
    cu = torch.tensor([0, 1, 63, 129], device=device) if packed else None
    n = len(cu) - 1 if packed else b
    def rand(shape):
        return (torch.randn(shape, dtype=dtype, device=device) * .1).requires_grad_()
    # Noncontiguous x exercises the public strided-input path.
    x = rand((b, d, t)).transpose(1, 2).detach().requires_grad_()
    w, bias, residual = rand((d, width)), rand((d,)), rand((b, t, d))
    initial = rand((n, d, width)) if states else None
    inputs = [x, w, bias, residual] + ([initial] if states else [])
    dy = torch.randn_like(x)
    dht = torch.randn_like(initial) if states else None
    def run():
        out, final = causal_conv1d(x, w, bias, residual=residual, activation='silu', initial_state=initial,
                                  output_final_state=states, cu_seqlens=cu)
        outputs, grads = ((out, final), (dy, dht)) if states else ((out,), (dy,))
        return [out] + list(torch.autograd.grad(outputs, inputs, grads))
    expected = run()
    monkeypatch.setenv(FLAGS[0], '1')
    actual = run()
    assert_parity(expected, actual)
    if not states:
        # Independent FP32 convolution reference on each packed segment.
        copies = [z.detach().float().requires_grad_() for z in inputs]
        xx, ww, bb, rr = copies
        spans = [(batch, 0, t) for batch in range(b)] if not packed else [(0, 0, 1), (0, 1, 63), (0, 63, 129)]
        parts = [F.silu(F.conv1d(xx[batch:batch+1, start:end].transpose(1, 2), ww[:, None], bb,
                              padding=width-1, groups=d)[..., :end-start]).transpose(1, 2)
                 + rr[batch:batch+1, start:end] for batch, start, end in spans]
        ref = torch.cat(parts, dim=1 if packed else 0)
        reference = [ref] + list(torch.autograd.grad(ref, copies, dy.float()))
        assert_parity([z.to(dtype) for z in reference], actual)


@GPU
@pytest.mark.parametrize('packed,safe,gva,k,v', [(False, False, False, 128, 128), (True, True, False, 128, 128),
                                               (True, False, True, 60, 96)])
@pytest.mark.parametrize('experiment', ['intra', 'state', 'combined'])
def test_full_kda_forward_backward(gpu, monkeypatch, packed, safe, gva, k, v, experiment):
    torch, device = gpu
    from fla.ops.kda import chunk_kda

    b, t, h = 1, 129, 2
    hv = h * (2 if gva else 1)
    cu = torch.tensor([0, 1, 63, 129], device=device) if packed else None
    n = len(cu) - 1 if packed else b
    def rand(shape):
        return (torch.randn(shape, dtype=torch.bfloat16, device=device) * .1).requires_grad_()
    q, kk, vv = rand((b,t,h,k)), rand((b,t,h,k)), rand((b,t,hv,v))
    g = (-torch.rand((b,t,hv,k), device=device) * .1).requires_grad_()
    beta = torch.rand((b,t,hv), device=device, requires_grad=True)
    h0 = (torch.randn((n,hv,k,v), device=device) * .01).requires_grad_()
    inputs = (q, kk, vv, g, beta, h0)
    do, dht = torch.randn_like(vv), torch.randn_like(h0)
    def run():
        out, final = chunk_kda(q, kk, vv, g, beta, initial_state=h0, output_final_state=True,
                              use_qk_l2norm_in_kernel=True, safe_gate=safe, cu_seqlens=cu)
        return [out, final] + list(torch.autograd.grad((out, final), inputs, (do, dht)))
    expected = run()
    if experiment in ('intra', 'combined'):
        monkeypatch.setenv(FLAGS[1], '32,4,2')
    if experiment in ('state', 'combined'):
        monkeypatch.setenv(FLAGS[2], '16,2,2')
    before = [z.detach().clone() for z in inputs]
    actual = run()
    assert_parity(expected, actual)
    for ref, got in zip(before, inputs):
        torch.testing.assert_close(ref, got, rtol=0, atol=0)


def test_recurrent_policy_preserves_blackwell_warp_limit(monkeypatch):
    monkeypatch.setenv(FLAGS[2], '32,4,2')
    with pytest.raises(ValueError):
        POLICY.launch_config(FLAGS[2], (16, 32, 64), (2,))


@pytest.mark.parametrize('mode', ['fixed', 'graph'])
def test_intra_outputs_remain_separate(monkeypatch, mode):
    import ast
    from types import SimpleNamespace

    import torch

    source = ROOT / 'fla/ops/kda/chunk_intra.py'
    tree = ast.parse(source.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'chunk_kda_bwd_intra')
    function.decorator_list = []
    calls = []

    class Kernel:
        def __init__(self, fixed):
            self.fixed = fixed

        def __getitem__(self, grid):
            def launch(**kwargs):
                calls.append((self.fixed, kwargs))
                kwargs['db'].zero_()
            return launch

    namespace = dict(torch=torch, triton=SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b,
                     next_power_of_2=lambda x: 1 << (x-1).bit_length()),
                     IS_NVIDIA=True, IS_GATHER_SUPPORTED=False, chunk_kda_bwd_kernel_intra=Kernel(False),
                     fixed_kernel=lambda kernel: Kernel(True), launch_config=POLICY.launch_config,
                     get_static_buffer=lambda name, shape, dtype, device: torch.zeros(shape, dtype=dtype, device=device))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    monkeypatch.setenv(FLAGS[1], '64,4,2')
    def full():
        return torch.zeros(1, 64, 2, 128)
    kwargs = dict(q=full(), k=full(), g=full(), beta=torch.zeros(1, 64, 2),
                  dAqk=torch.zeros(1, 64, 2, 64), dAkk=torch.zeros(1, 64, 2, 64),
                  dq=full(), dk=full(), dg=full(), db=torch.zeros(1, 64, 2))
    kwargs['use_graph'] = mode == 'graph'
    outputs = namespace['chunk_kda_bwd_intra'](**kwargs)
    assert len(calls) == 1
    fixed, launched = calls[0]
    assert fixed == (mode == 'fixed')
    assert launched['BK'] == (64 if mode == 'fixed' else 32)
    for name, out in zip(('dq', 'dk', 'dg'), (outputs[0], outputs[1], outputs[3])):
        assert out.data_ptr() != kwargs[name].data_ptr()
