# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Launch-policy checks and repeated dense/packed state-backward comparisons."""

import ast
import importlib.util
import inspect
import os
import sys
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('kda_tuning_policy', ROOT / 'fla/ops/kda/backward_tuning.py')
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


@pytest.mark.parametrize('bv,order,expected', [
    ('', '0', None), ('16', '0', (16, False)), ('32', '1', (32, True)), ('64', '0', (64, False)),
])
def test_state_policy(monkeypatch, bv, order, expected):
    monkeypatch.setenv('FLA_KDA_BWD_DHU_BV', bv)
    monkeypatch.setenv('FLA_KDA_BWD_DHU_HEAD_FIRST', order)
    assert policy.state_backward_config() == expected


@pytest.mark.parametrize('bv,order', [('', '1'), ('128', '0'), ('32,4,3', '0'), ('16', 'yes'), ('-1', '0')])
def test_bad_state_policy(monkeypatch, bv, order):
    monkeypatch.setenv('FLA_KDA_BWD_DHU_BV', bv)
    monkeypatch.setenv('FLA_KDA_BWD_DHU_HEAD_FIRST', order)
    with pytest.raises(ValueError):
        policy.state_backward_config()


@pytest.mark.parametrize('raises', [False, True])
def test_dispatch_preserved_and_scope_reset(monkeypatch, raises):
    common = ModuleType('fla.ops.common.chunk_delta_h')
    common._KDA_DHU_LAUNCH_CONFIG = ContextVar('test_dhu', default=None)
    utils = ModuleType('fla.utils')
    utils.IS_NVIDIA_BLACKWELL = True
    received = []

    def dispatched(**kwargs):
        received.append((common._KDA_DHU_LAUNCH_CONFIG.get(), kwargs))
        if raises:
            raise RuntimeError('backend error')
        return 'backend result'

    common.chunk_gated_delta_rule_bwd_dhu = dispatched
    monkeypatch.setitem(sys.modules, common.__name__, common)
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    monkeypatch.setenv('FLA_KDA_BWD_DHU_BV', '16')
    monkeypatch.setenv('FLA_KDA_BWD_DHU_HEAD_FIRST', '1')
    if raises:
        with pytest.raises(RuntimeError, match='backend error'):
            policy.state_backward(intra_affine_summary='saved summary')
    else:
        assert policy.state_backward(intra_affine_summary='saved summary') == 'backend result'
    assert received == [((16, True), {'intra_affine_summary': 'saved summary'})]
    assert common._KDA_DHU_LAUNCH_CONFIG.get() is None


@pytest.mark.parametrize('bv', [16, 32, 64])
@pytest.mark.parametrize('head_first', [False, True])
def test_fixed_launch_keeps_blackwell_restriction(monkeypatch, bv, head_first):
    path = ROOT / 'fla/ops/common/chunk_delta_h.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_state_backward_launch')
    runtime = ModuleType('triton.runtime')

    class JITFunction:
        pass

    runtime.JITFunction = JITFunction
    monkeypatch.setitem(sys.modules, 'triton.runtime', runtime)
    jit = JITFunction()
    wrapped = SimpleNamespace(fn=SimpleNamespace(fn=jit))
    context = ContextVar('test_launch', default=None)
    namespace = dict(_KDA_DHU_LAUNCH_CONFIG=context, IS_NVIDIA_BLACKWELL=True,
                     chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64=wrapped)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    select = namespace[fn.name]
    assert select(None, None, None, None, None) == (wrapped, {})
    context.set((bv, head_first))
    selected, kwargs = select(None, object(), object(), None, object())
    assert selected is jit
    assert kwargs == dict(BV=bv, HEAD_FIRST=head_first, num_warps=2, num_stages=3,
                          USE_G=False, USE_GK=True, USE_INITIAL_STATE=True,
                          USE_FINAL_STATE_GRADIENT=False, IS_VARLEN=True)
    namespace['IS_NVIDIA_BLACKWELL'] = False
    with pytest.raises(ValueError, match='Blackwell'):
        select(None, None, None, None, None)


@pytest.mark.skipif(os.getenv('FLA_KDA_STATE_BWD_GPU_TEST') != '1', reason='explicit GPU opt-in')
@pytest.mark.parametrize('bv', [16, 32, 64])
@pytest.mark.parametrize('head_first', [False, True])
@pytest.mark.parametrize('dtype_name,dim,packed,state_v_first,gva,states', [
    ('float16', 64, False, False, False, True),
    ('float16', 64, True, True, True, True),
    ('float16', 60, True, False, False, False),
    ('float16', 128, False, True, True, False),
    ('bfloat16', 128, False, False, False, True),
    ('bfloat16', 128, True, True, True, True),
    ('bfloat16', 128, True, False, True, False),
    ('bfloat16', 64, False, True, False, False),
])
def test_repeated_state_outputs(monkeypatch, bv, head_first, dtype_name, dim, packed, state_v_first, gva, states):
    import torch

    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
    from fla.ops.kda.backward_tuning import state_backward
    from fla.utils import IS_NVIDIA_BLACKWELL, device

    if not IS_NVIDIA_BLACKWELL:
        pytest.skip('Blackwell launch tuning')
    torch.manual_seed(42)
    b, t, h, hv, v = (1 if packed else 2), 500, 2, (4 if gva else 2), dim
    dtype = getattr(torch, dtype_name)

    def rand(shape):
        return torch.randn(shape, device=device, dtype=dtype) * .05

    cu = torch.tensor([0, 63, 257, 500], device=device, dtype=torch.int64) if packed else None
    n = 3 if packed else b
    shape = (v, dim) if state_v_first else (dim, v)
    args = dict(q=rand((b, t, h, dim)), k=rand((b, t, h, dim)), w=rand((b, t, hv, dim)),
                do=rand((b, t, hv, v)), dv=rand((b, t, hv, v)),
                gk=-torch.rand((b, t, hv, dim), device=device),
                h0=rand((n, hv, *shape)).float() if states else None,
                dht=rand((n, hv, *shape)).float() if states else None,
                scale=.1, cu_seqlens=cu, chunk_size=64, state_v_first=state_v_first)
    originals = {name: value.clone() for name, value in args.items() if isinstance(value, torch.Tensor)}
    reference = inspect.unwrap(chunk_gated_delta_rule_bwd_dhu)(**args)
    monkeypatch.setenv('FLA_KDA_BWD_DHU_BV', str(bv))
    monkeypatch.setenv('FLA_KDA_BWD_DHU_HEAD_FIRST', str(int(head_first)))
    previous = None
    for _ in range(3):
        actual = state_backward(**args)
        for i, (want, got) in enumerate(zip(reference, actual)):
            if want is None:
                assert got is None
                continue
            assert torch.isfinite(got).all()
            torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-3)
            if previous is not None:
                torch.testing.assert_close(got, previous[i], atol=0, rtol=0)
        previous = actual
    for name, original in originals.items():
        torch.testing.assert_close(args[name], original, atol=0, rtol=0)


@pytest.mark.skipif(os.getenv('FLA_KDA_STATE_BWD_GPU_TEST') != '1', reason='explicit GPU opt-in')
@pytest.mark.parametrize('bv', [16, 32, 64])
@pytest.mark.parametrize('head_first', [False, True])
@pytest.mark.parametrize('packed', [False, True])
def test_intracard_subscans(bv, head_first, packed):
    import torch

    from fla.ops.common.chunk_delta_h import _KDA_DHU_LAUNCH_CONFIG
    from fla.ops.common.intracard_cp import intracard_bwd_dhu
    from fla.utils import IS_NVIDIA_BLACKWELL, device

    if not IS_NVIDIA_BLACKWELL:
        pytest.skip('Blackwell launch tuning')
    torch.manual_seed(42)
    t, h, dim = 32768, 2, 128

    def rand(shape):
        return torch.randn(shape, device=device, dtype=torch.bfloat16) * .02

    cu_cpu = torch.tensor([0, 32767, 32768] if packed else [0, t], dtype=torch.long)
    cu = cu_cpu.to(device)
    args = {name: rand((1, t, h, dim)) for name in ('q', 'k', 'w', 'do', 'dv')}
    args.update(gk=-torch.rand((1, t, h, dim), device=device), scale=.1, cu_seqlens=cu,
                cu_seqlens_cpu=cu_cpu, chunk_size=64, max_splits=4)
    token = _KDA_DHU_LAUNCH_CONFIG.set(None)
    try:
        expected = intracard_bwd_dhu(**args)
        _KDA_DHU_LAUNCH_CONFIG.set((bv, head_first))
        for _ in range(3):
            actual = intracard_bwd_dhu(**args)
            for want, got in zip(expected, actual):
                if want is None:
                    assert got is None
                else:
                    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-3)
    finally:
        _KDA_DHU_LAUNCH_CONFIG.reset(token)
