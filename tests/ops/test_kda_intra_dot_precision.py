# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Check intra-backward against the differentiable chunk-local attention matrices."""

import importlib.util
import inspect
import math
import os
from pathlib import Path

import pytest


@pytest.mark.parametrize('mode', [None, '', 'ieee', 'tf32x3', 'default', 'tf32', 'bad'])
@pytest.mark.parametrize('supported', [False, True])
def test_intra_dot_precision_policy(mode, supported, monkeypatch):
    path = Path(__file__).resolve().parents[2] / 'fla/ops/kda/backward_tuning.py'
    spec = importlib.util.spec_from_file_location('_kda_dot_policy', path)
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    if mode is None:
        monkeypatch.delenv('FLA_KDA_BWD_INTRA_DOT_PRECISION', raising=False)
    else:
        monkeypatch.setenv('FLA_KDA_BWD_INTRA_DOT_PRECISION', mode)
    if mode in ('tf32', 'bad') or (mode == 'tf32x3' and not supported):
        with pytest.raises(ValueError):
            policy.intra_dot_precision(supported)
    else:
        assert policy.intra_dot_precision(supported) == (None if mode == 'default' else mode or 'ieee')


def intra_reference(q, k, g, beta, dAqk, dAkk, boundaries):
    import torch

    groups = g.shape[2] // q.shape[2]
    q = q.double().repeat_interleave(groups, dim=2).detach().requires_grad_()
    k = k.double().repeat_interleave(groups, dim=2).detach().requires_grad_()
    g = g.double().detach().requires_grad_()
    beta = beta.double().detach().requires_grad_()
    loss = q.new_zeros(())
    for batch in range(q.shape[0]):
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            for left in range(start, stop, 64):
                right = min(left + 64, stop)
                length = right - left
                qc = q[batch, left:right].transpose(0, 1)
                kc = k[batch, left:right].transpose(0, 1)
                gc = g[batch, left:right].transpose(0, 1)
                bc = beta[batch, left:right].transpose(0, 1)
                decay = torch.exp2(gc[:, :, None, :] - gc[:, None, :, :])
                aqk = (qc[:, :, None, :] * kc[:, None, :, :] * decay).sum(-1)
                akk = (kc[:, :, None, :] * kc[:, None, :, :] * decay).sum(-1) * bc[:, :, None]
                wqk = dAqk[batch, left:right, :, :length].double().transpose(0, 1).tril()
                wkk = dAkk[batch, left:right, :, :length].double().transpose(0, 1).tril()
                loss = loss + (aqk * wqk + akk * wkk).sum()
    dq, dk, db, dg = torch.autograd.grad(loss, (q, k, beta, g))
    # the kernel accepts base-2 cumulative gates but returns natural-log gate gradients
    return dq, dk, db, dg / math.log(2)


def test_intra_reference_single_token():
    torch = pytest.importorskip('torch')
    q = torch.tensor([[[[2., 3.]]]])
    k = torch.tensor([[[[4., 5.]]]])
    g = torch.tensor([[[[-1., -2.]]]])
    beta = torch.full((1, 1, 1), .5)
    aqk, akk = torch.zeros(1, 1, 1, 64), torch.zeros(1, 1, 1, 64)
    aqk[..., 0], akk[..., 0] = 2., 3.
    dq, dk, db, dg = intra_reference(q, k, g, beta, aqk, akk, [0, 1])
    torch.testing.assert_close(dq.flatten(), torch.tensor([8., 10.], dtype=torch.float64))
    torch.testing.assert_close(dk.flatten(), torch.tensor([16., 21.], dtype=torch.float64))
    torch.testing.assert_close(db.flatten(), torch.tensor([123.], dtype=torch.float64))
    torch.testing.assert_close(dg, torch.zeros_like(dg), atol=1e-12, rtol=0)


@pytest.mark.skipif(os.getenv('FLA_KDA_INTRA_DOT_GPU_TEST') != '1', reason='explicit GPU opt-in')
@pytest.mark.parametrize('dtype_name,dim', [('float16', 64), ('bfloat16', 128)])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('safe', [False, True])
@pytest.mark.parametrize('head_first', ['0', '1'])
@pytest.mark.parametrize('precision', ['ieee', 'tf32x3'])
def test_intra_dot_gradients(dtype_name, dim, packed, safe, head_first, precision, monkeypatch):
    import torch

    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
    from fla.utils import IS_NVIDIA, IS_TF32_SUPPORTED, assert_close, device

    if not IS_NVIDIA:
        pytest.skip('NVIDIA fixed-launch regression')
    if precision == 'tf32x3' and not IS_TF32_SUPPORTED:
        pytest.skip('TF32 requires supported NVIDIA hardware')
    assert os.environ.get('TRITON_F32_DEFAULT') != 'ieee', 'Run with global IEEE unset to test the local policy'
    torch.manual_seed(42)
    monkeypatch.setenv('FLA_KDA_BWD_INTRA_TILE', '32,1,3')
    monkeypatch.setenv('FLA_KDA_BWD_INTRA_HEAD_FIRST', head_first)
    monkeypatch.setenv('FLA_KDA_BWD_INTRA_DOT_PRECISION', precision)
    dtype = getattr(torch, dtype_name)
    b, t, h, hv = (1 if packed else 2), 129, 2, 4
    boundaries = [0, 17, 66, 129] if packed else [0, t]

    def rand(shape, dtype=torch.float32):
        return torch.randn(shape, dtype=dtype, device=device) * .1

    inputs = dict(
        q=rand((b, t, h, dim), dtype),
        k=rand((b, t, h, dim), dtype),
        g=-torch.rand(b, t, hv, dim, device=device).cumsum(1) * .01,
        beta=torch.rand(b, t, hv, dtype=dtype, device=device),
        dAqk=rand((b, t, hv, 64)),
        dAkk=rand((b, t, hv, 64)),
        dq=rand((b, t, hv, dim)),
        dk=rand((b, t, hv, dim)),
        db=rand((b, t, hv)),
        dg=rand((b, t, hv, dim)),
    )
    reference = intra_reference(**{k: inputs[k] for k in ('q', 'k', 'g', 'beta', 'dAqk', 'dAkk')}, boundaries=boundaries)
    snapshots = {name: value.clone() for name, value in inputs.items()}
    fn = inspect.unwrap(chunk_kda_bwd_intra)
    actual = fn(
        **inputs,
        cu_seqlens=torch.tensor(boundaries, device=device) if packed else None,
        safe_gate=safe,
    )
    for name, ref, got in zip(('dq', 'dk', 'db', 'dg'), reference, actual):
        expected = ref + inputs[name].double()
        assert got.shape == expected.shape and got.dtype == inputs[name].dtype
        assert torch.isfinite(got).all()
        assert_close(name, expected.float(), got.float(), 1e-4)
    for name, before in snapshots.items():
        torch.testing.assert_close(before, inputs[name], atol=0, rtol=0)
