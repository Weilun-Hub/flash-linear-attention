# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Intra-backward must promote beta before forming FP32 key/decay products."""

import inspect
import os

import pytest


@pytest.mark.skipif(os.getenv('FLA_KDA_INTRA_BETA_GPU_TEST') != '1', reason='explicit GPU opt-in')
@pytest.mark.parametrize('dtype_name,dim', [('float16', 64), ('bfloat16', 128)])
@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('safe', [False, True])
@pytest.mark.parametrize('head_first', ['0', '1'])
def test_intra_beta_storage_dtype(dtype_name, dim, packed, safe, head_first, monkeypatch):
    import torch

    from fla.ops.kda.chunk_intra import chunk_kda_bwd_intra
    from fla.utils import IS_NVIDIA, assert_close, device

    if not IS_NVIDIA:
        pytest.skip('NVIDIA fixed-launch regression')
    torch.manual_seed(42)
    monkeypatch.setenv('FLA_KDA_BWD_INTRA_TILE', '32,1,3')
    monkeypatch.setenv('FLA_KDA_BWD_INTRA_HEAD_FIRST', head_first)
    dtype = getattr(torch, dtype_name)
    b, t, h = (1 if packed else 2), 129, 2

    def rand(shape, dtype=torch.float32):
        return torch.randn(shape, dtype=dtype, device=device) * .5

    shape = (b, t, h, dim)
    beta = torch.rand(b, t, h, dtype=dtype, device=device)
    inputs = dict(q=rand(shape, dtype), k=rand(shape, dtype), g=-rand(shape).abs() * .1, beta=beta,
                  dAqk=rand((b, t, h, 64)), dAkk=rand((b, t, h, 64)),
                  dq=rand(shape), dk=rand(shape), db=rand((b, t, h)), dg=rand(shape),
                  cu_seqlens=torch.tensor([0, 17, 66, 129], device=device) if packed else None,
                  safe_gate=safe)
    snapshots = {name: value.clone() for name, value in inputs.items() if isinstance(value, torch.Tensor)}
    # beta.float() preserves every input value; only the multiplication precision changes
    fn = inspect.unwrap(chunk_kda_bwd_intra)
    expected = fn(**dict(inputs, beta=beta.float()))
    actual = fn(**inputs)
    for name, ref, got in zip(('dq', 'dk', 'db', 'dg'), expected, actual):
        assert got.dtype == ref.dtype
        assert torch.isfinite(got).all()
        assert_close(name, ref.float(), got.float(), 1e-5)
    for name, before in snapshots.items():
        torch.testing.assert_close(before, inputs[name], atol=0, rtol=0)
