# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import os

import pytest
import torch

import fla.ops.common.intracard_cp as intracard_cp_mod
from fla.ops.kda import chunk_kda
from fla.utils import assert_close, device, device_platform


@pytest.fixture(autouse=True)
def isolated_split_cache(monkeypatch):
    monkeypatch.delenv('FLA_INTRACARD_FORCE_SPLITS', raising=False)
    intracard_cp_mod._intracard_cache.clear()
    yield
    intracard_cp_mod._intracard_cache.clear()


def prepare_metadata(cu_seqlens):
    return intracard_cp_mod._prepare_intracard_cache_entry(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens,
        num_heads=64,
        chunk_size=64,
        max_splits=32,
        device=cu_seqlens.device,
    )


@pytest.mark.parametrize('count', [1, 2, 4, 8, 16])
def test_forced_split_boundaries(monkeypatch, count):
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', str(count))
    cu = torch.tensor([0, 32768], dtype=torch.int32)
    entry = prepare_metadata(cu)
    if count == 1:
        assert entry is None
        return
    assert entry.total_subseqs == count
    assert entry.S_split_total == count
    assert entry.cu_seqlens_subseq_values == list(range(0, 32769, 32768 // count))
    assert entry.cu_seqlens_split_values == [
        value for i in range(count) for value in (i * (32768 // count), (i + 1) * (32768 // count))
    ]
    assert prepare_metadata(cu) is entry


def test_forced_split_cache_switch(monkeypatch):
    cu = torch.tensor([0, 32768], dtype=torch.int32)
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', '2')
    first = prepare_metadata(cu)
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', '4')
    second = prepare_metadata(cu)
    assert first is not second
    assert second.total_subseqs == 4
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', '1')
    assert prepare_metadata(cu) is None
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', '2')
    assert prepare_metadata(cu) is first


def test_zero_override_preserves_heuristic(monkeypatch):
    monkeypatch.setattr(intracard_cp_mod, 'get_multiprocessor_count', lambda: 132)
    cu = torch.tensor([0, 32768], dtype=torch.int32)
    assert prepare_metadata(cu) is None
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', '0')
    assert prepare_metadata(cu) is None
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', '2')
    assert prepare_metadata(cu).total_subseqs == 2


@pytest.mark.parametrize(('count', 'boundaries'), [
    ('-1', [0, 512]), ('33', [0, 512]), ('invalid', [0, 512]),
    ('2', [0, 256, 512]), ('2', [64, 576]), ('2', [0, 513]),
    ('2', [0, 64]), ('2', [0, 0]), ('1', [0, 256, 512]),
])
def test_invalid_forced_splits(monkeypatch, count, boundaries):
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', count)
    with pytest.raises(ValueError):
        prepare_metadata(torch.tensor(boundaries, dtype=torch.int32))


@pytest.mark.skipif(os.environ.get('FLA_DISABLE_BACKEND_DISPATCH') == '1', reason='backend dispatch disabled')
@pytest.mark.skipif(device_platform not in ('cuda', 'hip'), reason='requires a CUDA or ROCm GPU')
@pytest.mark.parametrize('count', [1, 2, 4, 8])
@pytest.mark.parametrize('state_v_first', [False, True])
def test_forced_split_kda_gradients(monkeypatch, count, state_v_first):
    torch.manual_seed(42)
    monkeypatch.setenv('FLA_FLASH_KDA', '0')
    monkeypatch.setenv('FLA_INTRACARD_FORCE_SPLITS', str(count))
    shape = (1, 2048, 2, 64)
    dtype = torch.bfloat16
    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(shape, device=device, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(shape, device=device, dtype=dtype)
    g = -torch.rand(shape, device=device, dtype=torch.float32) * 0.02
    beta = torch.randn(shape[:-1], device=device, dtype=dtype).sigmoid()
    cu = torch.tensor([0, shape[1]], device=device, dtype=torch.int32)
    cu_cpu = cu.cpu()
    do = torch.randn_like(v)

    def run(enabled):
        monkeypatch.setenv('FLA_INTRACARD_CP', '1' if enabled else '0')
        inputs = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        o, _ = chunk_kda(
            q=inputs[0],
            k=inputs[1],
            v=inputs[2],
            g=inputs[3],
            beta=inputs[4],
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            state_v_first=state_v_first,
            chunk_size=64,
        )
        return (o, *torch.autograd.grad(o, inputs, do))

    ref = run(False)
    actual = run(True)
    for name, expected, result, tolerance in zip(
        ('o', 'dq', 'dk', 'dv', 'dg', 'db'), ref, actual, (0.01, 0.01, 0.01, 0.01, 0.02, 0.02),
    ):
        assert_close(name, expected, result, tolerance)
        assert torch.isfinite(result).all()
    if count > 1:
        assert any(entry.total_subseqs == count for entry in intracard_cp_mod._intracard_cache.values())
