# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Tests for intra-card CP metadata, dispatch, and end-to-end paths."""

import os

import pytest
import torch

import fla.ops.common.intracard_cp as intracard_cp_mod
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu
from fla.ops.common.intracard_cp import _intracard_cache
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda
from fla.utils import assert_close, device, device_platform


@pytest.fixture(autouse=True)
def clear_intracard_cache():
    _intracard_cache.clear()
    yield
    _intracard_cache.clear()


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(device_platform not in ("cuda", "hip"), reason="requires a CUDA or ROCm GPU")
def test_chunk_kda_intracard_cache_hit_same_cu_seqlens_object(monkeypatch):
    """E2E: chunk_kda should reuse intracard precompute cache on second call.

    This test intentionally uses a very long varlen sequence so that:
    1) the intracard path is selected, and
    2) early_return is bypassed and split path is exercised.
    """
    # Enable intracard CP backend explicitly as it's disabled by default
    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    torch.manual_seed(0)
    dtype = torch.bfloat16

    # T must be large enough to bypass early_return in intracard_fwd_h.
    # With chunk_size=64 and MIN_SUBSEQ_CHUNKS=128, subseq_len floor is 8192.
    # We choose T=32768 to satisfy both:
    #   - early_return check: seq_len >= 2 * subseq_len
    #   - split threshold: seq_len >= 3 * subseq_len
    B, T, H, D = 1, 32768, 1, 32

    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = torch.full((B, T, H, D), -0.05, device=device, dtype=dtype)
    beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    A_log = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()

    call_count = 0
    original_precompute = intracard_cp_mod._precompute_intracard_indices

    def counted_precompute(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_precompute(*args, **kwargs)

    monkeypatch.setattr(intracard_cp_mod, "_precompute_intracard_indices", counted_precompute)

    with torch.inference_mode():
        o1, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )
        o2, _ = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            use_gate_in_kernel=True,
            A_log=A_log,
            dt_bias=dt_bias,
        )

    assert call_count == 1, "second call should hit cache and skip precompute"
    assert len(_intracard_cache) == 1
    key = next(iter(_intracard_cache))
    entry = _intracard_cache[key]
    assert key[0] == id(cu_seqlens)
    assert entry.cu_seqlens_ref() is cu_seqlens
    assert torch.allclose(o1, o2, atol=1e-4, rtol=1e-4)


def test_intracard_cache_miss_when_cu_seqlens_contents_change(monkeypatch):
    chunk_size = 64
    cu_seqlens = torch.tensor([0, 512, 640, 1152], dtype=torch.int32)
    monkeypatch.setattr(intracard_cp_mod, "compute_subseq_len", lambda *args, **kwargs: 2 * chunk_size)
    monkeypatch.setattr(intracard_cp_mod, "get_multiprocessor_count", lambda: 1)

    first = intracard_cp_mod._prepare_intracard_cache_entry(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens,
        num_heads=1,
        chunk_size=chunk_size,
        max_splits=32,
        device=cu_seqlens.device,
    )
    assert first is not None
    assert first.split_info.split_seq_ids == [0, 2]

    cu_seqlens.copy_(torch.tensor([0, 128, 640, 1152], dtype=cu_seqlens.dtype))
    second = intracard_cp_mod._prepare_intracard_cache_entry(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens,
        num_heads=1,
        chunk_size=chunk_size,
        max_splits=32,
        device=cu_seqlens.device,
    )

    assert second is not None
    assert second is not first
    assert second.split_info.split_seq_ids == [1, 2]
    assert second.cu_seqlens_split_values != first.cu_seqlens_split_values


def test_intracard_backend_disabled_by_default():
    """Verify that IntraCardCPBackend is disabled by default."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    # When env var is not set, backend should be disabled (default_enable=False)
    assert IntraCardCPBackend.default_enable is False


def test_intracard_backend_disabled_when_env_var_is_zero(monkeypatch):
    """Verify that IntraCardCPBackend is disabled when FLA_INTRACARD_CP=0."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    monkeypatch.setenv("FLA_INTRACARD_CP", "0")
    assert IntraCardCPBackend.is_enabled() is False


def test_intracard_backend_enabled_when_env_var_is_one(monkeypatch):
    """Verify that IntraCardCPBackend is enabled when FLA_INTRACARD_CP=1."""
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    assert IntraCardCPBackend.is_enabled() is True


def test_intracard_backend_verifiers():
    from fla.ops.common.backends.intracard import IntraCardCPBackend

    backend = IntraCardCPBackend()
    tensor = torch.empty(1)
    accepted, reason = backend.chunk_gated_delta_rule_fwd_h_verifier(
        k=tensor,
        w=tensor,
        u=tensor,
        cu_seqlens=tensor,
    )
    assert accepted is True
    assert reason is None

    accepted, reason = backend.chunk_gated_delta_rule_fwd_h_verifier(k=tensor, w=tensor, u=tensor)
    assert accepted is False
    assert reason == "cu_seqlens is None"

    accepted, reason = backend.chunk_gated_delta_rule_fwd_h_verifier(
        k=tensor,
        w=tensor,
        u=tensor,
        cu_seqlens=tensor,
        chunk_offsets=tensor,
    )
    assert accepted is False
    assert reason == "static chunk_offsets are not supported"

    accepted, reason = backend.chunk_gated_delta_rule_bwd_dhu_verifier(
        q=tensor,
        k=tensor,
        w=tensor,
        do=tensor,
        dv=tensor,
        scale=1.0,
        cu_seqlens=tensor,
    )
    assert accepted is True
    assert reason is None

    common_kwargs = {"q": tensor, "k": tensor, "w": tensor, "do": tensor, "dv": tensor}
    rejection_cases = [
        ({"scale": 1.0}, "cu_seqlens is None"),
        ({"scale": 1.0, "cu_seqlens": tensor, "chunk_offsets": tensor}, "static chunk_offsets are not supported"),
        ({"scale": 1.0, "cu_seqlens": tensor, "use_graph": True}, "use_graph=True is not supported"),
        ({"cu_seqlens": tensor}, "scale is None"),
        (
            {"scale": 1.0, "cu_seqlens": tensor, "g": tensor, "gk": tensor},
            "simultaneous scalar and per-key gates are not supported",
        ),
    ]
    for call_kwargs, expected_reason in rejection_cases:
        accepted, reason = backend.chunk_gated_delta_rule_bwd_dhu_verifier(**common_kwargs, **call_kwargs)
        assert accepted is False
        assert reason == expected_reason


def test_intracard_split_metadata_uses_explicit_pairs():
    split_info = intracard_cp_mod.SplitSeqInfo(
        split_seq_ids=[0, 2],
        start_subseq_idx=[0, 5],
        num_subseqs=[4, 4],
    )
    boundaries = [0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152]

    metadata = intracard_cp_mod._precompute_intracard_indices(split_info, boundaries, N_orig=3)

    assert metadata == (
        [0, 128, 128, 256, 256, 384, 384, 512, 640, 768, 768, 896, 896, 1024, 1024, 1152],
        8,
        [1, 2, 3, 6, 7, 8],
        [0, 1, 2, 5, 6, 7],
        [0, 4, 5],
        [3, 4, 8],
        6,
        [0, 4, 8],
        [0, 3, 6],
    )


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(device_platform not in ("cuda", "hip"), reason="requires a CUDA or ROCm GPU")
@pytest.mark.parametrize(
    ("gate_mode", "state_v_first", "H", "HV", "use_h0", "use_dht", "BT"),
    [
        pytest.param("g", False, 1, 2, True, True, 64, id="scalar-gate-gva"),
        pytest.param("gk", True, 1, 1, False, False, 32, id="vector-gate-v-first"),
        pytest.param("none", False, 1, 1, True, False, 64, id="ungated-mixed-varlen"),
    ],
)
def test_intracard_bwd_dhu(
    monkeypatch,
    gate_mode: str,
    state_v_first: bool,
    H: int,
    HV: int,
    use_h0: bool,
    use_dht: bool,
    BT: int,
):
    torch.manual_seed(42)
    B, K, V = 1, 32, 24
    long_length = 8 * BT
    if gate_mode == "none":
        cu_seqlens_values = [0, long_length, long_length + 2 * BT, 2 * long_length + 2 * BT]
    else:
        cu_seqlens_values = [0, long_length]
    T, N = cu_seqlens_values[-1], len(cu_seqlens_values) - 1
    dtype = torch.bfloat16
    q = (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.1).contiguous()
    k = (torch.randn(B, T, H, K, device=device, dtype=dtype) * 0.01).contiguous()
    w = (torch.randn(B, T, HV, K, device=device, dtype=dtype) * 0.01).contiguous()
    do = (torch.randn(B, T, HV, V, device=device, dtype=dtype) * 0.1).contiguous()
    dv = (torch.randn(B, T, HV, V, device=device, dtype=dtype) * 0.1).contiguous()
    position = torch.arange(1, BT + 1, device=device, dtype=torch.float32).repeat(T // BT)
    g = (-0.001 * position[None, :, None]).expand(B, T, HV).contiguous() if gate_mode == "g" else None
    gk = (-0.001 * position[None, :, None, None]).expand(B, T, HV, K).contiguous() if gate_mode == "gk" else None
    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    h0 = torch.randn(state_shape, device=device, dtype=torch.float32) if use_h0 else None
    dht = torch.randn(state_shape, device=device, dtype=torch.float32) if use_dht else None
    cu_seqlens = torch.tensor(cu_seqlens_values, device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()
    monkeypatch.setattr(intracard_cp_mod, "compute_subseq_len", lambda *args, **kwargs: 2 * BT)

    kwargs = {
        "q": q,
        "k": k,
        "w": w,
        "do": do,
        "dv": dv,
        "g": g,
        "gk": gk,
        "h0": h0,
        "dht": dht,
        "scale": K ** -0.5,
        "state_v_first": state_v_first,
        "cu_seqlens": cu_seqlens,
        "cu_seqlens_cpu": cu_seqlens_cpu,
        "chunk_size": BT,
    }
    monkeypatch.setenv("FLA_INTRACARD_CP", "0")
    dh_ref, dh0_ref, dv_ref = chunk_gated_delta_rule_bwd_dhu(**kwargs)

    call_count = 0
    original_bwd = intracard_cp_mod.intracard_bwd_dhu

    def counted_bwd(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_bwd(*args, **kwargs)

    monkeypatch.setattr(intracard_cp_mod, "intracard_bwd_dhu", counted_bwd)
    monkeypatch.setenv("FLA_INTRACARD_CP", "1")
    dh_tri, dh0_tri, dv_tri = chunk_gated_delta_rule_bwd_dhu(**kwargs)

    assert call_count == 1
    assert_close("dh", dh_ref, dh_tri, 0.01)
    assert_close("dv", dv_ref, dv_tri, 0.01)
    if dh0_ref is not None:
        assert_close("dh0", dh0_ref, dh0_tri, 0.01)
    assert torch.isfinite(dh_tri).all()
    assert torch.isfinite(dv_tri).all()


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(device_platform not in ("cuda", "hip"), reason="requires a CUDA or ROCm GPU")
@pytest.mark.parametrize(
    ("operation", "state_v_first", "disable_recompute"),
    [
        pytest.param("gdn", False, False, id="gdn"),
        pytest.param("kda", True, False, id="kda-recompute"),
        pytest.param("kda", False, True, id="kda-save-intermediates"),
    ],
)
def test_intracard_training_route_parity(monkeypatch, operation: str, state_v_first: bool, disable_recompute: bool):
    torch.manual_seed(42)
    B, T, H, D, BT = 1, 512, 1, 32, 64
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, D, device=device, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    beta = torch.randn(B, T, H, device=device, dtype=dtype).sigmoid()
    gate_shape = (B, T, H) if operation == "gdn" else (B, T, H, D)
    g = -torch.rand(gate_shape, device=device, dtype=torch.float32) * 0.02
    h0 = torch.randn(B, H, D, D, device=device, dtype=torch.float32)
    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()
    monkeypatch.setattr(intracard_cp_mod, "compute_subseq_len", lambda *args, **kwargs: 2 * BT)
    monkeypatch.setenv("FLA_FLASH_KDA", "0")
    monkeypatch.setenv("FLA_FLASH_QLA", "0")

    def run(enabled: bool):
        monkeypatch.setenv("FLA_INTRACARD_CP", "1" if enabled else "0")
        inputs = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta, h0)]
        q_i, k_i, v_i, g_i, beta_i, h0_i = inputs
        if operation == "gdn":
            o, ht = chunk_gated_delta_rule(
                q=q_i,
                k=k_i,
                v=v_i,
                g=g_i,
                beta=beta_i,
                initial_state=h0_i,
                output_final_state=True,
                state_v_first=state_v_first,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                chunk_size=BT,
            )
        else:
            o, ht = chunk_kda(
                q=q_i,
                k=k_i,
                v=v_i,
                g=g_i,
                beta=beta_i,
                initial_state=h0_i,
                output_final_state=True,
                state_v_first=state_v_first,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                disable_recompute=disable_recompute,
                chunk_size=BT,
            )
        grads = torch.autograd.grad((o * do).sum() + (ht * dht).sum(), inputs)
        return (o, ht, *grads)

    ref = run(False)
    tri = run(True)
    names = ("o", "ht", "dq", "dk", "dv", "dg", "db", "dh0")
    tolerances = (0.01, 0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.01)
    for name, ref_tensor, tri_tensor, tolerance in zip(names, ref, tri, tolerances):
        assert_close(name, ref_tensor, tri_tensor, tolerance)
        assert torch.isfinite(tri_tensor).all()


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(device_platform not in ("cuda", "hip"), reason="requires a CUDA or ROCm GPU")
@pytest.mark.parametrize(
    (
        "K",
        "V",
        "H",
        "HV",
        "BT",
        "gate_mode",
        "use_qk_l2norm_in_kernel",
        "disable_recompute",
        "cu_seqlens_values",
    ),
    [
        pytest.param(128, 128, 64, 64, 64, "precomputed", False, False, [0, 192, 576], id="pregated-mha-k128-ragged"),
        pytest.param(256, 128, 1, 2, 32, "precomputed", True, False, [0, 448], id="pregated-gva-k256-bt32"),
        pytest.param(128, 96, 1, 1, 64, "fused", True, False, [0, 384], id="fused-gate-beta"),
        pytest.param(128, 96, 1, 1, 64, "safe-fused", False, True, [0, 384], id="safe-fused-save-intermediates"),
    ],
)
def test_intracard_kda_training_modes(
    monkeypatch,
    K: int,
    V: int,
    H: int,
    HV: int,
    BT: int,
    gate_mode: str,
    use_qk_l2norm_in_kernel: bool,
    disable_recompute: bool,
    cu_seqlens_values: list[int],
):
    torch.manual_seed(42)
    B, T = 1, cu_seqlens_values[-1]
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, K, device=device, dtype=dtype)
    k = torch.randn(B, T, H, K, device=device, dtype=dtype)
    if not use_qk_l2norm_in_kernel:
        q = torch.nn.functional.normalize(q.float(), dim=-1).to(dtype)
        k = torch.nn.functional.normalize(k.float(), dim=-1).to(dtype)
    v = torch.randn(B, T, HV, V, device=device, dtype=dtype)
    if gate_mode == "precomputed":
        g = -torch.rand(B, T, HV, K, device=device, dtype=torch.float32) * 0.02
        beta = torch.randn(B, T, HV, device=device, dtype=dtype).sigmoid()
        A_log, dt_bias = None, None
    else:
        g = torch.randn(B, T, HV, K, device=device, dtype=dtype) * 0.2 - 4
        beta = torch.randn(B, T, HV, device=device, dtype=dtype)
        A_log = torch.log(torch.empty(HV, device=device, dtype=torch.float32).uniform_(0.5, 1.5))
        dt_bias = torch.randn(HV * K, device=device, dtype=torch.float32) * 0.1
    do = torch.randn_like(v)
    cu_seqlens = torch.tensor(cu_seqlens_values, device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()
    monkeypatch.setattr(intracard_cp_mod, "compute_subseq_len", lambda *args, **kwargs: 2 * BT)
    monkeypatch.setenv("FLA_FLASH_KDA", "0")

    def run(enabled: bool):
        monkeypatch.setenv("FLA_INTRACARD_CP", "1" if enabled else "0")
        inputs = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
        q_i, k_i, v_i, g_i, beta_i = inputs
        gate_kwargs = {}
        if gate_mode != "precomputed":
            A_log_i, dt_bias_i = [x.detach().clone().requires_grad_(True) for x in (A_log, dt_bias)]
            inputs.extend((A_log_i, dt_bias_i))
            gate_kwargs = {
                "A_log": A_log_i,
                "dt_bias": dt_bias_i,
                "use_gate_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "safe_gate": gate_mode == "safe-fused",
                "lower_bound": -5.0 if gate_mode == "safe-fused" else None,
            }
        o, final_state = chunk_kda(
            q=q_i,
            k=k_i,
            v=v_i,
            g=g_i,
            beta=beta_i,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            disable_recompute=disable_recompute,
            chunk_size=BT,
            **gate_kwargs,
        )
        assert final_state is None
        grads = torch.autograd.grad((o * do).sum(), inputs)
        return (o, *grads)

    ref = run(False)
    tri = run(True)
    names = ("o", "dq", "dk", "dv", "dg", "db")
    if gate_mode != "precomputed":
        names += ("dA_log", "ddt_bias")
    for name, ref_tensor, tri_tensor in zip(names, ref, tri):
        tolerance = 0.02 if name in ("dg", "db", "dA_log", "ddt_bias") else 0.01
        assert_close(name, ref_tensor, tri_tensor, tolerance)
        assert torch.isfinite(tri_tensor).all()


@pytest.mark.skipif(os.environ.get("FLA_DISABLE_BACKEND_DISPATCH") == "1", reason="backend dispatch disabled")
@pytest.mark.skipif(device_platform not in ("cuda", "hip"), reason="requires a CUDA or ROCm GPU")
def test_chunk_gdn_intracard_gqa(monkeypatch):
    """E2E: chunk_gated_delta_rule intracard path produces correct results with GQA (Hq < H).

    Uses a long varlen sequence to exercise the intracard split path,
    with Hq=2 key/query heads and H=4 value/output heads.
    """
    import torch.nn.functional as F

    from fla.ops.gated_delta_rule import chunk_gated_delta_rule

    torch.manual_seed(0)
    dtype = torch.bfloat16

    # T must be large enough to bypass early_return in intracard_fwd_h.
    B, T, Hq, H, D = 1, 32768, 2, 4, 64

    q = F.normalize(torch.randn(B, T, Hq, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
    k = F.normalize(torch.randn(B, T, Hq, D, device=device, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, H, device=device, dtype=torch.float32))
    beta = torch.randn(B, T, H, device=device, dtype=torch.float32).sigmoid()

    cu_seqlens = torch.tensor([0, T], device=device, dtype=torch.int32)
    cu_seqlens_cpu = cu_seqlens.cpu()

    # run with the intra-card backend enabled
    with torch.inference_mode():
        o_intra, ht_intra = chunk_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            output_final_state=True,
        )

    # Run without intracard: disable the backend temporarily
    from fla.ops.common.backends import common_registry
    saved_backends = common_registry._backends.copy()
    common_registry._backends.clear()
    try:
        with torch.inference_mode():
            o_ref, ht_ref = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=g, beta=beta,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                output_final_state=True,
            )
    finally:
        common_registry._backends = saved_backends

    assert torch.allclose(o_intra, o_ref, atol=1e-2, rtol=1e-2), \
        f"Output mismatch: max diff={(o_intra - o_ref).abs().max().item()}"
    assert torch.allclose(ht_intra, ht_ref, atol=1e-2, rtol=1e-2), \
        f"Final state mismatch: max diff={(ht_intra - ht_ref).abs().max().item()}"
