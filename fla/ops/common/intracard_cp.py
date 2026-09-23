# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Intra-card context parallelism for shared delta-rule state scans.

Optimized: all CPU-side index computation uses pure Python loops instead of
torch tensor operations (repeat_interleave, arange, cumsum, etc.) to eliminate
per-op overhead on tiny arrays. GPU tensors are created directly from Python
lists to minimize cudaStreamSynchronize calls.
"""

from __future__ import annotations

import logging
import weakref
from collections import OrderedDict
from typing import NamedTuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl

from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64,
    chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
)
from fla.ops.cp.chunk_delta_h import pre_process_bwd_kernel_merged, pre_process_fwd_kernel_merged
from fla.ops.cp.comm import all_gather_into_tensor
from fla.ops.utils.index import prepare_chunk_indices, prepare_chunk_offsets
from fla.utils import IS_TF32_SUPPORTED, autotune_cache_kwargs, get_multiprocessor_count

logger = logging.getLogger(__name__)


# Cache for intra-card state-scan precomputation (Python results + GPU tensors)
# Key: object identity and contents of cu_seqlens plus split configuration
_intracard_cache: OrderedDict[tuple, _CacheEntry] = OrderedDict()
_INTRACARD_CACHE_MAXSIZE = 32
_FLAT_AFFINE_MAX_SUMMARIES = 32


class _CacheEntry(NamedTuple):
    """Cache entry for intra-card state-scan precomputation.

    Caches both Python computation results and GPU tensors to eliminate
    redundant CPU→GPU transfers and Python loop computation.
    """
    # Keep a weak reference to validate id-based key safety.
    # If Python reuses an object id after GC, this guard prevents stale hits.
    cu_seqlens_ref: weakref.ReferenceType[torch.Tensor]
    # From prepare_subseq_cu_seqlens
    cu_seqlens_subseq_values: list[int]
    split_info: SplitSeqInfo
    total_subseqs: int
    # From _precompute_intracard_indices
    cu_seqlens_split_values: list[int]
    S_split_total: int
    non_first_indices: list[int]
    non_last_indices: list[int]
    first_subseq_indices: list[int]
    last_subseq_indices: list[int]
    num_non_first: int
    merge_seq_offsets: list[int]
    merge_init_offsets: list[int]
    # GPU tensors (cached to avoid H2D transfer)
    cu_seqlens_subseq_gpu: torch.Tensor
    cu_seqlens_split_flat: torch.Tensor
    chunk_indices_subseq: torch.Tensor
    chunk_offsets_subseq: torch.Tensor
    non_first_indices_gpu: torch.Tensor
    non_last_indices_gpu: torch.Tensor
    first_subseq_indices_gpu: torch.Tensor
    last_subseq_indices_gpu: torch.Tensor
    merge_seq_offsets_gpu: torch.Tensor
    merge_init_offsets_gpu: torch.Tensor
    split_seq_ids_gpu: torch.Tensor


class IntraCardAffineSummary(NamedTuple):
    """Explicitly prepared affine summaries shared by CP and intra-card scans."""

    cache: _CacheEntry | None
    per_split: torch.Tensor
    per_rank: torch.Tensor | None
    forward: bool
    use_tf32x3_affine_chain: bool
    boundary_states: torch.Tensor | None = None


class SplitSeqInfo(NamedTuple):
    """Information about split sequences (Python lists for zero-overhead access)."""
    split_seq_ids: list[int]       # [num_split_seqs] original sequence indices
    start_subseq_idx: list[int]    # [num_split_seqs] start index in subseq array
    num_subseqs: list[int]         # [num_split_seqs] number of sub-sequences per split

    @property
    def num_split_seqs(self) -> int:
        return len(self.split_seq_ids)

    def __bool__(self) -> bool:
        return self.num_split_seqs > 0


def _raw_chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    B, T, H, K, V, HV = *k.shape, u.shape[-1], u.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        h = k.new_empty(B, NT, HV, V, K)
        final_state = k.new_zeros(N, HV, V, K, dtype=torch.float32) if output_final_state else None
    else:
        h = k.new_empty(B, NT, HV, K, V)
        final_state = k.new_zeros(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta['BV']) * N * HV, )

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        T=T,
        N=N,
        HV=HV,
        H=H,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return h, v_new, final_state


def _raw_chunk_gated_delta_rule_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    B, T, H, K, V, HV = *q.shape, do.shape[-1], do.shape[2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)

    if state_v_first:
        dh = q.new_empty(B, NT, HV, V, K)
    else:
        dh = q.new_empty(B, NT, HV, K, V)
    dh0 = torch.empty_like(h0, dtype=torch.float32) if h0 is not None else None
    dv2 = torch.empty_like(dv)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']) * N * HV, )

    chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        dht=dht,
        dh0=dh0,
        do=do,
        dh=dh,
        dv=dv,
        dv2=dv2,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        N=N,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=BT,
        STATE_V_FIRST=state_v_first,
    )
    return dh, dh0, dv2


def compute_subseq_len(
    seq_len: int,
    num_sms: int,
    num_heads: int,
    chunk_size: int = 64,
) -> int:
    """Compute sub-sequence length for intracard splitting.

    For linear recurrence (fwd_h), the sequential scan is the bottleneck.
    Splitting always reduces the critical path and helps, as long as the
    sequence is long enough to amortize the pre_scan + merge overhead.

    The fwd_h kernel grid is num_v_blocks*N*HV where num_v_blocks ≈ 2.
    Each sub-sequence contributes 2*HV blocks. We target enough splits so
    that even a single long sequence can saturate all SMs.

    A floor on subseq_chunks (MIN_SUBSEQ_CHUNKS) prevents subseq_len from
    being too small, which would cause prepare_subseq_cu_seqlens to
    unnecessarily split shorter sequences in mixed-length batches
    (split threshold = 2 * subseq_len).
    """
    seq_chunks = (seq_len + chunk_size - 1) // chunk_size

    if seq_chunks < 8:
        return seq_len

    # Target splits: saturate SMs with the longest sequence alone.
    # Each sub-seq contributes NUM_V_BLOCKS * num_heads blocks.
    NUM_V_BLOCKS = 2
    target_splits = max(1, num_sms // (NUM_V_BLOCKS * num_heads))

    subseq_chunks = (seq_chunks + target_splits - 1) // target_splits

    # Floor: prevent subseq_len from being too small.
    # With chunk_size=64, MIN_SUBSEQ_CHUNKS=128 → subseq_len >= 8192 tokens,
    # split threshold (2 * subseq_len) = 16384 tokens.
    # Sequences shorter than it won't be split.
    MIN_SUBSEQ_CHUNKS = 128
    subseq_chunks = max(subseq_chunks, MIN_SUBSEQ_CHUNKS)

    return subseq_chunks * chunk_size


def prepare_subseq_cu_seqlens(
    cu_seqlens_cpu: torch.Tensor,
    subseq_len: int,
    chunk_size: int = 64,
    max_splits: int = 32,
) -> tuple[list[int], SplitSeqInfo | bool, int]:
    """Insert subseq split points into original cu_seqlens.

    Optimized: uses pure Python loops instead of torch tensor operations
    for the small index arrays (typically 1-32 elements).

    Returns:
        boundaries: List of cu_seqlens boundaries (can be used directly by _precompute_intracard_indices)
        split_info: SplitSeqInfo for sequences that need splitting, or False if no splitting needed
        total_subseqs: Total number of sub-sequences after splitting
    """
    N = len(cu_seqlens_cpu) - 1
    if N == 0:
        return cu_seqlens_cpu.tolist(), False, 0

    subseq_chunks = (subseq_len + chunk_size - 1) // chunk_size
    threshold_subseq_len = 2 * subseq_len

    split_seq_ids: list[int] = []
    start_subseq_idxs: list[int] = []
    num_subseqs_list: list[int] = []

    # Build boundaries using pure Python loop
    boundaries: list[int] = [0]
    cumsum_offset = 0

    for i in range(N):
        seq_start = int(cu_seqlens_cpu[i].item())
        seq_end = int(cu_seqlens_cpu[i + 1].item())
        seq_len_i = seq_end - seq_start
        seq_chunks_i = (seq_len_i + chunk_size - 1) // chunk_size

        if seq_len_i >= threshold_subseq_len:
            # This sequence needs splitting
            num_ss = min(max_splits, (seq_chunks_i + subseq_chunks - 1) // subseq_chunks)
            chunks_per = (seq_chunks_i + num_ss - 1) // num_ss
            actual_ssl = chunks_per * chunk_size

            split_seq_ids.append(i)
            start_subseq_idxs.append(cumsum_offset)
            num_subseqs_list.append(num_ss)

            for j in range(num_ss):
                boundary = min(seq_start + (j + 1) * actual_ssl, seq_end)
                boundaries.append(boundary)
            cumsum_offset += num_ss
        else:
            # No split needed, single sub-sequence
            boundaries.append(seq_end)
            cumsum_offset += 1

    if not split_seq_ids:
        return cu_seqlens_cpu.tolist(), False, 0

    total_subseqs = cumsum_offset

    split_info = SplitSeqInfo(
        split_seq_ids=split_seq_ids,
        start_subseq_idx=start_subseq_idxs,
        num_subseqs=num_subseqs_list,
    )

    return boundaries, split_info, total_subseqs


def intracard_pre_scan(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    cu_seqlens_subseq_split: torch.Tensor,
    S_split: int,
    chunk_size: int = 64,
    use_tf32x3_affine_chain: bool = False,
):
    H, K, V, HV = kg.shape[2], kg.shape[3], u.shape[3], u.shape[2]
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64

    hm = kg.new_empty(S_split, HV, K, V + K, dtype=torch.float32)
    split_len_hint = triton.cdiv(kg.shape[1], S_split)

    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV, S_split)
    pre_process_fwd_kernel_merged[grid](
        k=kg,
        v=u,
        w=w,
        g=g,
        gk=gk,
        bg=None,
        u=u,
        hm=hm,
        cu_seqlens=cu_seqlens_subseq_split,
        T=split_len_hint,
        N=S_split,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK,
        MULTI_SEQS=True,
        AFFINE_CHAIN_PRECISION=(
            "tf32x3" if use_tf32x3_affine_chain and IS_TF32_SUPPORTED
            else ("ieee" if not IS_TF32_SUPPORTED else None)
        ),
    )

    return hm


def intracard_pre_scan_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None,
    gk: torch.Tensor | None,
    scale: float,
    cu_seqlens_subseq_split: torch.Tensor,
    S_split: int,
    chunk_size: int = 64,
    use_tf32x3_affine_chain: bool = False,
) -> torch.Tensor:
    H, K, V, HV = q.shape[2], q.shape[3], do.shape[3], do.shape[2]
    BK = triton.next_power_of_2(K)
    BLOCK_SIZE = 32 if K <= 64 else 64
    dhm = q.new_empty(S_split, HV, K, V + K, dtype=torch.float32)
    split_len_hint = triton.cdiv(q.shape[1], S_split)

    grid = (triton.cdiv(V, BLOCK_SIZE) + triton.cdiv(K, BLOCK_SIZE), HV, S_split)
    pre_process_bwd_kernel_merged[grid](
        q=q,
        k=k,
        w=w,
        g=g,
        gk=gk,
        do=do,
        dhm=dhm,
        dv=dv,
        cu_seqlens=cu_seqlens_subseq_split,
        scale=scale,
        T=split_len_hint,
        N=S_split,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BT=chunk_size,
        BLOCK_SIZE=BLOCK_SIZE,
        BK1=BK,
        USE_BG=False,
        MULTI_SEQS=True,
        AFFINE_CHAIN_PRECISION=(
            "tf32x3" if use_tf32x3_affine_chain and IS_TF32_SUPPORTED
            else ("ieee" if not IS_TF32_SUPPORTED else None)
        ),
    )
    return dhm


@triton.autotune(
    configs=[
        triton.Config({'BC': BC}, num_warps=num_warps, num_stages=num_stages)
        for BC in [32, 64]
        for num_warps in [4, 8]
        for num_stages in [1, 2]
    ],
    key=['HV', 'K', 'V', 'NUM_SUMMARIES', 'FORWARD', 'AFFINE_CHAIN_PRECISION'],
    **autotune_cache_kwargs,
)
@triton.jit
def compose_affine_summaries_kernel(
    hm,
    rank_hm,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BC: tl.constexpr,
    NUM_SUMMARIES: tl.constexpr,
    FORWARD: tl.constexpr,
    AFFINE_CHAIN_PRECISION: tl.constexpr,
):
    """Compose per-split ``[E | M]`` transforms."""
    i_c = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_k = tl.arange(0, BK)
    o_c = i_c * BC + tl.arange(0, BC)
    m_k = o_k < K
    m_c = o_c < V + K

    stride_s = HV * K * (V + K)
    stride_h = K * (V + K)
    first_s = 0 if FORWARD else NUM_SUMMARIES - 1
    first_base = first_s * stride_s + i_h * stride_h
    p_first = hm + first_base + o_k[:, None] * (V + K) + o_c[None, :]
    b_affine = tl.load(p_first, mask=m_k[:, None] & m_c[None, :], other=0.0).to(tl.float32)

    for idx in range(1, NUM_SUMMARIES):
        i_s = idx if FORWARD else NUM_SUMMARIES - 1 - idx
        base = i_s * stride_s + i_h * stride_h
        p_m = hm + base + V + o_k[:, None] * (V + K) + o_k[None, :]
        b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_affine = tl.dot(b_m, b_affine, input_precision=AFFINE_CHAIN_PRECISION)

        p_e = hm + base + o_k[:, None] * (V + K) + o_c[None, :]
        b_e = tl.load(p_e, mask=m_k[:, None] & (o_c[None, :] < V), other=0.0).to(tl.float32)
        b_affine += b_e

    p_out = rank_hm + i_h * stride_h + o_k[:, None] * (V + K) + o_c[None, :]
    tl.store(p_out, b_affine, mask=m_k[:, None] & m_c[None, :])


def compose_affine_summaries(
    hm: torch.Tensor,
    *,
    forward: bool,
    use_tf32x3_affine_chain: bool = False,
) -> torch.Tensor:
    """Return the affine composition of a single sequence's split summaries."""
    num_summaries, HV, K, width = hm.shape
    V = width - K
    if num_summaries == 1:
        return hm[0]
    if not 16 <= K <= 128:
        raise ValueError(f"affine-summary composition supports 16 <= K <= 128, got K={K}")

    rank_hm = hm.new_empty(HV, K, V + K)
    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (triton.cdiv(V + K, meta['BC']), HV)

    compose_affine_summaries_kernel[grid](
        hm=hm,
        rank_hm=rank_hm,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        NUM_SUMMARIES=num_summaries,
        FORWARD=forward,
        AFFINE_CHAIN_PRECISION=("tf32x3" if use_tf32x3_affine_chain and IS_TF32_SUPPORTED else "ieee"),
    )
    return rank_hm


@triton.autotune(
    configs=[
        triton.Config({'BV': BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [2, 3]
    ],
    key=['HV', 'K', 'V', 'NUM_GLOBAL_SUMMARIES', 'FORWARD'],
    **autotune_cache_kwargs,
)
@triton.jit
def merge_flat_affine_summaries_kernel(
    boundary_states,
    ag_hm,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NUM_LOCAL_SUMMARIES: tl.constexpr,
    NUM_GLOBAL_SUMMARIES: tl.constexpr,
    RANK: tl.constexpr,
    FORWARD: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    AFFINE_CHAIN_PRECISION: tl.constexpr,
):
    i_v = tl.program_id(0).to(tl.int64)
    i_h = tl.program_id(1).to(tl.int64)
    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    stride_s = HV * K * (V + K)
    stride_h = K * (V + K)
    local_start = RANK * NUM_LOCAL_SUMMARIES
    local_end = local_start + NUM_LOCAL_SUMMARIES

    for step in range(NUM_GLOBAL_SUMMARIES):
        i_s = step if FORWARD else NUM_GLOBAL_SUMMARIES - 1 - step
        if i_s >= local_start:
            if i_s < local_end:
                i_local = i_s - local_start
                if STATE_V_FIRST:
                    p_out = boundary_states + (i_local * HV + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
                    tl.store(p_out, tl.trans(b_h), mask=m_v[:, None] & m_k[None, :])
                else:
                    p_out = boundary_states + (i_local * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
                    tl.store(p_out, b_h, mask=m_k[:, None] & m_v[None, :])

        if step < NUM_GLOBAL_SUMMARIES - 1:
            base = i_s * stride_s + i_h * stride_h
            p_he = ag_hm + base + o_k[:, None] * (V + K) + o_v[None, :]
            b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            p_m = ag_hm + base + V + o_k[:, None] * (V + K) + o_k[None, :]
            b_m = tl.load(p_m, mask=m_k[:, None] & m_k[None, :], other=0.0).to(tl.float32)
            b_h = tl.dot(b_m, b_h, input_precision=AFFINE_CHAIN_PRECISION) + b_he


def merge_flat_affine_summaries(
    summary: IntraCardAffineSummary,
    *,
    group: dist.ProcessGroup,
    state_v_first: bool,
    use_tf32x3_affine_chain: bool,
) -> torch.Tensor | None:
    """Return local boundary states from one global affine chain."""
    hm = summary.per_split
    num_local_summaries, HV, K, width = hm.shape
    V = width - K
    world_size = dist.get_world_size(group=group)
    num_global_summaries = world_size * num_local_summaries
    if num_global_summaries > _FLAT_AFFINE_MAX_SUMMARIES:
        return None

    if world_size == 1:
        ag_hm = hm
    else:
        gathered_hm, _ = all_gather_into_tensor(hm, group=group)
        ag_hm = gathered_hm.flatten(0, 1)

    if state_v_first:
        boundary_states = hm.new_empty(num_local_summaries, HV, V, K)
    else:
        boundary_states = hm.new_empty(num_local_summaries, HV, K, V)
    BK = triton.next_power_of_2(K)
    rank = dist.get_rank(group=group)

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), HV)

    merge_flat_affine_summaries_kernel[grid](
        boundary_states=boundary_states,
        ag_hm=ag_hm,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        NUM_LOCAL_SUMMARIES=num_local_summaries,
        NUM_GLOBAL_SUMMARIES=num_global_summaries,
        RANK=rank,
        FORWARD=summary.forward,
        STATE_V_FIRST=state_v_first,
        AFFINE_CHAIN_PRECISION=(
            "tf32x3" if use_tf32x3_affine_chain and IS_TF32_SUPPORTED
            else ("ieee" if not IS_TF32_SUPPORTED else None)
        ),
    )
    return boundary_states


def materialize_rank_affine_summary(summary: IntraCardAffineSummary) -> IntraCardAffineSummary:
    """Compose a rank summary when the flat merge is not applicable."""
    if summary.per_rank is not None:
        return summary
    per_rank = compose_affine_summaries(
        summary.per_split,
        forward=summary.forward,
        use_tf32x3_affine_chain=summary.use_tf32x3_affine_chain,
    )
    return summary._replace(per_rank=per_rank)


@triton.jit(do_not_specialize=['STEP', 'NUM_SEQ_ENTRIES'])
def intracard_merge_kernel_tiled_step(
    h,
    ag_hm,
    seq_offsets,
    init_offsets,
    h0_seq_ids,
    h0,
    STEP,
    NUM_SEQ_ENTRIES,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    BI: tl.constexpr,
    FORWARD: tl.constexpr,
    HAS_H0: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    AFFINE_CHAIN_PRECISION: tl.constexpr,
):
    i_kv = tl.program_id(0).to(tl.int64)
    i_seq = tl.program_id(1).to(tl.int64)
    i_h = tl.program_id(2).to(tl.int64)
    NV: tl.constexpr = tl.cdiv(V, BV)
    i_k, i_v = i_kv // NV, i_kv % NV

    if i_seq >= NUM_SEQ_ENTRIES:
        return

    ss_start = tl.load(seq_offsets + i_seq).to(tl.int64)
    ss_end = tl.load(seq_offsets + i_seq + 1).to(tl.int64)
    init_base = tl.load(init_offsets + i_seq).to(tl.int64)
    num_subseqs = ss_end - ss_start
    step = STEP.to(tl.int64)
    if step >= num_subseqs - 1:
        return

    if FORWARD:
        i_ss = ss_start + step
        out_idx = init_base + step
        prev_idx = out_idx - 1
    else:
        i_ss = ss_end - 1 - step
        out_idx = init_base + num_subseqs - 2 - step
        prev_idx = out_idx + 1

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    m_k = o_k < K
    m_v = o_v < V
    stride_hm_s = HV * K * (V + K)
    stride_hm_h = K * (V + K)
    base = i_ss * stride_hm_s + i_h * stride_hm_h

    p_he = ag_hm + base + o_k[:, None] * (V + K) + o_v[None, :]
    b_he = tl.load(p_he, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    b_out = tl.zeros([BK, BV], dtype=tl.float32)

    for i_i in range(tl.cdiv(K, BI)):
        o_i = i_i * BI + tl.arange(0, BI)
        m_i = o_i < K
        p_m = ag_hm + base + V + o_k[:, None] * (V + K) + o_i[None, :]
        b_m = tl.load(p_m, mask=m_k[:, None] & m_i[None, :], other=0.0).to(tl.float32)

        if step == 0:
            if HAS_H0:
                orig_seq_id = tl.load(h0_seq_ids + i_seq).to(tl.int64)
                if STATE_V_FIRST:
                    p_prev = h0 + (orig_seq_id * HV + i_h) * V * K + o_v[None, :] * K + o_i[:, None]
                else:
                    p_prev = h0 + (orig_seq_id * HV + i_h) * K * V + o_i[:, None] * V + o_v[None, :]
                b_prev = tl.load(p_prev, mask=m_i[:, None] & m_v[None, :], other=0.0).to(tl.float32)
            else:
                b_prev = tl.zeros([BI, BV], dtype=tl.float32)
        else:
            if STATE_V_FIRST:
                p_prev = h + (prev_idx * HV + i_h) * V * K + o_v[None, :] * K + o_i[:, None]
            else:
                p_prev = h + (prev_idx * HV + i_h) * K * V + o_i[:, None] * V + o_v[None, :]
            b_prev = tl.load(p_prev, mask=m_i[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        b_out = tl.dot(b_m, b_prev, b_out, input_precision=AFFINE_CHAIN_PRECISION)

    b_out += b_he
    if STATE_V_FIRST:
        p_out = h + (out_idx * HV + i_h) * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_out, tl.trans(b_out), mask=m_v[:, None] & m_k[None, :])
    else:
        p_out = h + (out_idx * HV + i_h) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_out, b_out, mask=m_k[:, None] & m_v[None, :])


def intracard_merge(
    hm: torch.Tensor,
    split_info: SplitSeqInfo,
    num_non_first: int,
    merge_seq_offsets: list[int],
    merge_init_offsets: list[int],
    device: torch.device,
    initial_state: torch.Tensor | None = None,
    state_v_first: bool = False,
    use_tf32x3_affine_chain: bool = False,
    forward: bool = True,
    seq_offsets: torch.Tensor | None = None,
    init_offsets: torch.Tensor | None = None,
    h0_seq_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, int]:
    """Merge sub-sequence boundary states using pre-computed parameters.

    All CPU-side preparation (cumsum, offset lists) is done in the caller
    using pure Python loops. Cached callers provide GPU metadata tensors;
    direct callers materialize them here before launching the merge kernel.
    """
    from fla.ops.cp.chunk_delta_h import merge_fwd_bwd_kernel

    if num_non_first == 0:
        return None, 0

    HV = hm.shape[1]
    K = hm.shape[2]
    V = hm.shape[3] - K
    BK = triton.next_power_of_2(K)

    num_split_seqs = split_info.num_split_seqs

    if seq_offsets is None or init_offsets is None or h0_seq_ids is None:
        all_int_data = merge_seq_offsets + merge_init_offsets + split_info.split_seq_ids
        all_tensor = torch.tensor(all_int_data, dtype=torch.int32, device=device)
        n_so = len(merge_seq_offsets)
        n_io = len(merge_init_offsets)
        seq_offsets = all_tensor[:n_so]
        init_offsets = all_tensor[n_so:n_so+n_io]
        h0_seq_ids = all_tensor[n_so+n_io:]

    if state_v_first:
        boundary_states_merge = hm.new_empty(num_non_first, HV, V, K, dtype=torch.float32)
    else:
        boundary_states_merge = hm.new_empty(num_non_first, HV, K, V, dtype=torch.float32)

    affine_chain_precision = (
        "tf32x3" if use_tf32x3_affine_chain and IS_TF32_SUPPORTED
        else ("ieee" if not IS_TF32_SUPPORTED else None)
    )

    if BK > 128:
        BK_TILE = 32
        BV_TILE = 32
        BI_TILE = 32
        grid = (triton.cdiv(K, BK_TILE) * triton.cdiv(V, BV_TILE), num_split_seqs, HV)
        for step in range(max(split_info.num_subseqs) - 1):
            intracard_merge_kernel_tiled_step[grid](
                h=boundary_states_merge,
                ag_hm=hm,
                seq_offsets=seq_offsets,
                init_offsets=init_offsets,
                h0_seq_ids=h0_seq_ids,
                h0=initial_state,
                STEP=step,
                NUM_SEQ_ENTRIES=num_split_seqs,
                HV=HV,
                K=K,
                V=V,
                BK=BK_TILE,
                BV=BV_TILE,
                BI=BI_TILE,
                FORWARD=forward,
                HAS_H0=initial_state is not None,
                STATE_V_FIRST=state_v_first,
                AFFINE_CHAIN_PRECISION=affine_chain_precision,
                num_warps=4,
                num_stages=2,
            )
        return boundary_states_merge, num_non_first

    def grid(meta):
        return (triton.cdiv(V, meta['BV']), num_split_seqs, HV)

    merge_fwd_bwd_kernel[grid](
        h=boundary_states_merge,
        ag_hm=hm,
        pre_or_post_num_ranks=num_split_seqs,
        rank=0,
        seq_offsets=seq_offsets,
        init_offsets=init_offsets,
        h0_seq_ids=h0_seq_ids,
        h0=initial_state,
        h_seq_idx=None,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        FORWARD=forward,
        INTRACARD_MODE=True,
        NUM_SEQ_ENTRIES=num_split_seqs,
        STATE_V_FIRST=state_v_first,
        AFFINE_CHAIN_PRECISION=affine_chain_precision,
    )

    return boundary_states_merge, num_non_first


def _precompute_intracard_indices(
    split_info: SplitSeqInfo,
    cu_seqlens_subseq_values: list[int],
    N_orig: int,
) -> tuple[list[int], int, list[int], list[int], list[int], list[int], int, list[int], list[int]]:
    """Pre-compute all derived indices using pure Python loops.

    Returns:
        cu_seqlens_split_values: flattened cu_seqlens boundaries for split seqs (for pre_scan)
        S_split_total: total number of sub-sequences from splits
        non_first_indices: indices for scattering merge results into initial_state_expanded
        non_last_indices: indices for scattering merge results into dht_expanded
        first_subseq_indices: indices of first sub-sequence for each original sequence
        last_subseq_indices: indices of last sub-sequence for each original sequence
        num_non_first: total non-first sub-sequences (merge work)
        merge_seq_offsets: cumulative sub-sequence counts for merge kernel
        merge_init_offsets: cumulative non-first counts for merge kernel
    """
    starts = split_info.start_subseq_idx
    num_ss = split_info.num_subseqs
    split_ids = split_info.split_seq_ids

    # store explicit pairs because split sequences need not be adjacent in the packed input
    cu_seqlens_split_values: list[int] = []
    S_split_total = 0
    for s, n in zip(starts, num_ss):
        for j in range(n):
            cu_seqlens_split_values.extend((cu_seqlens_subseq_values[s + j], cu_seqlens_subseq_values[s + j + 1]))
        S_split_total += n

    # num_subseqs_per_seq: [N_orig], default 1 for unsplit sequences
    num_subseqs_per_seq = [1] * N_orig
    for sid, nss in zip(split_ids, num_ss):
        num_subseqs_per_seq[sid] = nss

    # non_first_indices: for scattering merged initial states
    non_first_indices: list[int] = []
    for s, n in zip(starts, num_ss):
        for j in range(1, n):
            non_first_indices.append(s + j)

    non_last_indices: list[int] = []
    for s, n in zip(starts, num_ss):
        for j in range(n - 1):
            non_last_indices.append(s + j)

    # first_subseq_indices: for scattering original initial states
    first_subseq_indices: list[int] = [0]
    running = 0
    for i in range(N_orig - 1):
        running += num_subseqs_per_seq[i]
        first_subseq_indices.append(running)

    # last_subseq_indices: for gathering final states
    last_subseq_indices: list[int] = []
    running = 0
    for n in num_subseqs_per_seq:
        running += n
        last_subseq_indices.append(running - 1)

    # merge parameters
    merge_seq_offsets: list[int] = [0]
    merge_init_offsets: list[int] = [0]
    for n in num_ss:
        merge_seq_offsets.append(merge_seq_offsets[-1] + n)
        merge_init_offsets.append(merge_init_offsets[-1] + n - 1)
    num_non_first = merge_init_offsets[-1]

    return (
        cu_seqlens_split_values,
        S_split_total,
        non_first_indices,
        non_last_indices,
        first_subseq_indices,
        last_subseq_indices,
        num_non_first,
        merge_seq_offsets,
        merge_init_offsets,
    )


def _prepare_intracard_cache_entry(
    cu_seqlens: torch.LongTensor,
    cu_seqlens_cpu: torch.LongTensor | None,
    num_heads: int,
    chunk_size: int,
    max_splits: int,
    device: torch.device,
) -> _CacheEntry | None:
    if cu_seqlens_cpu is None:
        cu_seqlens_cpu = cu_seqlens.cpu()

    seq_lens = torch.diff(cu_seqlens_cpu)
    max_seq_len = int(seq_lens.max().item())
    subseq_len = compute_subseq_len(max_seq_len, get_multiprocessor_count(), num_heads, chunk_size)
    if (seq_lens < 2 * subseq_len).all():
        return None

    cu_seqlens_fingerprint = tuple(cu_seqlens_cpu.tolist())
    cache_key = (id(cu_seqlens), cu_seqlens_fingerprint, subseq_len, chunk_size, max_splits, str(device))
    cached = _intracard_cache.get(cache_key)
    if cached is not None:
        if cached.cu_seqlens_ref() is cu_seqlens:
            _intracard_cache.move_to_end(cache_key)
            return cached
        _intracard_cache.pop(cache_key, None)

    cu_seqlens_subseq_values, split_info, total_subseqs = prepare_subseq_cu_seqlens(
        cu_seqlens_cpu,
        subseq_len,
        chunk_size,
        max_splits=max_splits,
    )
    if not split_info:
        return None

    (
        cu_seqlens_split_values,
        S_split_total,
        non_first_indices,
        non_last_indices,
        first_subseq_indices,
        last_subseq_indices,
        num_non_first,
        merge_seq_offsets,
        merge_init_offsets,
    ) = _precompute_intracard_indices(split_info, cu_seqlens_subseq_values, len(cu_seqlens_cpu) - 1)

    dtype = cu_seqlens_cpu.dtype
    cu_seqlens_subseq_gpu = torch.tensor(cu_seqlens_subseq_values, dtype=dtype, device=device)
    cached = _CacheEntry(
        cu_seqlens_ref=weakref.ref(cu_seqlens),
        cu_seqlens_subseq_values=cu_seqlens_subseq_values,
        split_info=split_info,
        total_subseqs=total_subseqs,
        cu_seqlens_split_values=cu_seqlens_split_values,
        S_split_total=S_split_total,
        non_first_indices=non_first_indices,
        non_last_indices=non_last_indices,
        first_subseq_indices=first_subseq_indices,
        last_subseq_indices=last_subseq_indices,
        num_non_first=num_non_first,
        merge_seq_offsets=merge_seq_offsets,
        merge_init_offsets=merge_init_offsets,
        cu_seqlens_subseq_gpu=cu_seqlens_subseq_gpu,
        cu_seqlens_split_flat=torch.tensor(cu_seqlens_split_values, dtype=dtype, device=device),
        chunk_indices_subseq=prepare_chunk_indices(cu_seqlens_subseq_gpu, chunk_size),
        chunk_offsets_subseq=prepare_chunk_offsets(cu_seqlens_subseq_gpu, chunk_size),
        non_first_indices_gpu=torch.tensor(non_first_indices, dtype=torch.long, device=device),
        non_last_indices_gpu=torch.tensor(non_last_indices, dtype=torch.long, device=device),
        first_subseq_indices_gpu=torch.tensor(first_subseq_indices, dtype=torch.long, device=device),
        last_subseq_indices_gpu=torch.tensor(last_subseq_indices, dtype=torch.long, device=device),
        merge_seq_offsets_gpu=torch.tensor(merge_seq_offsets, dtype=torch.int32, device=device),
        merge_init_offsets_gpu=torch.tensor(merge_init_offsets, dtype=torch.int32, device=device),
        split_seq_ids_gpu=torch.tensor(split_info.split_seq_ids, dtype=torch.int32, device=device),
    )
    _intracard_cache[cache_key] = cached
    while len(_intracard_cache) > _INTRACARD_CACHE_MAXSIZE:
        _intracard_cache.popitem(last=False)
    return cached


def prepare_intracard_fwd_affine_summary(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_size: int = 64,
    max_splits: int = 32,
    use_tf32x3_affine_chain: bool = False,
) -> IntraCardAffineSummary | None:
    """Prepare per-split summaries once for a single contiguous CP sequence."""
    K = k.shape[-1]
    if cu_seqlens is None or cu_seqlens.numel() != 2 or not 16 <= K <= 128:
        return None

    cached = _prepare_intracard_cache_entry(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        num_heads=u.shape[2],
        chunk_size=chunk_size,
        max_splits=max_splits,
        device=k.device,
    )
    if cached is not None and cached.split_info.num_split_seqs != 1:
        return None

    split_cu_seqlens = cu_seqlens if cached is None else cached.cu_seqlens_split_flat
    num_summaries = 1 if cached is None else cached.S_split_total

    hm = intracard_pre_scan(
        kg=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        cu_seqlens_subseq_split=split_cu_seqlens,
        S_split=num_summaries,
        chunk_size=chunk_size,
        use_tf32x3_affine_chain=use_tf32x3_affine_chain,
    )
    return IntraCardAffineSummary(
        cache=cached,
        per_split=hm,
        per_rank=hm[0] if num_summaries == 1 else None,
        forward=True,
        use_tf32x3_affine_chain=use_tf32x3_affine_chain,
    )


def prepare_intracard_bwd_affine_summary(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_size: int = 64,
    max_splits: int = 32,
    use_tf32x3_affine_chain: bool = False,
) -> IntraCardAffineSummary | None:
    """Prepare per-split backward summaries once for CP and the local reverse scan."""
    K = q.shape[-1]
    if scale is None or cu_seqlens is None or cu_seqlens.numel() != 2 or not 16 <= K <= 128:
        return None

    cached = _prepare_intracard_cache_entry(
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        num_heads=do.shape[2],
        chunk_size=chunk_size,
        max_splits=max_splits,
        device=q.device,
    )
    if cached is not None and cached.split_info.num_split_seqs != 1:
        return None

    split_cu_seqlens = cu_seqlens if cached is None else cached.cu_seqlens_split_flat
    num_summaries = 1 if cached is None else cached.S_split_total

    dhm = intracard_pre_scan_bwd(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        scale=scale,
        cu_seqlens_subseq_split=split_cu_seqlens,
        S_split=num_summaries,
        chunk_size=chunk_size,
        use_tf32x3_affine_chain=use_tf32x3_affine_chain,
    )
    return IntraCardAffineSummary(
        cache=cached,
        per_split=dhm,
        per_rank=dhm[0] if num_summaries == 1 else None,
        forward=False,
        use_tf32x3_affine_chain=use_tf32x3_affine_chain,
    )


def intracard_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    max_splits: int = 32,
    use_tf32x3_affine_chain: bool = False,
    return_intra_initial_state: bool = False,
    intra_initial_state: torch.Tensor | None = None,
    intra_affine_summary: IntraCardAffineSummary | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    assert cu_seqlens is not None, "intracard_fwd_h requires cu_seqlens"

    K = k.shape[-1]
    assert K <= 256, "current kernel does not support key head dimensions larger than 256"
    V = u.shape[-1]
    HV = u.shape[2]
    device = k.device
    if intra_affine_summary is not None:
        if not intra_affine_summary.forward:
            raise ValueError("forward state scan received a backward affine summary")
        cached = intra_affine_summary.cache
    else:
        cached = _prepare_intracard_cache_entry(
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            num_heads=HV,
            chunk_size=chunk_size,
            max_splits=max_splits,
            device=device,
        )
    if cached is None:
        prepared_initial_state = (
            intra_affine_summary.boundary_states
            if intra_affine_summary is not None and intra_affine_summary.boundary_states is not None
            else initial_state
        )
        result = _raw_chunk_gated_delta_rule_fwd_h(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            initial_state=prepared_initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            state_v_first=state_v_first,
        )
        return (*result, prepared_initial_state) if return_intra_initial_state else result

    if state_v_first:
        state_shape = (cached.total_subseqs, HV, V, K)
    else:
        state_shape = (cached.total_subseqs, HV, K, V)

    prepared_initial_state = intra_initial_state
    if prepared_initial_state is None and intra_affine_summary is not None:
        prepared_initial_state = intra_affine_summary.boundary_states

    if prepared_initial_state is None:
        if intra_affine_summary is not None:
            hm = intra_affine_summary.per_split
        else:
            hm = intracard_pre_scan(
                kg=k,
                w=w,
                u=u,
                g=g,
                gk=gk,
                cu_seqlens_subseq_split=cached.cu_seqlens_split_flat,
                S_split=cached.S_split_total,
                chunk_size=chunk_size,
                use_tf32x3_affine_chain=use_tf32x3_affine_chain,
            )

        initial_states_merge, num_non_first = intracard_merge(
            hm=hm,
            split_info=cached.split_info,
            num_non_first=cached.num_non_first,
            merge_seq_offsets=cached.merge_seq_offsets,
            merge_init_offsets=cached.merge_init_offsets,
            device=device,
            initial_state=initial_state,
            state_v_first=state_v_first,
            use_tf32x3_affine_chain=use_tf32x3_affine_chain,
            seq_offsets=cached.merge_seq_offsets_gpu,
            init_offsets=cached.merge_init_offsets_gpu,
            h0_seq_ids=cached.split_seq_ids_gpu,
        )

        initial_state_expanded = k.new_zeros(state_shape, dtype=torch.float32)

        if initial_state is not None:
            initial_state_expanded[cached.first_subseq_indices_gpu] = initial_state

        if initial_states_merge is not None and num_non_first > 0:
            initial_state_expanded[cached.non_first_indices_gpu] = initial_states_merge
    else:
        if prepared_initial_state.shape != state_shape:
            raise ValueError(
                f"intra_initial_state must have shape {state_shape}, got {tuple(prepared_initial_state.shape)}"
            )
        if prepared_initial_state.dtype != torch.float32:
            raise ValueError(
                f"intra_initial_state must have dtype torch.float32, got {prepared_initial_state.dtype}"
            )
        if prepared_initial_state.device != device:
            raise ValueError(f"intra_initial_state must be on {device}, got {prepared_initial_state.device}")
        initial_state_expanded = prepared_initial_state

    h, v_new, final_state_subseq = _raw_chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        gk=gk,
        initial_state=initial_state_expanded,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
        save_new_value=save_new_value,
        cu_seqlens=cached.cu_seqlens_subseq_gpu,
        chunk_indices=cached.chunk_indices_subseq,
        chunk_offsets=cached.chunk_offsets_subseq,
        state_v_first=state_v_first,
    )

    if output_final_state and final_state_subseq is not None:
        final_state = final_state_subseq[cached.last_subseq_indices_gpu]
    else:
        final_state = final_state_subseq

    result = h, v_new, final_state
    return (*result, initial_state_expanded) if return_intra_initial_state else result


def intracard_bwd_dhu(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    h0: torch.Tensor | None = None,
    dht: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.LongTensor | None = None,
    chunk_offsets: torch.LongTensor | None = None,
    max_splits: int = 32,
    use_tf32x3_affine_chain: bool = False,
    intra_affine_summary: IntraCardAffineSummary | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    assert cu_seqlens is not None, "intracard_bwd_dhu requires cu_seqlens"
    assert scale is not None, "intracard_bwd_dhu requires scale"

    K = q.shape[-1]
    assert K <= 256, "current kernel does not support key head dimensions larger than 256"
    V = do.shape[-1]
    HV = do.shape[2]
    if intra_affine_summary is not None:
        if intra_affine_summary.forward:
            raise ValueError("backward state scan received a forward affine summary")
        cached = intra_affine_summary.cache
    else:
        cached = _prepare_intracard_cache_entry(
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            num_heads=HV,
            chunk_size=chunk_size,
            max_splits=max_splits,
            device=q.device,
        )
    if cached is None:
        prepared_dht = (
            intra_affine_summary.boundary_states
            if intra_affine_summary is not None and intra_affine_summary.boundary_states is not None
            else dht
        )
        return _raw_chunk_gated_delta_rule_bwd_dhu(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            h0=h0,
            dht=prepared_dht,
            scale=scale,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
        )

    if state_v_first:
        state_shape = (cached.total_subseqs, HV, V, K)
    else:
        state_shape = (cached.total_subseqs, HV, K, V)
    prepared_dht = intra_affine_summary.boundary_states if intra_affine_summary is not None else None
    if prepared_dht is not None:
        if prepared_dht.shape != state_shape:
            raise ValueError(f"intra dht must have shape {state_shape}, got {tuple(prepared_dht.shape)}")
        if prepared_dht.dtype != torch.float32 or prepared_dht.device != q.device:
            raise ValueError("intra dht must be an fp32 tensor on the same device as q")
        dht_expanded = prepared_dht
    else:
        if intra_affine_summary is not None:
            dhm = intra_affine_summary.per_split
        else:
            dhm = intracard_pre_scan_bwd(
                q=q,
                k=k,
                w=w,
                do=do,
                dv=dv,
                g=g,
                gk=gk,
                scale=scale,
                cu_seqlens_subseq_split=cached.cu_seqlens_split_flat,
                S_split=cached.S_split_total,
                chunk_size=chunk_size,
                use_tf32x3_affine_chain=use_tf32x3_affine_chain,
            )
        dht_merge, num_non_first = intracard_merge(
            hm=dhm,
            split_info=cached.split_info,
            num_non_first=cached.num_non_first,
            merge_seq_offsets=cached.merge_seq_offsets,
            merge_init_offsets=cached.merge_init_offsets,
            device=q.device,
            initial_state=dht,
            state_v_first=state_v_first,
            use_tf32x3_affine_chain=use_tf32x3_affine_chain,
            forward=False,
            seq_offsets=cached.merge_seq_offsets_gpu,
            init_offsets=cached.merge_init_offsets_gpu,
            h0_seq_ids=cached.split_seq_ids_gpu,
        )
        dht_expanded = q.new_zeros(state_shape, dtype=torch.float32)
        if dht is not None:
            dht_expanded[cached.last_subseq_indices_gpu] = dht
        if dht_merge is not None and num_non_first > 0:
            dht_expanded[cached.non_last_indices_gpu] = dht_merge

    h0_expanded = q.new_empty(state_shape, dtype=torch.float32) if h0 is not None else None
    dh, dh0_subseq, dv2 = _raw_chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        do=do,
        dv=dv,
        g=g,
        gk=gk,
        h0=h0_expanded,
        dht=dht_expanded,
        scale=scale,
        state_v_first=state_v_first,
        cu_seqlens=cached.cu_seqlens_subseq_gpu,
        chunk_size=chunk_size,
        chunk_indices=cached.chunk_indices_subseq,
        chunk_offsets=cached.chunk_offsets_subseq,
    )
    dh0 = dh0_subseq[cached.first_subseq_indices_gpu] if dh0_subseq is not None else None
    return dh, dh0, dv2
