# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch

from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
    prepare_chunk_gated_delta_rule_fwd_h_affine,
)
from fla.ops.cp import FLACPContext
from fla.ops.cp.chunk_delta_h import chunk_gated_delta_rule_fwd_h_pre_process, compress_h0
from fla.ops.gla.chunk import chunk_gla_fwd_o_gk
from fla.ops.kda.chunk_intra import chunk_kda_fwd_intra
from fla.ops.kda.gate import kda_gate_chunk_cumsum
from fla.ops.utils import chunk_local_cumsum
from fla.ops.utils.constant import RCP_LN2


def chunk_kda_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_gate_in_kernel: bool = False,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    cp_context: FLACPContext | None = None,
    use_graph: bool = False,
    chunk_offsets: torch.LongTensor | None = None,
):
    # Apply gate activation
    g_org = None
    if use_gate_in_kernel:
        g_org = g
        g = kda_gate_chunk_cumsum(
            g=g_org,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            lower_bound=lower_bound,
            use_graph=use_graph,
        )
    else:
        g = chunk_local_cumsum(
            g=g,
            scale=RCP_LN2,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            use_graph=use_graph,
        )

    # qg = None if disable_recompute is False
    w, u, qg, kg, Aqk, Akk = chunk_kda_fwd_intra(
        q=q,
        k=k,
        v=v,
        gk=g,
        beta=beta,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        safe_gate=safe_gate,
        disable_recompute=disable_recompute,
        use_graph=use_graph,
    )

    intra_affine_summary = None
    flat_boundary_states = None
    cp_world_size = None
    if (
        cp_context is not None
        and cp_context.group is not None
        and cp_context.layout == 'contiguous'
        and not use_graph
        and chunk_offsets is None
    ):
        intra_affine_summary = prepare_chunk_gated_delta_rule_fwd_h_affine(
            k=kg,
            w=w,
            u=u,
            gk=g,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=chunk_size,
        )
        cp_world_size = torch.distributed.get_world_size(group=cp_context.group)
        if intra_affine_summary is not None:
            from fla.ops.common.intracard_cp import (
                materialize_rank_affine_summary,
                merge_flat_affine_summaries,
            )
            use_flat_merge = (
                cp_context.num_seqs == 1
                and cp_context.pre_num_ranks + cp_context.post_num_ranks + 1 == cp_world_size
                and initial_state is None
            )
            if use_flat_merge:
                flat_boundary_states = merge_flat_affine_summaries(
                    intra_affine_summary,
                    group=cp_context.group,
                    state_v_first=state_v_first,
                    use_tf32x3_affine_chain=cp_context.use_tf32x3_affine_chain,
                )
            if flat_boundary_states is None:
                if cp_world_size != 1:
                    intra_affine_summary = materialize_rank_affine_summary(intra_affine_summary)
            else:
                intra_affine_summary = intra_affine_summary._replace(boundary_states=flat_boundary_states)
                initial_state = flat_boundary_states[:1]

    if cp_context is not None and flat_boundary_states is None and cp_world_size != 1:
        initial_state = chunk_gated_delta_rule_fwd_h_pre_process(
            k=kg,
            w=w,
            u=u,
            gk=g,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=cp_context,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
            use_graph=use_graph,
            precomputed_hm=(intra_affine_summary.per_rank if intra_affine_summary is not None else None),
        )

    save_intra_initial_state = cu_seqlens is not None and not disable_recompute and not return_intermediate_states
    intra_summary_kwargs = (
        {"intra_affine_summary": intra_affine_summary} if intra_affine_summary is not None else {}
    )
    state_result = chunk_gated_delta_rule_fwd_h(
        k=kg,
        w=w,
        u=u,
        gk=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        chunk_size=chunk_size,
        state_v_first=state_v_first,
        return_intra_initial_state=save_intra_initial_state,
        **intra_summary_kwargs,
    )
    if save_intra_initial_state:
        h, v_new, final_state, intra_initial_state = state_result
    else:
        h, v_new, final_state = state_result
        intra_initial_state = None

    if cp_context is not None:
        # In Context Parallel (CP) mode, global initial states are not supported at the entry point.
        # The `initial_state` here is computed internally via inter-rank communication.
        # Since only the first sequence in the local batch can be a continuation of a cross-rank sequence,
        # only the first state in the tensor is relevant. We compress it to optimize memory for `save_for_backward`.
        initial_state = compress_h0(initial_state, context=cp_context)

    o = chunk_gla_fwd_o_gk(
        q=q,
        v=v_new,
        g=g,
        A=Aqk,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        chunk_indices=chunk_indices,
        state_v_first=state_v_first,
        use_graph=use_graph,
    )
    if disable_recompute is False:
        # Delete to save memory
        w, u, qg, kg, v_new = None, None, None, None, None
        if not return_intermediate_states:
            h = None
        if use_gate_in_kernel:
            g = None
    return o, final_state, g, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state, intra_initial_state
