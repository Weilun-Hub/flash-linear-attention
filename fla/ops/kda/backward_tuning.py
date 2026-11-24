# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Opt-in KDA backward launch configurations."""

import os


def launch_config(name: str, blocks: tuple[int, ...], warps: tuple[int, ...]):
    raw = os.environ.get(name, '')
    if not raw:
        return None
    try:
        block, warp, stage = map(int, raw.split(','))
    except ValueError as exc:
        raise ValueError(f'{name} must be BLOCK,WARPS,STAGES') from exc
    if block not in blocks or warp not in warps or stage not in (1, 2, 3, 4):
        raise ValueError(f'{name}: blocks={blocks}, warps={warps}, stages=1/2/3/4')
    return block, warp, stage


def fixed_kernel(kernel):
    """Select the JIT kernel for an explicit launch configuration."""
    from triton.runtime import JITFunction

    while not isinstance(kernel, JITFunction):
        kernel = kernel.fn
    return kernel


def reconstruct_state(**kwargs):
    """Tune only KDA backward reconstruction; preserve shared forward dispatch."""
    import torch
    import triton

    from fla.ops.common.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
        chunk_gated_delta_rule_fwd_kernel_h_blockdim64,
    )
    from fla.ops.utils import prepare_chunk_indices, prepare_chunk_offsets
    from fla.utils import IS_NVIDIA

    config = launch_config('FLA_KDA_RECOMPUTE_H_TILE', (16, 32, 64), (2,))
    k, w, u = kwargs['k'], kwargs['w'], kwargs['u']
    chunk_size = kwargs.get('chunk_size', 64)
    if (config is None or not IS_NVIDIA or k.shape[-1] != 128 or u.shape[-1] != 128
            or chunk_size != 64 or kwargs.get('intra_initial_state') is not None):
        return chunk_gated_delta_rule_fwd_h(**kwargs)
    bv, warps, stages = config
    b, t, h, dim = k.shape
    hv, v = u.shape[2:]
    cu = kwargs.get('cu_seqlens')
    offsets = kwargs.get('chunk_offsets')
    if cu is None:
        n, nt = b, triton.cdiv(t, chunk_size)
    else:
        indices = kwargs.get('chunk_indices')
        if indices is None:
            indices = prepare_chunk_indices(cu, chunk_size)
        n, nt = len(cu) - 1, len(indices)
        if offsets is None:
            offsets = prepare_chunk_offsets(cu, chunk_size)
    vf = kwargs.get('state_v_first', False)
    shape = (v, dim) if vf else (dim, v)
    states = k.new_empty(b, nt, hv, *shape)
    final = k.new_zeros(n, hv, *shape, dtype=torch.float32) if kwargs.get('output_final_state', False) else None
    values = torch.empty_like(u)
    initial = kwargs.get('initial_state')
    gate = kwargs.get('gk')
    fixed_kernel(chunk_gated_delta_rule_fwd_kernel_h_blockdim64)[(triton.cdiv(v, bv) * n * hv,)](
        k=k, v=u, w=w, v_new=values, g=None, gk=gate, h=states, h0=initial, ht=final,
        cu_seqlens=cu, chunk_offsets=offsets, T=t, N=n, H=h, HV=hv, K=dim, V=v, BT=chunk_size, BV=bv,
        USE_G=False, USE_GK=gate is not None, USE_INITIAL_STATE=initial is not None,
        STORE_FINAL_STATE=final is not None, SAVE_NEW_VALUE=True, STATE_V_FIRST=vf, IS_VARLEN=cu is not None,
        num_warps=warps, num_stages=stages,
    )
    return states, values, final
