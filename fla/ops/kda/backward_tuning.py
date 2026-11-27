# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Opt-in KDA backward launch configurations."""

import os


def intra_dot_precision(tf32_supported: bool) -> str | None:
    """Select scoped precision, or delegate to Triton for a legacy-behavior ablation."""
    mode = os.getenv('FLA_KDA_BWD_INTRA_DOT_PRECISION', 'ieee') or 'ieee'
    if mode == 'default':
        # None matches tl.dot's omitted input_precision argument, including Triton's environment policy.
        return None
    if mode not in ('ieee', 'tf32x3'):
        raise ValueError('FLA_KDA_BWD_INTRA_DOT_PRECISION must be ieee, tf32x3 or default')
    if mode == 'tf32x3' and not tf32_supported:
        raise ValueError('KDA intra-backward tf32x3 requires an NVIDIA GPU with TF32 support')
    return mode


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


def state_backward_config() -> tuple[int, bool] | None:
    """Keep state-backward tuning separate from WY and intra dot precision."""
    raw = os.getenv('FLA_KDA_BWD_DHU_BV', '')
    order = os.getenv('FLA_KDA_BWD_DHU_HEAD_FIRST', '0')
    if order not in ('0', '1'):
        raise ValueError('FLA_KDA_BWD_DHU_HEAD_FIRST must be 0 or 1')
    if not raw:
        if order == '1':
            raise ValueError('Set FLA_KDA_BWD_DHU_BV when enabling head-first state backward')
        return None
    try:
        bv = int(raw)
    except ValueError as exc:
        raise ValueError('FLA_KDA_BWD_DHU_BV must be 16, 32 or 64') from exc
    if bv not in (16, 32, 64):
        raise ValueError('FLA_KDA_BWD_DHU_BV must be 16, 32 or 64')
    return bv, order == '1'


def state_backward(**kwargs):
    """Scope launch tuning to KDA while preserving backend and intra-card dispatch."""
    from fla.ops.common.chunk_delta_h import _KDA_DHU_LAUNCH_CONFIG, chunk_gated_delta_rule_bwd_dhu
    from fla.utils import IS_NVIDIA_BLACKWELL

    config = state_backward_config()
    if config is None:
        return chunk_gated_delta_rule_bwd_dhu(**kwargs)
    if not IS_NVIDIA_BLACKWELL:
        raise ValueError('KDA state-backward launch tuning currently requires NVIDIA Blackwell')
    token = _KDA_DHU_LAUNCH_CONFIG.set(config)
    try:
        return chunk_gated_delta_rule_bwd_dhu(**kwargs)
    finally:
        _KDA_DHU_LAUNCH_CONFIG.reset(token)
