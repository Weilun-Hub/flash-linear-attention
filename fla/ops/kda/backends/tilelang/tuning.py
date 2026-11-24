# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Explicit, opt-in WY backward launch configurations."""

import os


def backward_launch_config(k: int, v: int, default_bk: int, default_bv: int, default_warps: int):
    """Keep defaults unless an explicit, shape-compatible candidate is requested."""
    raw = os.environ.get('FLA_KDA_BWD_WY_TILE', '')
    head_first = os.environ.get('FLA_KDA_BWD_WY_HEAD_FIRST', '0')
    if head_first not in ('0', '1'):
        raise ValueError('FLA_KDA_BWD_WY_HEAD_FIRST must be 0 or 1')
    if not raw:
        return default_bk, default_bv, default_warps, head_first == '1'
    try:
        bk, bv, warps = map(int, raw.split(','))
    except ValueError as exc:
        raise ValueError('FLA_KDA_BWD_WY_TILE must be BK,BV,WARPS, e.g. 64,64,8') from exc
    if bk not in (32, 64, 128) or bv not in (32, 64, 128) or warps not in (4, 8):
        raise ValueError('WY tiles must be 32/64/128 and warps 4/8')
    # this kernel's head-dimension accesses are not masked
    if k % bk or v % bv:
        raise ValueError(f'WY tile {(bk, bv)} must divide head dimensions {(k, v)}')
    return bk, bv, warps, head_first == '1'


def backward_shape_supported(k: int, v: int) -> bool:
    """Reject unmasked head tails before dispatch, including explicit tile overrides."""
    if k <= 0 or v <= 0:
        return False
    if os.environ.get('FLA_KDA_BWD_WY_TILE', ''):
        bk, bv, _, _ = backward_launch_config(128, 128, 64, 64, 4)
    else:
        bk = min(max(1 << (k - 1).bit_length(), 16), 64)
        bv = min(max(1 << (v - 1).bit_length(), 16), 64)
    return k % bk == 0 and v % bv == 0
