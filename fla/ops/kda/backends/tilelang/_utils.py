# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch


def canonicalize_contiguous_strides(x: torch.Tensor) -> torch.Tensor:
    """Return a row-major tensor while avoiding copies for noncanonical singleton strides."""
    x = x.contiguous()
    strides = [1] * x.ndim
    for i in range(x.ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * max(x.shape[i + 1], 1)
    strides = tuple(strides)
    if x.stride() != strides:
        x = x.as_strided(x.shape, strides)
    return x
