# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

"""Intra-card CP backend for shared delta rule operations.

Accelerates prefill by splitting long sequences into sub-sequences
and processing them in parallel across SMs.

Only active for variable-length inputs without static graph offsets.
"""

from __future__ import annotations

import os
import warnings

import torch

from fla.ops.backends import BaseBackend
from fla.utils import IS_TF32_SUPPORTED

# Maximum number of sub-sequences per original sequence
# Limits merge chain depth to control precision loss
MAX_SUBSEQS = int(os.environ.get('FLA_INTRACARD_MAX_SPLITS', 32))

# use tf32x3 for the affine-chain dots in the pre-scan/merge kernels (NVIDIA only)
USE_TF32X3_AFFINE_CHAIN = os.environ.get('FLA_INTRACARD_TF32X3', '0') == '1'

MAX_INTRACARD_HEAD_DIM = 256

if USE_TF32X3_AFFINE_CHAIN and not IS_TF32_SUPPORTED:
    warnings.warn(
        "tf32x3 affine chain requires an NVIDIA GPU with compute capability >= 8.0; falling back to ieee precision",
        stacklevel=2,
    )


class IntraCardCPBackend(BaseBackend):
    """Intra-card context parallel backend for shared delta-rule state scans."""

    backend_type = "intracard_cp"
    package_name = None  # No external package needed
    env_var = "FLA_INTRACARD_CP"
    default_enable = False

    @classmethod
    def is_available(cls) -> bool:
        return True

    def prepare_chunk_gated_delta_rule_fwd_h_affine(
        self,
        k: torch.Tensor,
        w: torch.Tensor,
        u: torch.Tensor,
        g: torch.Tensor | None = None,
        gk: torch.Tensor | None = None,
        cu_seqlens: torch.LongTensor | None = None,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        chunk_size: int = 64,
    ) -> object | None:
        from fla.ops.common.intracard_cp import prepare_intracard_fwd_affine_summary

        return prepare_intracard_fwd_affine_summary(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=chunk_size,
            max_splits=MAX_SUBSEQS,
            use_tf32x3_affine_chain=USE_TF32X3_AFFINE_CHAIN,
        )

    def prepare_chunk_gated_delta_rule_bwd_dhu_affine(
        self,
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
    ) -> object | None:
        from fla.ops.common.intracard_cp import prepare_intracard_bwd_affine_summary

        return prepare_intracard_bwd_affine_summary(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            scale=scale,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=chunk_size,
            max_splits=MAX_SUBSEQS,
            use_tf32x3_affine_chain=USE_TF32X3_AFFINE_CHAIN,
        )

    def chunk_gated_delta_rule_fwd_h_verifier(
        self,
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
        chunk_offsets: torch.LongTensor | None = None,
        return_intra_initial_state: bool = False,
        intra_initial_state: torch.Tensor | None = None,
        intra_affine_summary: object | None = None,
    ) -> tuple[bool, str | None]:
        """Check if intracard CP should handle this call."""
        if cu_seqlens is None:
            return False, "cu_seqlens is None"
        if chunk_offsets is not None:
            return False, "static chunk_offsets are not supported"
        if k.shape[-1] > MAX_INTRACARD_HEAD_DIM:
            return False, f"key head dimension exceeds intra-card merge limit of {MAX_INTRACARD_HEAD_DIM}"

        return True, None

    def chunk_gated_delta_rule_fwd_h(
        self,
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
        chunk_offsets: torch.LongTensor | None = None,
        return_intra_initial_state: bool = False,
        intra_initial_state: torch.Tensor | None = None,
        intra_affine_summary: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Intra-card CP implementation of chunk_gated_delta_rule_fwd_h."""
        from fla.ops.common.intracard_cp import intracard_fwd_h

        return intracard_fwd_h(
            k=k, w=w, u=u, g=g, gk=gk,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            save_new_value=save_new_value,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_indices=chunk_indices,
            max_splits=MAX_SUBSEQS,
            state_v_first=state_v_first,
            use_tf32x3_affine_chain=USE_TF32X3_AFFINE_CHAIN,
            return_intra_initial_state=return_intra_initial_state,
            intra_initial_state=intra_initial_state,
            intra_affine_summary=intra_affine_summary,
        )

    def chunk_gated_delta_rule_bwd_dhu_verifier(
        self,
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
        use_graph: bool = False,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        intra_affine_summary: object | None = None,
    ) -> tuple[bool, str | None]:
        if cu_seqlens is None:
            return False, "cu_seqlens is None"
        if chunk_offsets is not None:
            return False, "static chunk_offsets are not supported"
        if use_graph:
            return False, "use_graph=True is not supported"
        if scale is None:
            return False, "scale is None"
        if g is not None and gk is not None:
            return False, "simultaneous scalar and per-key gates are not supported"
        if q.shape[-1] > MAX_INTRACARD_HEAD_DIM:
            return False, f"key head dimension exceeds intra-card merge limit of {MAX_INTRACARD_HEAD_DIM}"
        return True, None

    def chunk_gated_delta_rule_bwd_dhu(
        self,
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
        use_graph: bool = False,
        cu_seqlens_cpu: torch.LongTensor | None = None,
        intra_affine_summary: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        from fla.ops.common.intracard_cp import intracard_bwd_dhu

        return intracard_bwd_dhu(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            h0=h0,
            dht=dht,
            scale=scale,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            chunk_size=chunk_size,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            max_splits=MAX_SUBSEQS,
            use_tf32x3_affine_chain=USE_TF32X3_AFFINE_CHAIN,
            intra_affine_summary=intra_affine_summary,
        )
