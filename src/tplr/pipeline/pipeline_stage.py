# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import operator
from abc import ABC, abstractmethod
from typing import Any, Callable, cast, Optional, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.stage import _RootArgPlaceholder, _RecvInfo
from torch.distributed._tensor import DTensor, distribute_tensor
from torch.distributed._tensor.placement_types import Replicate

__all__ = [
    "PipelineStageProtocolCompression",
]

logger = logging.getLogger(__name__)


_HEADER_FIELDS = 6  # [compressed_flag, B, S, D, rank, comp_numel]


def _as_tuple(x: Any) -> tuple[Any, ...]:
    return x if isinstance(x, tuple) else (x,)


@torch.no_grad()
def project_embeddings_rowwise_inplace(embed: nn.Embedding, U_k: torch.Tensor):
    # rowwise: W <- W @ U @ Uᵀ (keeps W in subspace spanned by U_k)
    W = embed.weight

    if isinstance(W, DTensor):
        W_local = W.to_local()
        updated = (W_local @ U_k @ U_k.transpose(0, 1)).contiguous()
        W_local.copy_(updated.contiguous())
    else:
        W = W.data
        U_k = U_k.to(dtype=W.dtype, device=W.device, non_blocking=True)
        updated = (W @ U_k) @ U_k.transpose(0, 1)
        W.copy_(updated.contiguous())


class PipelineStageProtocolCompression(PipelineStage):
    """
    Pipeline stage that:
      * Compresses inter-stage activations:  X_res = X - (PE + T_fixed[token_indices]);  send  X_res @ U_k
      * Decompresses on recv:               X = (Xc @ U_kᵀ) + (PE + T_fixed[token_indices])
      * Ships `token_indices` from stage0 to all later stages as a separate tensor every step.
      * Compresses/decompresses backward gradients (no tokens involved).
      * No Grassmann updates; no per-layer within-stage compression; no paper order; rowwise only.
    """

    def __init__(
        self,
        submodule: nn.Module,
        stage_index: int,
        num_stages: int,
        device: torch.device,
        group: Optional[dist.ProcessGroup] = None,
        input_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        output_args: Optional[Union[torch.Tensor, tuple[torch.Tensor, ...]]] = None,
        dw_builder: Optional[Callable[[], Callable[..., None]]] = None,
    ):
        super().__init__(
            submodule=submodule,
            stage_index=stage_index,
            num_stages=num_stages,
            device=device,
            input_args=input_args,
            output_args=output_args,
            group=group,
            dw_builder=dw_builder,
        )
        # Shared compression state across stages
        self.U_k: torch.Tensor = None
        self.T_fixed: torch.Tensor = None

        # Token cache per microbatch id
        self._mb_token_indices: dict[int, torch.Tensor] = {}

    def set_compression_attributes(
        self,
        U_k: torch.Tensor | None,
        T_fixed: torch.Tensor,
        *,
        project_embedding: bool = True,
    ) -> None:
        """Late-bind compression tensors (must be called after weights exist)."""
        # shape sanity
        if self.U_k is None and U_k is None:
            raise ValueError("Given U_k can only be None if self.U_k is already set.")
        if U_k is not None and U_k.dim() != 2:
            raise ValueError(f"U_k must be 2D [D,k], got {tuple(U_k.shape)}")
        if T_fixed.dim() != 2:
            raise ValueError(f"T_fixed must be 2D [V,D], got {tuple(T_fixed.shape)}")

        # Non-DTensor path: place on this stage's device
        if U_k is not None:
            self.U_k = U_k.to(self.device)
            
        self.T_fixed = T_fixed.to(self.device)

        self._compression_ready = True

        # One-time projection on first stage
        if project_embedding and self.is_first:
            m = getattr(self.submod, "module", self.submod)
            embed = getattr(m, "tok_embeddings", None)
            if embed is not None and isinstance(embed, nn.Embedding):
                project_embeddings_rowwise_inplace(embed, self.U_k)

    # ---- Protocol compression (rowwise) ----
    def _compress(self, X: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        # X: [B, S, D], token_indices: [B, S]
        B, S, D = X.shape
        X_res = X - self.T_fixed[token_indices]  # [B, S, D] (PE omitted)
        Xc = torch.matmul(X_res, self.U_k)  # [B, S, k]
        return Xc.reshape(-1)  # no header

    def _decompress(
        self, packed: torch.Tensor, token_indices: torch.Tensor
    ) -> torch.Tensor:
        B, S = token_indices.shape
        D = self.U_k.shape[0]
        rank = self.U_k.shape[1]
        Xc = packed.view(B, S, rank)  # [B, S, k]
        X_res = torch.matmul(Xc, self.U_k.transpose(0, 1))  # [B, S, D]
        return X_res + self.T_fixed[token_indices.long()]  # [B, S, D]

    def _compress_grad(self, G: torch.Tensor) -> torch.Tensor:
        B, S, D = G.shape
        Gc = torch.matmul(G, self.U_k)  # [B, S, k]
        return Gc.reshape(-1)

    def _decompress_grad(self, packed: torch.Tensor) -> torch.Tensor:
        out_meta = self.get_outputs_meta()[0]  # meta [B, S, D] for the stage output
        B, S, D = out_meta.shape
        rank = self.U_k.shape[1]
        Gc = packed.view(B, S, rank)
        return torch.matmul(Gc, self.U_k.transpose(0, 1))

    # ---- Runtime shape/plumbing overrides ----

    def _calculate_packed_size(self, original_shape: tuple[int, ...]) -> int:
        """
        Size of packed activation/gradient: header(6) + compressed.
        """
        if len(original_shape) != 3:
            return int(torch.Size(original_shape).numel())
        B, S, D = original_shape
        rank = self.U_k.shape[1]
        return B * S * rank

    def _shape_inference(
        self,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        """
        Same as parent, but for non-first stages we synthesize a 2nd input meta for token_indices (B,S).
        """
        outputs_meta = super()._shape_inference(args, kwargs)
        # After parent fills self.inputs_meta:
        assert self.inputs_meta is not None
        if not self.is_first:
            # inputs_meta[0] is the activation meta with shape [B,S,D]
            act_meta = self.inputs_meta[0]
            if isinstance(act_meta, torch.Tensor) and act_meta.dim() == 3:
                B, S = act_meta.shape[:2]
                tok_meta = torch.zeros(B, S, dtype=torch.long, device="meta")
                self.inputs_meta = (act_meta, tok_meta)
        return outputs_meta

    def _prepare_forward_infra(
        self,
        num_microbatches: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, ...]:
        """
        Allocate:
          - on non-first stages: a flat buffer for packed activation + a [B,S] long buffer for token_indices
          - on first stage: use root args (token_indices) as usual
        """
        outputs = tuple()
        if self.inputs_meta is None:
            outputs = self._shape_inference(args, kwargs)
        assert self.inputs_meta is not None

        for chunk_id in range(num_microbatches):
            if self.is_first:
                # Stage0 takes root args; no recv buffers
                self.args_recv_info[chunk_id] = tuple(
                    [_RootArgPlaceholder(i) for i in self.inputs_meta]
                )
            else:
                # activation meta [B,S,D] comes from parent shape inference
                act_meta = self.inputs_meta[0]
                assert isinstance(act_meta, torch.Tensor) and act_meta.dim() == 3
                packed_size = self._calculate_packed_size(tuple(act_meta.shape))

                act_buf = torch.empty(
                    packed_size, dtype=act_meta.dtype, device=self.device
                )
                if self.has_backward:
                    act_buf.requires_grad_(True)

                B, S = act_meta.shape[:2]
                tok_buf = torch.empty(B, S, dtype=torch.long, device=self.device)

                self.args_recv_info[chunk_id] = (
                    _RecvInfo(
                        f"recv_packed_act_{self.stage_index}_from_{self.stage_index - 1}",
                        self.stage_index - 1,
                        act_buf,
                    ),
                    _RecvInfo(
                        f"recv_tokens_{self.stage_index}_from_{self.stage_index - 1}",
                        self.stage_index - 1,
                        tok_buf,
                    ),
                )

        # Act send info: same as parent (sequential)
        self.act_send_info = {
            idx: ([] if self.is_last else [self.stage_index + 1])
            for idx in range(len(self.get_outputs_meta()))
        }
        return outputs

    # Receive + decompress activations; also cache token_indices for this microbatch
    def _retrieve_recv_activations(self, fwd_chunk_id: int):
        if self.is_first:
            # Stage0 receives no activations; parent will use call args
            return super()._retrieve_recv_activations(fwd_chunk_id)

        recv_infos = self.args_recv_info[fwd_chunk_id]

        # ✅ Wait for irecvs and fetch the *filled* tensors
        mapped = self._map_tensor_from_recv_info(recv_infos)
        if not isinstance(mapped, tuple):
            mapped = (mapped,)  # when there's a single tensor

        # Expect: (packed_activation_flat, token_indices)
        packed, token_indices = mapped

        X = self._decompress(packed, token_indices)
        X = X.detach().requires_grad_(True)

        # Cache tokens for forwarding to the next stage
        self._mb_token_indices[fwd_chunk_id] = token_indices.detach()
        return (X,)

    # Hook into forward path only to cache token_indices on stage0
    def forward_one_chunk(
        self,
        fwd_chunk_id: int,
        args: tuple[Any, ...],
        kwargs: Optional[dict[str, Any]] = None,
    ):
        if self.is_first:
            # Try to find token_indices in args/kwargs (assume [B,S] long)
            tok = None
            for a in _as_tuple(args):
                if (
                    isinstance(a, torch.Tensor)
                    and a.dtype == torch.long
                    and a.dim() == 2
                ):
                    tok = a
                    break
            if tok is None and kwargs:
                for v in kwargs.values():
                    if (
                        isinstance(v, torch.Tensor)
                        and v.dtype == torch.long
                        and v.dim() == 2
                    ):
                        tok = v
                        break
            if tok is None:
                raise RuntimeError(
                    "Stage0 could not locate token_indices ([B,S] long) in inputs"
                )
            self._mb_token_indices[fwd_chunk_id] = tok.detach()

        return super().forward_one_chunk(fwd_chunk_id, args, kwargs)

    # Send compressed activations AND token indices to next stage
    def get_fwd_send_ops(self, fwd_chunk_id: int) -> list[dist.P2POp]:
        output_tuple, _ = self.fwd_cache[fwd_chunk_id]
        ops: list[dist.P2POp] = []

        # Retrieve token indices cached for this microbatch (set in forward path)
        if not self.is_last:
            if fwd_chunk_id not in self._mb_token_indices:
                raise RuntimeError(
                    f"{self.log_prefix} missing token_indices for mb={fwd_chunk_id}"
                )
            tokens = self._mb_token_indices[fwd_chunk_id]

        for idx, out in enumerate(output_tuple):
            dst_stages = self.act_send_info[idx]
            if not dst_stages:
                continue

            # Compress activation with decoupling
            packed = self._compress(out, self._mb_token_indices[fwd_chunk_id])

            for dst in dst_stages:
                peer_rank = self.stage_index_to_group_rank[dst]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                # 1) send packed activation
                ops.append(dist.P2POp(dist.isend, packed, peer_global_rank, self.group))
                # 2) send token indices as a separate tensor
                ops.append(dist.P2POp(dist.isend, tokens, peer_global_rank, self.group))

        return ops

    # Backward: send compressed input grads back; no tokens involved
    def get_bwd_send_ops(self, bwd_chunk_id: int) -> list[dist.P2POp]:
        self._check_chunk_id(bwd_chunk_id)
        if not self.has_backward or self.is_first:
            return []

        if self.grad_send_info is None:
            self.grad_send_info = self._create_grad_send_info(self.args_recv_info[0])

        ops: list[dist.P2POp] = []
        grads_input = self.bwd_cache.pop(bwd_chunk_id)

        for grad, grad_recv_stage in zip(grads_input, self.grad_send_info):
            if isinstance(grad, torch.Tensor) and grad_recv_stage is not None:
                packed_grad = self._compress_grad(grad)
                peer_rank = self.stage_index_to_group_rank[grad_recv_stage]
                peer_global_rank = (
                    peer_rank
                    if self.group is None
                    else dist.get_global_rank(self.group, peer_rank)
                )
                ops.append(
                    dist.P2POp(dist.isend, packed_grad, peer_global_rank, self.group)
                )
            else:
                if not (grad is None and grad_recv_stage is None):
                    raise RuntimeError(
                        f"[{self.stage_index}] for chunk {bwd_chunk_id} has gradients {grad} "
                        f"and expects to send to stage {grad_recv_stage}"
                    )
        return ops

    # Allocate packed buffers for grad recv; grads carry no tokens
    def _create_grad_recv_info(self, act_send_info: dict) -> tuple["_RecvInfo", ...]:
        grad_recv_info: tuple["_RecvInfo", ...] = ()
        if not self.is_last:
            infos = []
            for idx, dst_list in act_send_info.items():
                if not dst_list:
                    continue
                out_meta = self.get_outputs_meta()[idx]  # meta [B,S,D]
                packed_size = self._calculate_packed_size(tuple(out_meta.shape))
                buf = torch.empty(packed_size, dtype=out_meta.dtype, device=self.device)
                infos.append(
                    _RecvInfo(
                        f"recv_grad_packed_for_{self.stage_index}_from_{dst_list[0]}",
                        dst_list[0],
                        buf,
                    )
                )
            grad_recv_info = tuple(infos)
        return grad_recv_info

    # Decompress grads on recv
    def _retrieve_recv_grads(self, bwd_chunk_id: int):
        if self.is_last:
            return super()._retrieve_recv_grads(bwd_chunk_id)
        recv_infos = self.grad_recv_info[bwd_chunk_id]
        packed_grads = self._map_tensor_from_recv_info(recv_infos)
        if not isinstance(packed_grads, tuple):
            packed_grads = (packed_grads,)
        grads = tuple(self._decompress_grad(p) for p in packed_grads)
        return grads

    def _validate_fwd_input(self, args, kwargs):
        # Stage 0 uses root args; let the base checks run.
        if self.is_first:
            return super()._validate_fwd_input(args, kwargs)
        # Non-first stages receive compressed buffers that don't match the model's input meta;
        # we'll decompress them ourselves, so skip the parent's strict shape checks.
        return
