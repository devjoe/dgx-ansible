#!/usr/bin/env python3
"""Experiment-only vLLM Qwen3 MoE activation steering hook.

This module is intentionally loaded by a wrapper launcher instead of installed
as a package plugin. Importing it registers a replacement Qwen3Moe model class
for the current vLLM process only.
"""

from __future__ import annotations

import logging
import os
from itertools import islice
from typing import Any

import torch

from vllm.distributed import get_pp_group
from vllm.model_executor.models import qwen3_5, qwen3_moe
from vllm.model_executor.models.registry import ModelRegistry
from vllm.sequence import IntermediateTensors

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_layers(value: str) -> set[int]:
    layers: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            layers.update(range(int(start_s), int(end_s) + 1))
        else:
            layers.add(int(part))
    return layers


class QwenSteeringState:
    def __init__(self, vllm_config: Any):
        self.enabled = _env_bool("QWEN_STEERING_ENABLED")
        self.apply_count = 0
        self._cache: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self.directions: dict[int, torch.Tensor] = {}
        self.layers: set[int] = set()
        self.model_name = str(getattr(vllm_config.model_config, "model", ""))
        target_model = os.environ.get("QWEN_STEERING_TARGET_MODEL", "").strip()
        if target_model and self.model_name != target_model:
            self.enabled = False
        if not self.enabled:
            return

        path = os.environ.get("QWEN_STEERING_DIRECTIONS", "").strip()
        if not path:
            raise RuntimeError("QWEN_STEERING_DIRECTIONS is required when steering is enabled")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        raw_directions = payload.get("directions")
        if not isinstance(raw_directions, dict):
            raise RuntimeError(f"No directions dict found in {path}")

        self.layers = _parse_layers(os.environ.get("QWEN_STEERING_LAYERS", ""))
        if not self.layers:
            self.layers = {int(layer) for layer in raw_directions}
        missing = sorted(layer for layer in self.layers if layer not in raw_directions)
        if missing:
            raise RuntimeError(f"Requested steering layers missing from directions: {missing}")

        self.directions = {
            int(layer): tensor.detach().float().cpu()
            for layer, tensor in raw_directions.items()
        }
        self.scale = float(os.environ.get("QWEN_STEERING_SCALE", "0.05"))
        self.sign = float(os.environ.get("QWEN_STEERING_SIGN", "1.0"))
        self.mode = os.environ.get("QWEN_STEERING_MODE", "both").strip().lower()
        self.method = os.environ.get("QWEN_STEERING_METHOD", "signed_projection").strip().lower()
        self.decode_token_threshold = int(os.environ.get("QWEN_STEERING_DECODE_TOKEN_THRESHOLD", "1"))
        logger.warning(
            "Qwen steering enabled: model=%s layers=%s scale=%s sign=%s mode=%s method=%s",
            self.model_name,
            sorted(self.layers),
            self.scale,
            self.sign,
            self.mode,
            self.method,
        )

    def _request_phase(self, hidden_states: torch.Tensor) -> str:
        # vLLM flattens token batches. For this experiment we use the token count
        # as a cheap prefill/decode proxy; batched decode can exceed the threshold.
        return "decode" if hidden_states.shape[0] <= self.decode_token_threshold else "prefill"

    def _direction(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        direction_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if direction_tensor is not None:
            direction = direction_tensor.to(dtype=hidden_states.dtype)
            return direction / direction.norm().clamp_min(1e-6)

        key = (layer_idx, hidden_states.device, hidden_states.dtype)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        direction = self.directions[layer_idx].to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        direction = direction / direction.norm().clamp_min(1e-6)
        self._cache[key] = direction
        return direction

    def apply(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        direction_tensor: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled or layer_idx not in self.layers or self.scale == 0:
            return hidden_states
        phase = self._request_phase(hidden_states)
        if self.mode not in {"both", phase}:
            return hidden_states

        direction = self._direction(layer_idx, hidden_states, direction_tensor)
        if self.method == "add":
            steered = hidden_states + (self.scale * self.sign) * direction
        elif self.method == "signed_projection":
            projection = hidden_states.float().matmul(direction.float())
            delta = (self.scale * self.sign) * projection.to(hidden_states.dtype).unsqueeze(-1) * direction
            steered = hidden_states - delta
        else:
            raise RuntimeError(f"Unsupported QWEN_STEERING_METHOD={self.method!r}")

        return steered


class SteeredQwen3MoeModel(qwen3_moe.Qwen3MoeModel):
    def __init__(self, *, vllm_config: Any, prefix: str = "", decoder_layer_type: type[torch.nn.Module] = qwen3_moe.Qwen3MoeDecoderLayer):
        super().__init__(
            vllm_config=vllm_config,
            prefix=prefix,
            decoder_layer_type=decoder_layer_type,
        )
        self.qwen_steering = QwenSteeringState(vllm_config)
        for layer_idx, direction in self.qwen_steering.directions.items():
            if torch.cuda.is_available():
                direction = direction.to(
                    device=torch.device("cuda", torch.cuda.current_device())
                )
            self.register_buffer(
                f"qwen_steering_direction_{layer_idx}",
                direction,
                persistent=False,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state(
            [], self.start_layer, hidden_states, residual
        )
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            steering_layer = layer_idx + 1
            direction = getattr(
                self,
                f"qwen_steering_direction_{steering_layer}",
                None,
            )
            hidden_states = self.qwen_steering.apply(
                steering_layer,
                hidden_states,
                direction,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


class SteeredQwen3MoeForCausalLM(qwen3_moe.Qwen3MoeForCausalLM):
    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        original_model_cls = qwen3_moe.Qwen3MoeModel
        qwen3_moe.Qwen3MoeModel = SteeredQwen3MoeModel
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
        finally:
            qwen3_moe.Qwen3MoeModel = original_model_cls


class SteeredQwen3_5Model(qwen3_5.Qwen3_5Model):
    def __init__(self, *, vllm_config: Any, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.qwen_steering = QwenSteeringState(vllm_config)
        for layer_idx, direction in self.qwen_steering.directions.items():
            if torch.cuda.is_available():
                direction = direction.to(
                    device=torch.device("cuda", torch.cuda.current_device())
                )
            self.register_buffer(
                f"qwen_steering_direction_{layer_idx}",
                direction,
                persistent=False,
            )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            steering_layer = layer_idx + 1
            direction = getattr(
                self,
                f"qwen_steering_direction_{steering_layer}",
                None,
            )
            hidden_states = self.qwen_steering.apply(
                steering_layer,
                hidden_states,
                direction,
            )
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class SteeredQwen3_5MoeForConditionalGeneration(
    qwen3_5.Qwen3_5MoeForConditionalGeneration
):
    def __init__(self, *, vllm_config: Any, prefix: str = "model"):
        original_model_cls = qwen3_5.Qwen3_5Model
        qwen3_5.Qwen3_5Model = SteeredQwen3_5Model
        try:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            if not hasattr(self.config, "image_token_index") and hasattr(
                self.config, "image_token_id"
            ):
                self.config.image_token_index = self.config.image_token_id
        finally:
            qwen3_5.Qwen3_5Model = original_model_cls


ModelRegistry.register_model("Qwen3MoeForCausalLM", SteeredQwen3MoeForCausalLM)
logger.warning("Registered experiment Qwen3MoeForCausalLM steering hook")
ModelRegistry.register_model(
    "Qwen3_5MoeForConditionalGeneration",
    SteeredQwen3_5MoeForConditionalGeneration,
)
logger.warning("Registered experiment Qwen3_5MoeForConditionalGeneration steering hook")
