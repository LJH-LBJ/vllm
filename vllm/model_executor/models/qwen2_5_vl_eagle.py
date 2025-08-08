# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable
from typing import Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig, Qwen2_5_VLVisionConfig)
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.layers.quantization.torchao import TorchAOConfig
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM
from vllm.model_executor.models.qwen2_5_vl import (Qwen2_5_VLForConditionalGeneration,
                                              Qwen2_5_VisionTransformer)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

from .utils import (AutoWeightsLoader, WeightsMapper, cast_overflow_tensors,
                    init_vllm_registered_model, maybe_prefix,
                    merge_multimodal_embeddings, PPMissingLayer)

from .interfaces import (MultiModalEmbeddings, SupportsLoRA,
                         SupportsMultiModal, SupportsPP)

logger = init_logger(__name__)


class Qwen2_5Model(nn.Module):
    
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        start_layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.config = (
            vllm_config.speculative_config.draft_model_config.hf_config)
        self.multimodal_config = (
            vllm_config.speculative_config.draft_model_config.multimodal_config)
        # embbeding
        if get_pp_group().is_first_rank or (self.config.tie_word_embeddings
                                            and get_pp_group().is_last_rank):
            self.embed_tokens = VocabParallelEmbedding(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()
        
         # 轻量语言模型初始化
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(
                self.config,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, f"layers.{i + start_layer_id}"),
            ) for i in range(self.config.num_hidden_layers)
        ])
        # EAGLE特征融合层
        self.fc = torch.nn.Linear(self.config.hidden_size * 2,
                                  self.config.hidden_size,
                                  bias=False)
        self.norm = RMSNorm(self.config.hidden_size,
                            eps=self.config.rms_norm_eps)
        
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings(input_ids)
        # eagle 的设计，通过融合当前输入和历史的隐藏状态来增强预测能力
        hidden_states = self.fc(
            torch.cat((inputs_embeds, hidden_states), dim=-1))
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, hidden_states

    def load_weights(self, weights: Iterable[tuple[str,
                                                   torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # name = name.removeprefix("model.")
            # TODO :和训练的模型有关，可能需要修改
            if (name.find("t2d") or name.find("d2t") or name.find("hidden_norm"))  and name not in params_dict:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # if PP disabled then draft will share embed with target
                if get_pp_group().world_size == 1 and \
                    "embed_tokens." in name:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                # TODO: 训练一个合适的模型
                if name.startswith("fc"):
                    loaded_weight = loaded_weight[:, :self.config.hidden_size * 2]
                    logger.warning(f"reduce size loaded_weight:{name}: {loaded_weight.shape}")
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

class EagleQwen2_5_VLForCausalLM(Qwen2ForCausalLM):
    
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.multimodal_config = vllm_config.model_config.multimodal_config

        # 目标模型的层数（用于 draft model 的 start_layer_id）
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config)
        # draft model quantization config may differ from target model
        quant_config = VllmConfig.get_quantization_config(
            vllm_config.speculative_config.draft_model_config,
            vllm_config.load_config)
        # 初始化 QWEN2.5 的 EAGLE 模型
        self.model = Qwen2_5Model(vllm_config=vllm_config,
                                prefix=maybe_prefix(prefix, "draft_model"),
                                start_layer_id=target_layer_num,
                                quant_config=quant_config)
        # QWEN2.5 的 logit scale 配置
        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(self.config.vocab_size,
                                                scale=logit_scale)
    
    def load_weights(self, weights):
        loader = AutoWeightsLoader(self,
                                   skip_prefixes=(["lm_head."]),
                                   )
        model_weights = {}
        
        for name, loaded_weight in weights:
            if "lm_head" not in name:
                name = "model." + name
            model_weights[name] = loaded_weight

        loader.load_weights(model_weights.items())

    def forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            inputs_embeds: Optional[torch.Tensor] = None,
            **kwargs: object,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self.model(input_ids, positions, hidden_states, inputs_embeds)
    
    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: Optional[MultiModalEmbeddings] = None,
    ) -> torch.Tensor:
        inputs_embeds = self.model.get_input_embeddings(input_ids)
        if multimodal_embeddings is not None \
            and len(multimodal_embeddings) != 0:
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                self.config.image_token_index)
        return inputs_embeds

        