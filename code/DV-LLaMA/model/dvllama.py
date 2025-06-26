from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from .dvllama_arch import DVLLaMAMetaModel, DVLLaMAMetaForCausalLM


class DVLLaMAConfig(LlamaConfig):
    model_type = "dvllama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_type = "dvllama"
        # 添加DVLLaMA特有的配置参数
        self.vision_hidden_size = kwargs.pop("vision_hidden_size", 768)
        self.vision_num_layers = kwargs.pop("vision_num_layers", 12)
        self.dynamic_adapter_type = kwargs.pop("dynamic_adapter_type", "parallel")
        self.adapter_dim = kwargs.pop("adapter_dim", 8)


class DVLLaMAModel(DVLLaMAMetaModel, LlamaModel):
    config_class = DVLLaMAConfig

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # DVLLaMA特有的初始化逻辑
        self._init_dynamic_vision_encoder(config)


class DVLLaMAForCausalLM(LlamaForCausalLM, DVLLaMAMetaForCausalLM):
    config_class = DVLLaMAConfig

    def __init__(self, config, **kwargs):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = DVLLaMAModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        videos: Optional[torch.FloatTensor] = None,  # 新增视频输入支持
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            # 为多模态输入准备输入和标签
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                videos  # 更新为支持视频输入
            )

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        outputs.labels = labels

        return outputs

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        videos: Optional[torch.Tensor] = None,  # 新增视频输入支持
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None or videos is not None:
            # 为多模态生成准备输入
            (
                input_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=inputs,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
                videos=videos  # 更新为支持视频输入
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        videos = kwargs.pop("videos", None)  # 新增视频输入支持
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if videos is not None:  # 更新为支持视频输入
            _inputs['videos'] = videos
        return _inputs


# 注册DVLLaMA模型
AutoConfig.register("dvllama", DVLLaMAConfig)
AutoModelForCausalLM.register(DVLLaMAConfig, DVLLaMAForCausalLM)