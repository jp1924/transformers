# coding=utf-8
# Copyright 2025 The OpenBMB Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional, Tuple, Union

from transformers import PretrainedConfig
from transformers.utils import logging

from ..auto import CONFIG_MAPPING, AutoConfig


logger = logging.get_logger(__name__)


class MiniCPMOSpeechConfig(PretrainedConfig):
    model_type = "minicpm_o_speech_model"
    base_config_key = "speech_config"

    def __init__(
        self,
        attn_implementation: str = "sdpa",
        audio_bos_token_id: int = 21132,
        aug_loss_weight: bool = True,
        do_sample: bool = True,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        llm_dim: int = 2560,
        max_position_embeddings: int = 4096,
        num_attention_heads: int = 12,
        num_audio_tokens: int = 626,
        num_hidden_layers: int = 20,
        num_mel_bins: int = 100,
        num_spk_embs: int = 1,
        num_text_tokens: int = 21178,
        num_vq: int = 4,
        repetition_penalty: float = 1.0,
        spk_emb_token_id: int = 21143,
        streaming: bool = True,
        streaming_audio_chunk_size: int = 50,
        streaming_text_chunk_size: int = 10,
        streaming_text_reserved_len: int = 300,
        text_eos_token_id: int = 21133,
        top_k: int = 20,
        top_p: float = 0.7,
        use_llm_hidden_state: bool = False,
        use_mlp: bool = True,
        use_speaker_embedding: bool = True,
        use_text: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.llm_dim = llm_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.max_position_embeddings = max_position_embeddings
        self.num_audio_tokens = num_audio_tokens
        self.num_text_tokens = num_text_tokens
        self.num_mel_bins = num_mel_bins
        self.num_vq = num_vq
        self.use_speaker_embedding = use_speaker_embedding
        self.use_llm_hidden_state = use_llm_hidden_state
        self.spk_emb_token_id = spk_emb_token_id
        self.num_spk_embs = num_spk_embs
        self.audio_bos_token_id = audio_bos_token_id
        self.text_eos_token_id = text_eos_token_id
        self.use_text = use_text
        self.streaming = streaming
        self.streaming_text_chunk_size = streaming_text_chunk_size
        self.streaming_text_reserved_len = streaming_text_reserved_len
        self.streaming_audio_chunk_size = streaming_audio_chunk_size
        self.attn_implementation = attn_implementation
        self.use_mlp = use_mlp
        self.aug_loss_weight = aug_loss_weight
        self.do_sample = do_sample
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty


class MiniCPMOResamplerConfig(PretrainedConfig):
    def __init__(
        self,
        lpm_hidden_size: int = 3584,
        vpm_hidden_size: int = 1152,
        num_query_tokens: int = 64,
        pos_max_size: Tuple[int, int] = (70, 70),
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_query_tokens = num_query_tokens
        self.lpm_hidden_size = lpm_hidden_size
        self.vpm_hidden_size = vpm_hidden_size
        self.num_attention_heads = lpm_hidden_size // 128
        self.adaptive = adaptive
        self.pos_max_size = pos_max_size


class MiniCPMOVisionConfig(PretrainedConfig):
    model_type = "minicpm_o_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        attention_dropout: float = 0.0,
        hidden_act: str = "gelu_pytorch_tanh",
        hidden_size: int = 1152,
        image_size: int = 980,
        intermediate_size: int = 4304,
        layer_norm_eps: float = 1e-6,
        max_slice_nums: int = 9,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        num_hidden_layers: int = 27,
        patch_size: int = 14,
        scale_resolution: int = 448,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.patch_size = patch_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution


class MiniCPMOAudioConfig(PretrainedConfig):
    model_type = "minicpm_o_audio_model"
    base_config_key = "stt_config"

    def __init__(
        self,
        activation_dropout: float = 0.0,
        activation_function: str = "gelu",
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        hidden_size: int = 1024,  # d_model
        intermediate_size: int = 4096,  # encoder_ffn_dim
        layerdrop: float = 0.0,
        max_source_positions: int = 1500,
        num_attention_heads: int = 16,  # encoder_attention_heads
        num_hidden_layers: int = 24,  # encoder_layers
        num_mel_bins: int = 80,
        scale_embedding: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.num_mel_bins = num_mel_bins
        self.scale_embedding = scale_embedding
        self.layerdrop = layerdrop
        self.num_hidden_layers = num_hidden_layers
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.max_source_positions = max_source_positions


class MiniCPMOConfig(PretrainedConfig):
    model_type = "minicpmo"
    keys_to_ignore_at_inference = ["past_key_values"]

    sub_configs = {"text_config": AutoConfig, "vision_config": AutoConfig}

    def __init__(
        self,
        text_config: Union[str, dict, PretrainedConfig],
        audio_chunk_length=1.0,
        audio_config: Optional[Union[str, dict, PretrainedConfig]] = None,
        audio_pool_step=2,
        batch_vision_input=True,
        drop_vision_last_layer=True,
        image_size=448,
        max_slice_nums=9,
        patch_slice_size=14,
        query_num=64,
        resampler_config: Optional[Union[str, dict, PretrainedConfig]] = None,
        scale_resolution=448,
        speech_config: Optional[Union[str, dict, PretrainedConfig]] = None,
        stream_input=False,
        use_cache=True,
        use_image_id=True,
        vision_batch_size=16,
        vision_config: Optional[Union[str, dict, PretrainedConfig]] = None,
        **kwargs,
    ):
        # MiniCPM-V specific
        self.patch_slice_size = patch_slice_size
        self.max_slice_nums = max_slice_nums
        self.scale_resolution = scale_resolution
        # MiniCPM-V specific

        self.use_cache = use_cache
        self.query_num = query_num
        self.image_size = image_size
        self.drop_vision_last_layer = drop_vision_last_layer
        self.batch_vision_input = batch_vision_input
        self.use_image_id = use_image_id
        self.vision_batch_size = vision_batch_size
        self.audio_pool_step = audio_pool_step
        self.audio_chunk_length = audio_chunk_length
        self.stream_input = stream_input

        if isinstance(text_config, str):
            self.text_config = AutoConfig.from_pretrained(text_config)
        elif isinstance(text_config, PretrainedConfig):
            self.text_config = text_config
        elif isinstance(text_config, dict):
            self.text_config = CONFIG_MAPPING[text_config.pop("model_type")].from_dict(text_config)
        else:
            raise ValueError(
                "text_config should be a string, PretrainedConfig or dict, but got {}".format(type(text_config))
            )

        # same as HuggingFaceM4/siglip-so400m-14-980-flash-attn2-navit add tgt_sizes
        if isinstance(vision_config, str):
            self.vision_config = AutoConfig.from_pretrained(vision_config)
        elif isinstance(vision_config, PretrainedConfig):
            self.vision_config = vision_config
        elif isinstance(vision_config, dict):
            vision_model_type = vision_config.pop("model_type")
            if vision_model_type == "minicpm_o_vision_model":
                self.vision_config = MiniCPMOVisionConfig.from_dict(vision_config)
            else:
                self.vision_config = CONFIG_MAPPING[vision_model_type].from_dict(vision_config)
        else:
            self.vision_config = None

        if isinstance(audio_config, str):
            self.audio_config = AutoConfig.from_pretrained(audio_config)
        elif isinstance(audio_config, PretrainedConfig):
            self.audio_config = audio_config
        elif isinstance(audio_config, dict):
            audio_model_type = audio_config.pop("model_type")
            if audio_model_type == "minicpm_o_audio_model":
                self.audio_config = MiniCPMOAudioConfig.from_dict(audio_config)
            else:
                self.audio_config = CONFIG_MAPPING[audio_model_type].from_dict(audio_config)
        else:
            self.audio_config = None

        if isinstance(speech_config, str):
            self.speech_config = AutoConfig.from_pretrained(speech_config)
        elif isinstance(speech_config, PretrainedConfig):
            self.speech_config = speech_config
        elif isinstance(speech_config, dict):
            speech_model_type = speech_config.pop("model_type")
            if speech_model_type == "minicpm_o_speech_model":
                self.speech_config = MiniCPMOSpeechConfig.from_dict(speech_config)
            else:
                self.speech_config = CONFIG_MAPPING[speech_model_type].from_dict(speech_config)
        else:
            self.speech_config = None

        self.resampler_config = MiniCPMOResamplerConfig()

        super().__init__(**kwargs)
