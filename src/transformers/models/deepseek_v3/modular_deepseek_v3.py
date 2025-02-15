import math
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import logging
from ..llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
from .configuration_deepseek_v3 import DeepseekV3Config


logger = logging.get_logger(__name__)


class DeepseekV3RMSNorm(LlamaRMSNorm):
    pass


class DeepseekV3RotaryEmbedding(LlamaRotaryEmbedding):
    def __init__(self, config: DeepseekV3Config, device=None):
        super().__init__(config, device)
        # RoPE dimensionality is `config.qk_rope_head_dim`
        inv_freq, self.attention_scaling = self.rope_init_fn(
            device=device,
            base=config.rope_theta,
            dim=config.qk_rope_head_dim,
        )

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                device=device, seq_len=seq_len, **self._rope_kwargs,
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekV3MLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.e_score_correction_bias = nn.Parameter(torch.empty((self.n_routed_experts)))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights, router_logits

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices


class DeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList(
            [
                DeepseekV3MLP(config, intermediate_size=config.moe_intermediate_size)
                for _ in range(config.n_routed_experts)
            ]
        )
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        topk_indices, topk_weights, router_logits = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.moe(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states, router_logits

    def moe(self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor):
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
        return final_hidden_states.type(hidden_states.dtype)


def _remove_inference_params(module: nn.Module):
    assert isinstance(module, DeepseekV3Attention)  # Sanity check
    module.inference_v_proj = None
    module.inference_q_decode = None


def _full_backward_hook(module: nn.Module, grad_input, grad_output):
    _remove_inference_params(module)


def _load_state_dict_post_hook(module: nn.Module, incompatible_keys):
    _remove_inference_params(module)


class AttentionParametersFingerprint:
    def __init__(
        self,
        q_b_proj: torch.Tensor,
        kv_b_proj: torch.Tensor,
        fingerprint_size: int = 256,
    ):
        self.q_b_shape = q_b_proj.shape
        self.kv_b_shape = kv_b_proj.shape
        q_b_proj = q_b_proj.flatten()
        kv_b_proj = kv_b_proj.flatten()
        q_b_sz = q_b_proj.shape[0]
        kv_b_sz = kv_b_proj.shape[0]
        assert fingerprint_size >= 2
        fp_q_sz = min(fingerprint_size // 2, q_b_sz)
        fp_kv_sz = min(fingerprint_size - fp_q_sz, kv_b_sz)
        self.device = q_b_proj.device
        self.fp_q_pos = torch.randint(
            low=0,
            high=q_b_sz,
            size=(fp_q_sz,),
            dtype=torch.int64,
            device=self.device,
        )
        self.fp_kv_pos = torch.randint(
            low=0,
            high=kv_b_sz,
            size=(fp_kv_sz,),
            dtype=torch.int64,
            device=self.device,
        )
        self.fingerprint = self._extract_fingerprint(q_b_proj, kv_b_proj).clone()

    def match(self, q_b_proj: torch.Tensor, kv_b_proj: torch.Tensor) -> bool:
        if q_b_proj.shape != self.q_b_shape or kv_b_proj.shape != self.kv_b_shape:
            return False
        if q_b_proj.device != self.device or kv_b_proj.device != self.device:
            return False
        args_fingerprint = self._extract_fingerprint(
            q_b_proj.flatten(), kv_b_proj.flatten()
        )
        return self.fingerprint.eq(args_fingerprint).all().item()

    def _extract_fingerprint(
        self, q_b_proj: torch.Tensor, kv_b_proj: torch.Tensor
    ) -> torch.Tensor:
        assert q_b_proj.ndim == kv_b_proj.ndim == 1
        assert q_b_proj.device == kv_b_proj.device == self.device
        return torch.cat((q_b_proj[self.fp_q_pos], kv_b_proj[self.fp_kv_pos]))


class DeepseekV3Attention(nn.Module):
    """
    Multi-head latent attention.

    We implement two different versions, referred to as "training" and
    "inference", which are used depending on `self.training`.

    The training variant is closely related to open source code released by the
    Deepseek authors. It does not combine Q and K decoding linear maps and does
    not exploit the low rank structure for efficient key-value caching. However,
    it is faster for training, and its linear maps are compatible with the
    weights released by Deepseek.

    The inference variant combines Q and K decoding linear maps. This allows to
    shrink the key-value cache to a single buffer without a heads dimension. It
    also needs less memory and is faster during inference. Without this variant,
    the advantage of multi-head latent attention over default multi-head
    self-attention is lost.

    Note that the inference variant needs parameter tensors derived from the
    linear blocks of the training variant. Some are the same or views, but
    in particular `inference_q_decode` needs to be computed. This needs some
    extra memory. The computations are done when inference needs them and the
    underlying parameters have changed. Changes are tracked by using a backward
    and `load_state_dict` hook, and also by way of a fingerprint. This is not
    perfect. If parameters are changed not through a backward pass or
    `load_state_dict` and the change does not affect the fingerprint, it may
    go unnoticed.

    To be safe, call :meth:`reset_inference_params` before running inference,
    this forces the inference parameters to be recomputed.

    If `config.attention_no_inference_mode == True`, the training variant will
    always be used. This is recommended only if inference is used sporadically
    (e.g., to compute validation scores during training).

    """
    def __init__(
        self,
        config: DeepseekV3Config,
        layer_idx: int,
        weights_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.is_causal = True
        # Maps input X to low-rank C_KV, K_R
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
            dtype=weights_dtype,
        )
        # Maps input X to low-rank C_Q
        self.q_a_proj = nn.Linear(
            config.hidden_size,
            self.q_lora_rank,
            bias=config.attention_bias,
            dtype=weights_dtype,
        )
        # Maps low-rank C_Q to Q (with RoPE part)
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
            dtype=weights_dtype,
        )
        # Maps low-rank C_KV to K (without RoPE part) and V
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            dtype=weights_dtype,
        )
        # Output projection
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
            dtype=weights_dtype,
        )

        self.q_a_layernorm = DeepseekV3RMSNorm(self.q_lora_rank)
        self.kv_a_layernorm = DeepseekV3RMSNorm(self.kv_lora_rank)

        # Parameters needed in inference variant, derived from those of the
        # training variant
        # inference_q_decode: (q_lora_rank, num_heads * (kv_lora_rank + qk_rope_head_dim))
        # inference_v_proj: (num_heads, kv_lora_rank, v_head_dim)
        self.inference_q_decode: Optional[torch.Tensor] = None
        self.inference_v_proj: Optional[torch.Tensor] = None
        # These hooks are called whenever a backward pass or `load_state_dict`
        # are called, which potentially changes model parameters. They reset
        # `inference_q_decode`, `inference_v_proj`
        self.register_full_backward_hook(_full_backward_hook)
        self.register_load_state_dict_post_hook(_load_state_dict_post_hook)
        # Fingerprint, used to test whether model parameters behind the
        # `inference_*` have changed. May miss certain changes
        self._inference_params_fingerprint: Optional[AttentionParametersFingerprint] = None

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling is not None:
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.training or self.config.attention_no_inference_mode:
            return self._forward_training(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs
            )
        else:
            return self._forward_inference(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                cache_position=cache_position,
                **kwargs
            )

    def _forward_training(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape

        c_q = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_pass, q_rot = self.q_b_proj(c_q).view(
            batch_size, seq_length, self.num_heads, self.qk_head_dim
        ).transpose(1, 2).split(
            (self.qk_nope_head_dim, self.qk_rope_head_dim),
            dim=-1
        )
        # q_pass: (batch_size, num_heads, seq_length, qk_nope_head_dim)
        # q_rot: (batch_size, num_heads, seq_length, qk_rope_head_dim)

        c_kv, k_rot = self.kv_a_proj_with_mqa(hidden_states).split(
            (self.kv_lora_rank, self.qk_rope_head_dim),
            dim=-1
        )
        c_kv = self.kv_a_layernorm(c_kv)
        k_pass, value_states = self.kv_b_proj(c_kv).view(
            batch_size, seq_length, self.num_heads, -1
        ).transpose(1, 2).split(
            (self.qk_nope_head_dim, self.v_head_dim),
            dim=-1
        )
        k_rot = k_rot.unsqueeze(1)

        # k_pass: (batch_size, num_heads, seq_length, qk_nope_head_dim)
        # k_rot: (batch_size, 1, seq_length, qk_rope_head_dim)
        # value_states: (batch_size, num_heads, seq_length, v_head_dim)

        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(*k_pass.shape[:-1], -1)
        # k_rot: (batch_size, num_heads, seq_length, qk_rope_head_dim)

        query_states = torch.cat((q_pass, q_rot), dim=-1)
        key_states = torch.cat((k_pass, k_rot), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_heads * self.v_head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _forward_inference(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache],
        cache_position: Optional[torch.LongTensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_length, _ = hidden_states.shape
        # Ensure that `inference_q_decode`, `inference_v_proj` are up-2-date
        self._update_inference_params()

        # Encoding: Input X to [C_KV, K_R]
        c_kv, k_rot = self.kv_a_proj_with_mqa(hidden_states).split(
            (self.kv_lora_rank, self.qk_rope_head_dim), dim=-1
        )
        k_rot = k_rot.unsqueeze(-2)
        # k_rot: (batch_size, seq_length, 1, qk_rope_head_dim)
        c_kv = self.kv_a_layernorm(c_kv).unsqueeze(-2)
        # c_kv: (batch_size, seq_length, 1, kv_lora_rank)
        c_q = self.q_a_layernorm(self.q_a_proj(hidden_states))
        # Decoding to Q equivalent
        q_nope, q_rot = torch.matmul(
            c_q,
            self.inference_q_decode.unsqueeze(0)
        ).view(
            batch_size, seq_length, self.num_heads, -1
        ).split(
            (self.kv_lora_rank, self.qk_rope_head_dim),
            dim=-1
        )
        # q_nope: (batch_size, seq_length, num_heads, kv_lora_rank)
        # q_rot: (batch_size, seq_length, num_heads, qk_rope_head_dim)
        # RoPE
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb(
            q_rot, k_rot, cos, sin, unsqueeze_dim=-2
        )
        # Reshape and transpose
        kv_cache_dim = self.kv_lora_rank + self.qk_rope_head_dim
        k_equiv = torch.cat((c_kv, k_rot), dim=-1).view(
            batch_size, 1, seq_length, kv_cache_dim
        )
        q_equiv = torch.cat((q_nope, q_rot), dim=-1).transpose(1, 2)
        # q_equiv: (batch_size, num_heads, seq_length, kv_lora_rank + qk_rope_head_dim)
        # k_equiv: (batch_size, 1, seq_length, kv_lora_rank + qk_rope_head_dim)

        if past_key_value is not None:
            # The KV cache has to maintain a single tensor only. We provide a bogus
            # tensor for value.
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            bogus_value_states = k_equiv[0, 0, :, 0].view(1, 1, -1, 1)
            k_equiv, _ = past_key_value.update(
                key_states=k_equiv,
                value_states=bogus_value_states,
                layer_idx=self.layer_idx,
                cache_kwargs=cache_kwargs,
            )
        v_equiv = k_equiv[..., :self.kv_lora_rank]
        # k_equiv: (batch_size, 1, cache_length, kv_lora_rank + qk_rope_head_dim)
        # v_equiv: (batch_size, 1, cache_length, kv_lora_rank), part of `k_equiv`

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query=q_equiv,
            key=k_equiv,
            value=v_equiv,
            attention_mask=attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # attn_output: (batch_size, seq_length, num_heads, kv_lora_rank)
        # inference_v_proj: (num_heads, kv_lora_rank, v_head_dim)
        attn_output = torch.matmul(
            attn_output.transpose(1, 2),
            self.inference_v_proj.unsqueeze(0),
        ).transpose(1, 2).reshape(
            batch_size, seq_length, self.num_heads * self.v_head_dim
        )
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def _convert_params_training_to_inference(self):
        q_b_1, q_b_2 = self.q_b_proj.weight.view(
            self.num_heads, -1, self.q_lora_rank
        ).split(
            (self.qk_nope_head_dim, self.qk_rope_head_dim),
            dim=1
        )
        kv_b_1, kv_b_2 = self.kv_b_proj.weight.view(
            self.num_heads, -1, self.kv_lora_rank
        ).split(
            (self.qk_nope_head_dim, self.v_head_dim),
            dim=1
        )
        self.inference_v_proj = kv_b_2.transpose(1, 2).contiguous()
        # inference_v_proj: (num_heads, kv_lora_rank, v_head_dim)

        # inference_q_decode from q_b_1, kv_b_1, and q_b_2
        # matmul in float32, to decrease numerical errors
        dtype = q_b_1.dtype
        q_decode_1 = torch.matmul(
            kv_b_1.transpose(1, 2).to(dtype=torch.float32),
            q_b_1.to(dtype=torch.float32),
        ).to(dtype=dtype).permute(2, 0, 1)
        q_b_2 = q_b_2.permute(2, 0, 1)
        self.inference_q_decode = torch.cat(
            (q_decode_1, q_b_2), dim=-1
        ).reshape(self.q_lora_rank, -1).contiguous()
        # inference_q_decode: (q_lora_rank, num_heads * (kv_lora_rank + qk_rope_head_dim))

    def _need_to_update_inference_params(self) -> bool:
        if self.inference_v_proj is None or self.inference_q_decode is None:
            return True
        if self._inference_params_fingerprint is None:
            return True
        return not self._inference_params_fingerprint.match(
            self.q_b_proj.weight, self.kv_b_proj.weight
        )

    def _update_inference_params(self):
        if self._need_to_update_inference_params():
            self.reset_inference_params()

    def reset_inference_params(self):
        """
        As detailed in the header comment, :meth:`forward` in inference mode
        requires some derived parameters, which are typically recomputed
        whenever primary model parameters change. Ths test for changes us not
        perfect. Calling this method forces the inference parameters to be
        recomputed.

        """
        if not self.config.attention_no_inference_mode:
            self._convert_params_training_to_inference()
            self._inference_params_fingerprint = AttentionParametersFingerprint(
                self.q_b_proj.weight, self.kv_b_proj.weight,
            )


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DeepseekV3Attention(
            config=config, layer_idx=layer_idx
        )

        if layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepseekV3MoE(config)
        else:
            self.mlp = DeepseekV3MLP(config)

        self.input_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = (torch.zeros((1,), device=hidden_states.device, dtype=torch.int64),)

        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class DeepseekV3PreTrainedModel(LlamaPreTrainedModel):
    pass


class DeepseekV3Model(LlamaModel):
    def __init__(self, config: DeepseekV3Config):
        super().__init__(config)
        self._register_load_state_dict_pre_hook(self.load_pre_hook)
        self._register_state_dict_hook(self.load_hook)
        self.post_init()
        self.layers = nn.ModuleList(
            [DeepseekV3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV3RotaryEmbedding(config=config)

    def load_pre_hook(self, state_dict, prefix, *args):
        """
        Weights have to be permuted for correct rope formulation. We can't do this in the weights
        as every other framework already uses the `Llama` original function (which is copyrighted btw).
        And I am not even sure it's better.... anyways end of my rant
        """

        def permute_for_rope(input_tensor):
            """
            When you go from the complex ROPE formulation to sin and cos one, you need
            to permute the query and key weights (to avoid doing it on the fly)
            """
            n_heads, dim1, dim2 = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2]
            input_tensor = input_tensor.reshape(n_heads * dim1, dim2)
            input_tensor = input_tensor.view(n_heads, dim1 // 2, 2, dim2)
            input_tensor = input_tensor.transpose(1, 2).reshape(n_heads, dim1, dim2)
            return input_tensor

        def permute_layer_for_rope(key, num_heads, head_dim, rope_dim):
            weight = state_dict[key]
            weight = weight.view(num_heads, head_dim, -1)
            weight_rot = weight[:, -rope_dim:]
            weight_rot = permute_for_rope(weight_rot)
            weight[:, -rope_dim:] = weight_rot
            weight = weight.view(-1, weight.shape[-1])
            state_dict[key] = weight

        for k in state_dict:
            if "q_b_proj." in k:
                permute_layer_for_rope(
                    k,
                    num_heads=self.config.num_attention_heads,
                    head_dim=self.config.qk_head_dim,
                    rope_dim=self.config.qk_rope_head_dim,
                )
            if "kv_a_proj_with_mqa." in k:
                permute_layer_for_rope(
                    k,
                    num_heads=1,
                    head_dim=self.config.kv_lora_rank + self.config.qk_rope_head_dim,
                    rope_dim=self.config.qk_rope_head_dim,
                )

    def load_hook(self, module, state_dict, prefix, *args):
        self.load_pre_hook(state_dict, prefix, *args)


class DeepseekV3ForCausalLM(LlamaForCausalLM):
    pass


__all__ = [
    "DeepseekV3PreTrainedModel",
    "DeepseekV3Model",
    "DeepseekV3ForCausalLM",
]
