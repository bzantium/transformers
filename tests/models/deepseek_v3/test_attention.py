from typing import Optional

import pytest

from transformers import is_torch_available
from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3Model
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RotaryEmbedding,
)
from transformers.cache_utils import DynamicCache, Cache
from ...test_modeling_common import ids_tensor


if is_torch_available():
    import torch


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)


def get_causal_attention_mask(
    model: DeepseekV3Model,
    batch_size: int,
    seq_len: int,
    past_key_value: Cache,
    cache_position: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
) -> torch.Tensor:
    input_ids = ids_tensor((batch_size, seq_len), model.config.vocab_size)
    inputs_embeds = model.embed_tokens(input_ids)
    causal_mask = model._update_causal_mask(
        attention_mask=attention_mask,
        input_tensor=inputs_embeds,
        cache_position=cache_position,
        past_key_values=past_key_value,
        output_attentions=output_attentions,
    )
    return causal_mask


@pytest.mark.parametrize(
    "batch_size, num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, q_lora_rank",
    [
        (2, 8, 12, 8, 12, 36, 48),
        (2, 1, 64, 8, 48, 32, 128),
        (1, 8, 12, 6, 16, 36, 48),
        (2, 8, 8, 16, 12, 2, 64),
    ]
)
def test_compare_training_vs_inference_mode(
    batch_size,
    num_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    kv_lora_rank,
    q_lora_rank,
):
    seq_len = 20
    dtype = torch.float32
    attn_implementation = "eager"
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        attn_implementation=attn_implementation,
        attention_bias=False,
        num_hidden_layers=1,
        vocab_size=128,
        intermediate_size=3 * 64,
        moe_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )

    attn=DeepseekV3Attention(config, layer_idx=0, weights_dtype=dtype)
    attn.apply(init_weights)
    rotary_emb = DeepseekV3RotaryEmbedding(config)

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size),
        dtype=dtype,
    )
    position_ids = torch.arange(seq_len, dtype=torch.int64).expand(batch_size, -1)
    position_embeddings = rotary_emb(hidden_states, position_ids)

    # Just to compute attention mask:
    bogus_model = DeepseekV3Model(config)
    past_key_value = DynamicCache()
    cache_position = position_ids[0]
    causal_mask = get_causal_attention_mask(
        model=bogus_model,
        batch_size=batch_size,
        seq_len=seq_len,
        past_key_value=past_key_value,
        cache_position=cache_position,
        output_attentions=True,
    )[:, :, :, :seq_len]

    output1, weights1 = attn._forward_training(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
        past_key_value=None,
        cache_position=None,
    )
    output2, weights2 = attn._forward_inference(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
        past_key_value=None,
        cache_position=None,
    )
    if attn_implementation == "eager":
        assert weights1 is not None and weights2 is not None
    torch.testing.assert_close(output1, output2)
    if weights1 is not None:
        torch.testing.assert_close(weights1, weights2)


@pytest.mark.parametrize(
    "batch_size, num_heads, qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank, q_lora_rank, seq_len, chunk_size",
    [
        (2, 8, 12, 8, 12, 36, 48, 20, 1),
        (2, 1, 64, 8, 48, 32, 128, 20, 2),
        (1, 8, 12, 6, 16, 36, 48, 20, 4),
        (2, 8, 8, 16, 12, 2, 64, 20, 5),
    ]
)
def test_batch_vs_sequential_inference(
    batch_size,
    num_heads,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    kv_lora_rank,
    q_lora_rank,
    seq_len,
    chunk_size,
):
    assert seq_len % chunk_size == 0
    dtype = torch.float32
    attn_implementation = "eager"
    config = DeepseekV3Config(
        hidden_size=64,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        kv_lora_rank=kv_lora_rank,
        q_lora_rank=q_lora_rank,
        attn_implementation=attn_implementation,
        attention_bias=False,
        num_hidden_layers=1,
        vocab_size=128,
        intermediate_size=3 * 64,
        moe_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )

    bogus_model = DeepseekV3Model(config)  # Just for attention mask
    past_key_value = DynamicCache()
    attn=DeepseekV3Attention(config, layer_idx=0, weights_dtype=dtype)
    attn.apply(init_weights)
    rotary_emb = DeepseekV3RotaryEmbedding(config)
    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size),
        dtype=dtype,
    )

    # Single forward pass
    position_ids = torch.arange(seq_len, dtype=torch.int64).expand(batch_size, -1)
    position_embeddings = rotary_emb(hidden_states, position_ids)
    causal_mask = get_causal_attention_mask(
        model=bogus_model,
        batch_size=batch_size,
        seq_len=seq_len,
        past_key_value=past_key_value,
        cache_position=position_ids[0],
    )[:, :, :, :seq_len]
    output1, _ = attn._forward_training(
        hidden_states=hidden_states,
        position_embeddings=position_embeddings,
        attention_mask=causal_mask,
        past_key_value=None,
        cache_position=None,
    )

    # Sequential computation, involving KV cache
    min_dtype = torch.finfo(dtype).min
    for position_id in range(0, seq_len, chunk_size):
        cache_position = torch.arange(
            position_id, position_id + chunk_size, dtype=torch.int64
        )
        causal_mask = torch.zeros(
            (chunk_size, position_id + chunk_size), dtype=dtype
        )
        if chunk_size > 1:
            causal_mask[:, position_id:] = min_dtype
            causal_mask[:, position_id:] = torch.triu(
                causal_mask[:, position_id:], diagonal=1
            )
        causal_mask = causal_mask[None, None, ...]
        input = hidden_states[:, position_id:(position_id + chunk_size), :]
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = rotary_emb(input, position_ids)
        output2, _ = attn._forward_training(
            hidden_states=input,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
        )
        print(f"position_id {position_id}, chunk_size {chunk_size}")
        torch.testing.assert_close(output1[:, position_id:(position_id + chunk_size), :], output2)
