from typing import Optional, Tuple

import torch


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def sdpa_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    is_causal: Optional[bool] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    `query` has shape `(batch, num_heads, q_len, head_dim)`, while `key`,
    `value` have shape `(batch, num_key_value_groups, kv_len, head_dim)`. Here,
    `num_key_value_groups <= num_heads` and
    `num_heads % num_key_value_groups == 0`.

    https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch-nn-functional-scaled-dot-product-attention
    `scaled_dot_product_attention` is supposed to support
    `num_key_value_groups < num_heads`. But at least up to PyTorch 2.5.1, this
    does not seem to be supported. The implementation here tries to use the
    feature and otherwise broadcasts `key` and `value` accordingly and tries again.

    """
    assert query.ndim == key.ndim == value.ndim == 4
    num_key_value_groups = key.shape[1]
    assert value.shape[1] == num_key_value_groups
    num_heads = query.shape[1]
    assert num_heads % num_key_value_groups == 0
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, : key.shape[-2]]
    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    if is_causal is None:
        is_causal = causal_mask is None and query.shape[2] > 1

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    for retry in range(2):
        try:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=causal_mask,
                dropout_p=dropout,
                scale=scaling,
                is_causal=is_causal,
            )
            break
        except RuntimeError as ex:
            if retry == 1 or num_key_value_groups == num_heads:
                raise ex  # Re-throw
            q_per_kv = num_heads // num_key_value_groups
            key = repeat_kv(key, n_rep=q_per_kv).contiguous()
            value = repeat_kv(value, n_rep=q_per_kv).contiguous()

    attn_output = attn_output.transpose(1, 2).contiguous()
    # attn_output: (batch, q_len, num_heads, head_dim)
    return attn_output, None
