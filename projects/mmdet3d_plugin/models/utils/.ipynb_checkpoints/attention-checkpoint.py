# ------------------------------------------------------------------------
# Modified to remove flash-attn dependency, using standard PyTorch attention
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_
from torch.nn.functional import linear
from einops import rearrange
from mmcv.runner import auto_fp16
from mmcv.runner.base_module import BaseModule


def _in_projection_packed(q, k, v, w, b=None):
    w_q, w_k, w_v = w.chunk(3)
    if b is None:
        b_q = b_k = b_v = None
    else:
        b_q, b_k, b_v = b.chunk(3)
    return linear(q, w_q, b_q), linear(k, w_k, b_k), linear(v, w_v, b_v)


class FlashAttention(nn.Module):
    """Drop-in replacement using standard PyTorch scaled dot-product attention."""
    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.fp16_enabled = True

    def forward(self, q, kv, causal=False, key_padding_mask=None):
        """
        q:  (B, T, H, D)
        kv: (B, S, 2, H, D)
        """
        k, v = kv.unbind(dim=2)  # each (B, S, H, D)

        # Rearrange to (B, H, T/S, D) for PyTorch attention
        q = rearrange(q, 'b t h d -> b h t d')
        k = rearrange(k, 'b s h d -> b h s d')
        v = rearrange(v, 'b s h d -> b h s d')

        scale = self.softmax_scale or (q.shape[-1] ** -0.5)

        # Build attention mask
        attn_mask = None
        if key_padding_mask is not None:
            # (B, S) -> (B, 1, 1, S)
            attn_mask = key_padding_mask[:, None, None, :].logical_not()
            attn_mask = attn_mask.expand(-1, q.shape[1], q.shape[2], -1)

        if causal:
            T, S = q.shape[2], k.shape[2]
            causal_mask = torch.triu(torch.ones(T, S, device=q.device, dtype=torch.bool), diagonal=1)
            if attn_mask is None:
                attn_mask = causal_mask
            else:
                attn_mask = attn_mask | causal_mask

        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = torch.dropout(attn, self.dropout_p if self.training else 0.0, self.training)

        out = torch.matmul(attn, v)  # (B, H, T, D)
        out = rearrange(out, 'b h t d -> b t h d')
        return out, None


class FlashMHA(nn.Module):

    def __init__(self, embed_dim, num_heads, bias=True, batch_first=True, attention_dropout=0.0,
                 causal=False, device=None, dtype=None, **kwargs) -> None:
        assert batch_first
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads

        self.in_proj_weight = nn.Parameter(torch.empty((3 * embed_dim, embed_dim)))
        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.inner_attn = FlashAttention(attention_dropout=attention_dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, key_padding_mask=None):
        q, k, v = _in_projection_packed(q, k, v, self.in_proj_weight, self.in_proj_bias)
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        kv = torch.stack([k, v], dim=2)

        context, attn_weights = self.inner_attn(q, kv, key_padding_mask=key_padding_mask, causal=self.causal)
        return self.out_proj(rearrange(context, 'b s h d -> b s (h d)')), attn_weights