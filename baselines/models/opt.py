"""OPT (Open Pre-trained Transformer) model blocks.

Implements the OPT architecture (Zhang et al., 2022) used in
OPT-1.3B and other OPT variants.  Key differences from GPT-2:

- ReLU activation in MLP (not GELU)
- Learned positional embeddings (same as GPT-2)
- Pre-norm (LayerNorm before attention and FFN)
- No weight tying between embedding and LM head
- Standard MHA (no GQA / rotary embeddings)

Typical OPT-1.3B config:
  num_layers=24, embedding_dim=2048, num_heads=32,
  d_ff=8192, vocab_size=50272, max_seq_len=2048
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class OPTAttention(nn.Module):
    """OPT multi-head self-attention with causal masking."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # OPT uses separate Q/K/V projections (not fused c_attn)
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(
                1, 1, max_seq_len, max_seq_len,
            ),
        )
        self.use_flash = use_flash and hasattr(
            F, "scaled_dot_product_attention",
        )

    def forward(self, x: Tensor) -> Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_heads, self.head_dim)
        v = v.transpose(1, 2)

        if self.use_flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=(
                    self.attn_dropout.p if self.training else 0.0
                ),
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (
                1.0 / math.sqrt(self.head_dim)
            )
            att = att.masked_fill(
                self.bias[:, :, :S, :S] == 0, float("-inf"),
            )
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, S, -1)
        return self.resid_dropout(self.out_proj(y))


class OPTMLP(nn.Module):
    """OPT feed-forward block with ReLU activation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff, bias=True)
        self.fc2 = nn.Linear(d_ff, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(F.relu(self.fc1(x))))


class OPTBlock(nn.Module):
    """OPT pre-norm transformer block.

    Architecture:
        x → LayerNorm → Attention → + → LayerNorm → MLP → +
        └─────────────── residual ──┘ └──── residual ──────┘
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int,
        dropout: float = 0.0,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.attn = OPTAttention(
            d_model, n_heads, max_seq_len, dropout,
            use_flash=use_flash,
        )
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.mlp = OPTMLP(d_model, d_ff, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.self_attn_layer_norm(x))
        x = x + self.mlp(self.final_layer_norm(x))
        return x


__all__ = [
    "OPTAttention",
    "OPTMLP",
    "OPTBlock",
]
