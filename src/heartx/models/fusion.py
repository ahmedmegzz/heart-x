"""Multi-modal fusion modules for HEART-X (concat / gated / attention)."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConcatFusion(nn.Module):
    """Baseline feature-level concatenation."""

    def __init__(self, emb_dim: int, n_modalities: int):
        super().__init__()
        self.out_dim = emb_dim * n_modalities
        self.last_weights: torch.Tensor | None = None

    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        # Uniform weights for logging compatibility
        b = parts[0].size(0)
        w = torch.full(
            (b, len(parts)), 1.0 / len(parts), device=parts[0].device, dtype=parts[0].dtype
        )
        self.last_weights = w
        return torch.cat(parts, dim=1)


class GatedFusion(nn.Module):
    """
    Learnable per-modality gates (sigmoid) applied before concatenation.

    For each modality embedding e_i, computes g_i = σ(W_i e_i + b_i) and
    returns concat(g_i ⊙ e_i). Gates are stored in ``last_weights`` as mean
    gate activation per modality (B, M) for analysis / visualization.
    """

    def __init__(self, emb_dim: int, n_modalities: int):
        super().__init__()
        self.gates = nn.ModuleList(
            [nn.Linear(emb_dim, emb_dim) for _ in range(n_modalities)]
        )
        self.out_dim = emb_dim * n_modalities
        self.last_weights: torch.Tensor | None = None

    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        assert len(parts) == len(self.gates)
        gated = []
        weights = []
        for e, gate in zip(parts, self.gates):
            g = torch.sigmoid(gate(e))
            gated.append(g * e)
            weights.append(g.mean(dim=1, keepdim=True))
        self.last_weights = torch.cat(weights, dim=1)  # (B, M)
        return torch.cat(gated, dim=1)


class AttentionFusion(nn.Module):
    """
    Cross-modal attention fusion over modality tokens.

    Treats each modality embedding as a token, applies multi-head self-attention,
    then pools with learned attention weights. Returns a single fused vector of
    size ``emb_dim`` (not concat). ``last_weights`` holds (B, M) attention mass.
    """

    def __init__(self, emb_dim: int, n_modalities: int, n_heads: int = 4):
        super().__init__()
        self.n_modalities = n_modalities
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim, num_heads=n_heads, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, emb_dim) * 0.02)
        self.out_dim = emb_dim
        self.last_weights: torch.Tensor | None = None

    def forward(self, parts: list[torch.Tensor]) -> torch.Tensor:
        # (B, M, D)
        tokens = torch.stack(parts, dim=1)
        b = tokens.size(0)
        q = self.query.expand(b, -1, -1)
        fused, attn_w = self.attn(q, tokens, tokens, need_weights=True, average_attn_weights=True)
        # attn_w: (B, 1, M) or (B, M) depending on version
        if attn_w.dim() == 3:
            self.last_weights = attn_w.squeeze(1)
        else:
            self.last_weights = attn_w
        return fused.squeeze(1)


def build_fusion(
    fusion_type: str,
    emb_dim: int,
    n_modalities: int,
) -> nn.Module:
    fusion_type = fusion_type.lower()
    if fusion_type == "concat":
        return ConcatFusion(emb_dim, n_modalities)
    if fusion_type == "gated":
        return GatedFusion(emb_dim, n_modalities)
    if fusion_type == "attention":
        return AttentionFusion(emb_dim, n_modalities)
    raise ValueError(f"Unknown fusion_type: {fusion_type}")
