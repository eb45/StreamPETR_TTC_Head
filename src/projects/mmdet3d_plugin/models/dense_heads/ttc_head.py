"""Lightweight MLP mapping per-query embeddings to scalar TTC in seconds."""
import torch
import torch.nn as nn


class TTCRiskHead(nn.Module):
    """Phase 3 architecture (configurable width / depth).

    Query features only (no explicit velocity input). See ``TTCRiskHeadV3`` for embed+vel.

    Default pattern matches the project spec (two internal hidden widths):
        LN(embed) -> Linear(embed->H) -> GELU -> Dropout
        -> Linear(H->H//2) -> GELU -> Dropout
        -> Linear(H//2->1) -> sigmoid * 10

    ``num_layers`` controls extra residual-width blocks: each adds
    ``Linear(H, H) -> GELU -> Dropout`` after the first projection.
    """

    use_velocity = False

    def __init__(
        self,
        embed_dims=256,
        hidden_dim=256,
        num_layers=0,
        dropout=0.1,
        ttc_max=10.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.ttc_max = ttc_max

        layers = [nn.LayerNorm(embed_dims)]
        in_dim = embed_dims
        mid = hidden_dim
        layers.extend(
            [
                nn.Linear(in_dim, mid),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        for _ in range(num_layers):
            layers.extend([nn.Linear(mid, mid), nn.GELU(), nn.Dropout(dropout)])
        layers.extend(
            [
                nn.Linear(mid, max(mid // 2, 1)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(mid // 2, 1), 1),
                nn.Sigmoid(),
            ]
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """Args:
            x: (*, embed_dims)
        Returns:
            (*, ) TTC in [0, ttc_max]
        """
        out = self.mlp(x).squeeze(-1) * self.ttc_max
        return out


class TTCRiskHeadV3(nn.Module):
    """TTC head ablation: fused **query embedding** + **predicted BEV velocity** (vx, vy).

    Same MLP pattern as ``TTCRiskHead`` after the first projection, but the input is
    ``[LayerNorm(embed); vx / vel_scale; vy / vel_scale]`` so the head can combine
    appearance/context from the decoder with explicit motion from the (frozen) bbox
    regression stream. Indices 8:10 on ``all_bbox_preds`` match ``ttc_pairwise_physics_from_preds``.
    """

    use_velocity = True

    def __init__(
        self,
        embed_dims=256,
        hidden_dim=256,
        num_layers=0,
        dropout=0.1,
        ttc_max=10.0,
        vel_scale=30.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.ttc_max = ttc_max
        self.vel_scale = float(vel_scale)

        self.embed_ln = nn.LayerNorm(embed_dims)
        mid = hidden_dim
        layers = [
            nn.Linear(embed_dims + 2, mid),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(num_layers):
            layers.extend([nn.Linear(mid, mid), nn.GELU(), nn.Dropout(dropout)])
        layers.extend(
            [
                nn.Linear(mid, max(mid // 2, 1)),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(max(mid // 2, 1), 1),
                nn.Sigmoid(),
            ]
        )
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, vel_xy=None):
        """Args:
            x: (*, embed_dims) decoder query features (matched queries).
            vel_xy: (*, 2) optional vx, vy in m/s (lidar BEV). Zeros if missing.
        Returns:
            (*,) TTC in [0, ttc_max]
        """
        if vel_xy is None:
            vel_xy = x.new_zeros(x.shape[:-1] + (2,))
        vel = vel_xy.to(dtype=x.dtype) / x.new_tensor(self.vel_scale)
        h = torch.cat([self.embed_ln(x), vel], dim=-1)
        out = self.mlp(h).squeeze(-1) * self.ttc_max
        return out
