"""Lightweight MLP mapping per-query embeddings to scalar TTC in seconds."""
import torch
import torch.nn as nn


class TTCRiskHead(nn.Module):
    """Phase 3 architecture (configurable width / depth).

    Default pattern matches the project spec (two internal hidden widths):
        LN(embed) -> Linear(embed->H) -> GELU -> Dropout
        -> Linear(H->H//2) -> GELU -> Dropout
        -> Linear(H//2->1) -> sigmoid * 10

    ``num_layers`` controls extra residual-width blocks: each adds
    ``Linear(H, H) -> GELU -> Dropout`` after the first projection.
    """

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
