import torch
import torch.nn as nn
from typing import Optional

class QFormerBlock(nn.Module):
    def __init__(self, query_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(query_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(query_dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(query_dim, 4*query_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*query_dim, query_dim),
        )
        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        q_sa, _ = self.self_attn(q, q, q, need_weights=False)
        q = q + self.drop(q_sa); q = self.norm1(q)
        q_ca, _ = self.cross_attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        q = q + self.drop(q_ca); q = self.norm2(q)
        q_ff = self.ff(q)
        q = q + self.drop(q_ff); q = self.norm3(q)
        return q


class QFormer(nn.Module):
    def __init__(self, 
            d_model: int, 
            query_dim: int = 256, 
            num_heads: int = 8, 
            num_layers: int = 4, 
            num_queries: int = 16, 
            dropout: float = 0.1
        ):

        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Parameter(torch.randn(num_queries, query_dim) * 0.02)

        self.vis_proj_k = nn.Linear(d_model, query_dim)
        self.vis_proj_v = nn.Linear(d_model, query_dim)

        self.layers = nn.ModuleList([QFormerBlock(query_dim, num_heads, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(query_dim)

    def forward(self, visual_tokens: torch.Tensor, visual_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = visual_tokens.size(0)
        q = self.query_embed.unsqueeze(0).expand(B, -1, -1)  # (B,Q,Dq)

        k = self.vis_proj_k(visual_tokens)  # (B,S,Dq)
        v = self.vis_proj_v(visual_tokens)  # (B,S,Dq)

        attn_mask = None
        if visual_mask is not None:
            attn_mask = ~visual_mask.bool()  # True -> mask

        for blk in self.layers:
            q = blk(q, k, v, key_padding_mask=attn_mask)
        q = self.norm(q)
        return q  # (B,Q,Dq)
