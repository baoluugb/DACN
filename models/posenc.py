
import math, torch, torch.nn as nn

class SinusoidalPositionalEncoding1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class SinusoidalPositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int, max_h: int = 256, max_w: int = 1024):
        super().__init__()
        d_half = d_model // 2
        pe_h = torch.zeros(max_h, d_half)
        pe_w = torch.zeros(max_w, d_model - d_half)
        pos_h = torch.arange(0, max_h, dtype=torch.float).unsqueeze(1)
        pos_w = torch.arange(0, max_w, dtype=torch.float).unsqueeze(1)
        div_h = torch.exp(torch.arange(0, d_half, 2).float() * (-math.log(10000.0) / d_half))
        div_w = torch.exp(torch.arange(0, (d_model - d_half), 2).float() * (-math.log(10000.0) / (d_model - d_half)))
        pe_h[:, 0::2] = torch.sin(pos_h * div_h); pe_h[:, 1::2] = torch.cos(pos_h * div_h)
        pe_w[:, 0::2] = torch.sin(pos_w * div_w); pe_w[:, 1::2] = torch.cos(pos_w * div_w)
        self.register_buffer('pe_h', pe_h, persistent=False)
        self.register_buffer('pe_w', pe_w, persistent=False)
        self.d_model = d_model

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        pe_h = self.pe_h[:H]; pe_w = self.pe_w[:W]
        pe = torch.cat([pe_h[:, None, :].expand(H, W, -1),
                        pe_w[None, :, :].expand(H, W, -1)], dim=-1)
        pe = pe.reshape(1, H*W, C).to(feat.dtype).to(feat.device)
        x = feat.flatten(2).permute(0, 2, 1)
        return x + pe
