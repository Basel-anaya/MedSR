import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiAxisMLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.mlp_h = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.mlp_w = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_h = self.mlp_h(x.permute(0, 3, 1, 2).reshape(-1, C)).reshape(B, W, C, H).permute(0, 2, 3, 1)
        x_w = self.mlp_w(x.permute(0, 2, 1, 3).reshape(-1, C)).reshape(B, H, C, W).permute(0, 2, 1, 3)
        x_c = self.mlp_c(x.reshape(B, C, -1).transpose(1, 2)).transpose(1, 2).reshape(B, C, H, W)
        return x_h + x_w + x_c

class MAXIMBlock(nn.Module):
    def __init__(self, dim, hidden_dim, mlp_ratio=4., dropout=0.0, layerscale_init=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiAxisMLP(dim, hidden_dim, dropout)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.ls1 = nn.Parameter(layerscale_init * torch.ones(dim))
        self.ls2 = nn.Parameter(layerscale_init * torch.ones(dim))

    def forward(self, x):
        x = x + self.ls1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        x = x + self.ls2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x.permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        return x

class MAXIM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, dim=64, depth=4, hidden_dim=128, dropout=0.0):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        self.blocks = nn.ModuleList([
            MAXIMBlock(dim, hidden_dim, dropout=dropout) for _ in range(depth)
        ])
        self.head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.head(x)
        return x