import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple

class MultiAxisMLP(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        def mlp():
            return nn.Sequential([
                nn.Dense(self.hidden_dim),
                nn.gelu,
                nn.Dropout(self.dropout, deterministic=deterministic),
                nn.Dense(self.dim),
                nn.Dropout(self.dropout, deterministic=deterministic)
            ])

        B, H, W, C = x.shape
        
        x_h = mlp()(jnp.reshape(jnp.transpose(x, (0, 3, 1, 2)), (-1, C)))
        x_h = jnp.transpose(jnp.reshape(x_h, (B, W, C, H)), (0, 2, 3, 1))
        
        x_w = mlp()(jnp.reshape(jnp.transpose(x, (0, 2, 1, 3)), (-1, C)))
        x_w = jnp.transpose(jnp.reshape(x_w, (B, H, C, W)), (0, 2, 1, 3))
        
        x_c = mlp()(jnp.reshape(jnp.transpose(x, (0, 3, 1, 2)), (B, -1, C)))
        x_c = jnp.reshape(jnp.transpose(x_c, (0, 2, 1)), (B, C, H, W))
        
        return x_h + x_w + x_c

class MAXIMBlock(nn.Module):
    dim: int
    hidden_dim: int
    mlp_ratio: float = 4.
    dropout: float = 0.0
    layerscale_init: float = 1e-5

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        norm1 = nn.LayerNorm()(x)
        attn = MultiAxisMLP(self.dim, self.hidden_dim, self.dropout)(norm1, deterministic)
        ls1 = self.param('ls1', nn.initializers.constant(self.layerscale_init), (self.dim,))
        x = x + ls1[None, None, None, :] * attn

        norm2 = nn.LayerNorm()(x)
        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        mlp = nn.Sequential([
            nn.Dense(mlp_hidden_dim),
            nn.gelu,
            nn.Dropout(self.dropout, deterministic=deterministic),
            nn.Dense(self.dim),
            nn.Dropout(self.dropout, deterministic=deterministic)
        ])
        ls2 = self.param('ls2', nn.initializers.constant(self.layerscale_init), (self.dim,))
        x = x + ls2[None, None, None, :] * mlp(norm2)
        return x

class MAXIM(nn.Module):
    in_channels: int = 3
    out_channels: int = 3
    dim: int = 64
    depth: int = 4
    hidden_dim: int = 128
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        x = nn.Conv(self.dim, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.Conv(self.dim, kernel_size=(3, 3), padding='SAME')(x)

        for _ in range(self.depth):
            x = MAXIMBlock(self.dim, self.hidden_dim, dropout=self.dropout)(x, deterministic)

        x = nn.Conv(self.dim, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)
        return x
