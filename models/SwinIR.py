import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple, List, Optional
import math

class SwinIR(nn.Module):
    img_size: int = 64
    patch_size: int = 1
    in_chans: int = 3
    embed_dim: int = 96
    depths: List[int] = (6, 6, 6, 6)
    num_heads: List[int] = (6, 6, 6, 6)
    window_size: int = 7
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.1
    patch_norm: bool = True
    upscale: int = 2
    upsampler: str = 'pixelshuffledirect'

    @nn.compact
    def __call__(self, x, train: bool = True):
        residual = x
        x = nn.Conv(self.embed_dim, kernel_size=(self.patch_size, self.patch_size), strides=(self.patch_size, self.patch_size))(x)

        if self.patch_norm:
            x = nn.LayerNorm()(jnp.transpose(x, (0, 2, 3, 1)))
            x = jnp.transpose(x, (0, 3, 1, 2))

        B, C, H, W = x.shape
        x = jnp.reshape(jnp.transpose(x, (0, 2, 3, 1)), (B, H * W, C))

        for i_layer in range(len(self.depths)):
            x = SwinTransformerBlock(
                dim=self.embed_dim,
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                shift_size=0 if (i_layer % 2 == 0) else self.window_size // 2,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=self.drop_path_rate,
            )(x, H, W, train)

        x = nn.LayerNorm()(x)
        x = jnp.transpose(jnp.reshape(x, (B, H, W, C)), (0, 3, 1, 2))

        x = x + residual

        if self.upsampler == 'pixelshuffledirect':
            x = nn.Conv(4 * self.in_chans * self.upscale ** 2, kernel_size=(3, 3), padding='SAME')(x)
            x = jax.image.resize(x, (B, self.upscale * H, self.upscale * W, self.in_chans), method='nearest')
        else:
            raise NotImplementedError(f"Upsampler {self.upsampler} is not implemented.")

        return x

class SwinTransformerBlock(nn.Module):
    dim: int
    num_heads: int
    window_size: int = 7
    shift_size: int = 0
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    drop: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.

    @nn.compact
    def __call__(self, x, H, W, train: bool = True):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = nn.LayerNorm()(x)
        x = jnp.reshape(x, (B, H, W, C))

        # Pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
        _, Hp, Wp, _ = x.shape

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = jnp.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = jnp.reshape(x_windows, (-1, self.window_size * self.window_size, C))

        # W-MSA/SW-MSA
        attn_windows = WindowAttention(self.dim, self.window_size, self.num_heads, self.qkv_bias, self.attn_drop)(x_windows, train)

        # Merge windows
        attn_windows = jnp.reshape(attn_windows, (-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = jnp.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = jnp.reshape(x, (B, H * W, C))

        # FFN
        x = shortcut + DropPath(self.drop_path)(x, train)
        x = x + DropPath(self.drop_path)(Mlp(self.dim * self.mlp_ratio, drop=self.drop)(nn.LayerNorm()(x)), train)

        return x

class WindowAttention(nn.Module):
    dim: int
    window_size: int
    num_heads: int
    qkv_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.

    def setup(self):
        head_dim = self.dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = self.param('relative_position_bias_table',
            nn.initializers.truncated_normal(stddev=.02),
            ((2 * self.window_size - 1) * (2 * self.window_size - 1), self.num_heads))

        coords_h = jnp.arange(self.window_size)
        coords_w = jnp.arange(self.window_size)
        coords = jnp.stack(jnp.meshgrid(coords_h, coords_w))
        coords_flatten = coords.reshape(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = jnp.transpose(relative_coords, (1, 2, 0))
        relative_coords = relative_coords.at[:, :, 0].add(self.window_size - 1)
        relative_coords = relative_coords.at[:, :, 1].add(self.window_size - 1)
        relative_coords = relative_coords.at[:, :, 0].multiply(2 * self.window_size - 1)
        relative_position_index = jnp.sum(relative_coords, axis=-1)

        self.relative_position_index = relative_position_index

    @nn.compact
    def __call__(self, x, train: bool = True):
        B_, N, C = x.shape
        qkv = nn.Dense(3 * self.dim, use_bias=self.qkv_bias)(x)
        qkv = jnp.reshape(qkv, (B_, N, 3, self.num_heads, C // self.num_heads))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2)))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.reshape(-1)]
        relative_position_bias = jnp.reshape(relative_position_bias, 
                                             (self.window_size * self.window_size, 
                                              self.window_size * self.window_size, -1))
        relative_position_bias = jnp.transpose(relative_position_bias, (2, 0, 1))
        attn = attn + relative_position_bias[None, ...]

        attn = jax.nn.softmax(attn, axis=-1)
        attn = nn.Dropout(self.attn_drop)(attn, deterministic=not train)

        x = jnp.matmul(attn, v)
        x = jnp.transpose(x, (0, 2, 1, 3))
        x = jnp.reshape(x, (B_, N, C))
        x = nn.Dense(self.dim)(x)
        x = nn.Dropout(self.proj_drop)(x, deterministic=not train)
        return x

class Mlp(nn.Module):
    hidden_features: Optional[int] = None
    out_features: Optional[int] = None
    act_layer: Any = nn.gelu
    drop: float = 0.

    @nn.compact
    def __call__(self, x, train: bool = True):
        out_features = self.out_features or x.shape[-1]
        hidden_features = self.hidden_features or x.shape[-1]
        x = nn.Dense(hidden_features)(x)
        x = self.act_layer(x)
        x = nn.Dropout(self.drop)(x, deterministic=not train)
        x = nn.Dense(out_features)(x)
        x = nn.Dropout(self.drop)(x, deterministic=not train)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = jnp.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = jnp.reshape(windows, (-1, window_size, window_size, C))
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = jnp.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = jnp.reshape(x, (B, H, W, -1))
    return x

def DropPath(drop_prob):
    def drop_path(x, train):
        if drop_prob == 0. or not train:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + jax.random.uniform(jax.random.PRNGKey(0), shape, dtype=x.dtype)
        random_tensor = jnp.floor(random_tensor)
        return jnp.divide(x, keep_prob) * random_tensor
    return drop_path
