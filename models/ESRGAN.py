import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple, List, Optional

class ResidualDenseBlock(nn.Module):
    nf: int = 64
    gc: int = 32
    bias: bool = True

    @nn.compact
    def __call__(self, x):
        x1 = nn.leaky_relu(nn.Conv(self.gc, kernel_size=(3, 3), padding='SAME', use_bias=self.bias)(x), negative_slope=0.2)
        x2 = nn.leaky_relu(nn.Conv(self.gc, kernel_size=(3, 3), padding='SAME', use_bias=self.bias)(jnp.concatenate([x, x1], axis=-1)), negative_slope=0.2)
        x3 = nn.leaky_relu(nn.Conv(self.gc, kernel_size=(3, 3), padding='SAME', use_bias=self.bias)(jnp.concatenate([x, x1, x2], axis=-1)), negative_slope=0.2)
        x4 = nn.leaky_relu(nn.Conv(self.gc, kernel_size=(3, 3), padding='SAME', use_bias=self.bias)(jnp.concatenate([x, x1, x2, x3], axis=-1)), negative_slope=0.2)
        x5 = nn.Conv(self.nf, kernel_size=(3, 3), padding='SAME', use_bias=self.bias)(jnp.concatenate([x, x1, x2, x3, x4], axis=-1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    nf: int
    gc: int = 32
    res_scale: float = 0.2

    @nn.compact
    def __call__(self, x):
        out = ResidualDenseBlock(self.nf, self.gc)(x)
        out = ResidualDenseBlock(self.nf, self.gc)(out)
        out = ResidualDenseBlock(self.nf, self.gc)(out)
        return out * self.res_scale + x

class ChannelAttention(nn.Module):
    num_feat: int
    reduction: int = 16

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        y = jnp.mean(x, axis=(1, 2))
        y = nn.Dense(self.num_feat // self.reduction, use_bias=False)(y)
        y = nn.relu(y)
        y = nn.Dense(self.num_feat, use_bias=False)(y)
        y = jax.nn.sigmoid(y)
        y = y.reshape(b, 1, 1, c)
        return x * y

def pixel_shuffle_upscale(x, factor=2):
    b, h, w, c = x.shape
    x = x.reshape(b, h, w, factor, factor, c//(factor**2))
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = x.reshape(b, h*factor, w*factor, c//(factor**2))
    return x

class RealESRGANGenerator(nn.Module):
    num_in_ch: int = 3
    num_out_ch: int = 3
    scale: int = 4
    num_feat: int = 64
    num_block: int = 23
    num_grow_ch: int = 32
    res_scale: float = 0.2
    use_attention: bool = False
    beta_channel: bool = False
    final_activation: str = 'tanh'

    @nn.compact
    def __call__(self, x):
        if self.scale == 2:
            x = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            x = pixel_unshuffle(x, scale=4)

        feat = nn.Conv(self.num_feat, kernel_size=(3, 3), padding='SAME')(x)
        body_feat = feat
        for _ in range(self.num_block):
            body_feat = RRDB(self.num_feat, self.num_grow_ch, self.res_scale)(body_feat)
        body_feat = nn.Conv(self.num_feat, kernel_size=(3, 3), padding='SAME')(body_feat)

        if self.use_attention:
            body_feat = ChannelAttention(self.num_feat)(body_feat)

        feat = feat + body_feat

        # Upsampling
        if self.scale == 4:
            feat = nn.leaky_relu(pixel_shuffle_upscale(nn.Conv(self.num_feat * 4, kernel_size=(3, 3), padding='SAME')(feat)), negative_slope=0.2)
            feat = nn.leaky_relu(pixel_shuffle_upscale(nn.Conv(self.num_feat * 4, kernel_size=(3, 3), padding='SAME')(feat)), negative_slope=0.2)
        elif self.scale == 2:
            feat = nn.leaky_relu(pixel_shuffle_upscale(nn.Conv(self.num_feat * 4, kernel_size=(3, 3), padding='SAME')(feat)), negative_slope=0.2)

        feat = nn.leaky_relu(nn.Conv(self.num_feat, kernel_size=(3, 3), padding='SAME')(feat), negative_slope=0.2)
        out = nn.Conv(self.num_out_ch, kernel_size=(3, 3), padding='SAME')(feat)

        if self.final_activation == 'tanh':
            out = jnp.tanh(out)
        elif self.final_activation == 'sigmoid':
            out = jax.nn.sigmoid(out)

        if self.beta_channel:
            beta = nn.Conv(self.num_out_ch, kernel_size=(3, 3), padding='SAME')(feat)
            return out, beta
        else:
            return out

class Discriminator(nn.Module):
    input_shape: Tuple[int, int, int]

    @nn.compact
    def __call__(self, img, train: bool = True):
        _, in_height, in_width, in_channels = img.shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)

        def discriminator_block(x, out_filters, first_block=False):
            x = nn.Conv(out_filters, kernel_size=(3, 3), padding='SAME')(x)
            if not first_block:
                x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            x = nn.Conv(out_filters, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.leaky_relu(x, negative_slope=0.2)
            return x

        x = img
        for i, out_filters in enumerate([64, 128, 256, 512]):
            x = discriminator_block(x, out_filters, first_block=(i == 0))

        x = nn.Conv(1, kernel_size=(3, 3), padding='SAME')(x)
        return x

def pixel_unshuffle(x, scale):
    b, h, w, c = x.shape
    out_channel = c * (scale ** 2)
    out_h = h // scale
    out_w = w // scale
    x = x.reshape(b, out_h, scale, out_w, scale, c)
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
    x = x.reshape(b, out_h, out_w, out_channel)
    return x
