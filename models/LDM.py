import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Any, Tuple
import math

class ResidualBlock(nn.Module):
    in_channels: int
    out_channels: int

    @nn.compact
    def __call__(self, x):
        residual = nn.Conv(self.out_channels, kernel_size=(1, 1))(x) if self.in_channels != self.out_channels else x
        out = nn.relu(nn.BatchNorm(use_running_average=not self.is_training)(nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(x)))
        out = nn.BatchNorm(use_running_average=not self.is_training)(nn.Conv(self.out_channels, kernel_size=(3, 3), padding='SAME')(out))
        out += residual
        return nn.relu(out)

class VAE(nn.Module):
    in_channels: int = 3
    latent_dim: int = 4

    @nn.compact
    def __call__(self, x):
        # Encoder
        x = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(128, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)

        b, h, w, c = x.shape
        x = x.reshape((b, h * w * c))

        x = nn.Dense(256)(x)
        x = nn.relu(x)
        mu = nn.Dense(self.latent_dim * h * w)(x)
        logvar = nn.Dense(self.latent_dim * h * w)(x)

        # Reparameterization trick
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(self.make_rng('dropout'), logvar.shape)
        z = mu + eps * std

        # Decoder
        z = z.reshape((b, self.latent_dim, h, w))
        x = nn.Conv(64, kernel_size=(3, 3), padding='SAME')(z)
        x = nn.relu(x)
        x = nn.Conv(32, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(self.in_channels, kernel_size=(3, 3), padding='SAME')(x)
        x = nn.sigmoid(x)

        return x, mu, logvar

class SinusoidalPositionEmbeddings(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = jnp.exp(jnp.arange(half_dim) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = jnp.concatenate([jnp.sin(embeddings), jnp.cos(embeddings)], axis=-1)
        return embeddings

class UNet(nn.Module):
    in_channels: int
    out_channels: int
    time_dim: int = 256

    @nn.compact
    def __call__(self, x, t):
        # Time embedding
        time_mlp = SinusoidalPositionEmbeddings(self.time_dim)(t)
        time_mlp = nn.Dense(self.time_dim)(time_mlp)
        time_mlp = nn.gelu(time_mlp)
        time_mlp = nn.Dense(self.time_dim)(time_mlp)

        # U-Net architecture
        x1 = self.double_conv(64)(x)
        x2 = self.down(128)(x1)
        x3 = self.down(256)(x2)

        x3 = ResidualBlock(256, 256)(x3)
        x3 = ResidualBlock(256, 256)(x3)

        x = self.up(128)(x3)
        x = self.up(64)(x)
        x = nn.Conv(self.out_channels, kernel_size=(1, 1))(x)

        return x

    def double_conv(self, features):
        module = nn.Sequential([
            nn.Conv(features, kernel_size=(3, 3), padding='SAME'),
            nn.GroupNorm(32),
            nn.gelu,
            nn.Conv(features, kernel_size=(3, 3), padding='SAME'),
            nn.GroupNorm(32),
            nn.gelu
        ])
        return module

    def down(self, features):
        module = nn.Sequential([
            nn.Conv(features, kernel_size=(3, 3), strides=(1, 1), padding='SAME'),
            nn.GroupNorm(32),
            nn.gelu
        ])
        return module

    def up(self, features):
        module = nn.Sequential([
            nn.Conv(features, kernel_size=(3, 3), strides=(1, 1), padding='SAME'),
            nn.GroupNorm(32),
            nn.gelu
        ])
        return module

class LDM(nn.Module):
    latent_dim: int = 4
    time_steps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    def setup(self):
        self.vae = VAE(in_channels=3, latent_dim=self.latent_dim)
        self.unet = UNet(in_channels=self.latent_dim, out_channels=self.latent_dim)

        # Setup noise schedule
        self.betas = jnp.linspace(self.beta_start, self.beta_end, self.time_steps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas, axis=0)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = jax.random.normal(self.make_rng('sampling'), x_start.shape)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        if noise is None:
            noise = jax.random.normal(self.make_rng('sampling'), x_start.shape)

        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = self.unet(x_noisy, t)

        loss = jnp.mean((noise - predicted_noise) ** 2)

        return loss

    def p_sample(self, x, t, t_index):
        betas_t = self.betas[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recip_alphas_t = jnp.sqrt(1. / self.alphas[t]).reshape(-1, 1, 1, 1)

        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.unet(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.betas[t] * (1. - self.alphas_cumprod[t-1]) / (1. - self.alphas_cumprod[t])
            noise = jax.random.normal(self.make_rng('sampling'), x.shape)
            return model_mean + jnp.sqrt(posterior_variance_t) * noise

    def p_sample_loop(self, shape):
        b = shape[0]
        img = jax.random.normal(self.make_rng('sampling'), shape)

        for i in reversed(range(0, self.time_steps)):
            t = jnp.full((b,), i, dtype=jnp.int32)
            img = self.p_sample(img, t, i)

        return img

    def __call__(self, x, t):
        # Encode the input
        mu, _, _ = self.vae(x)

        # Add noise according to t
        z = self.q_sample(mu, t)

        # Pass through UNet
        noise_pred = self.unet(z, t)
        return noise_pred

    def train_step(self, x):
        t = jax.random.randint(self.make_rng('sampling'), (x.shape[0],), 0, self.time_steps)
        return self.p_losses(x, t)

    def sample(self, batch_size, img_size):
        return self.p_sample_loop((batch_size, self.latent_dim, img_size, img_size))

    def encode(self, x):
        return self.vae(x)[0]  # Return only mu

    def decode(self, z):
        b, h, w, c = z.shape
        return self.vae.decode(z.reshape((b, h * w * c)))
