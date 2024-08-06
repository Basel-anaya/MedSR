import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
import tensorflow as tf
import tensorflow_hub as hub

class CustomSRLoss(nn.Module):
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 0.001
    delta: float = 0.1

    def setup(self):
        # Load pre-trained VGG19 model
        vgg_model = hub.load('https://tfhub.dev/google/imagenet/vgg19/feature_vector/4')
        self.vgg_model = lambda x: vgg_model(x)
        
        self.edge_kernel = jnp.array([[-1, -1, -1],
                                      [-1,  8, -1],
                                      [-1, -1, -1]], dtype=jnp.float32).reshape(1, 3, 3, 1)

    @nn.compact
    def __call__(self, sr, hr):
        # Pixel-wise loss (L1 loss)
        pixel_loss = jnp.mean(jnp.abs(sr - hr))
        
        # Perceptual loss
        sr_features = self.vgg_model(tf.image.grayscale_to_rgb(tf.convert_to_tensor(sr)))
        hr_features = self.vgg_model(tf.image.grayscale_to_rgb(tf.convert_to_tensor(hr)))
        perceptual_loss = jnp.mean((sr_features - hr_features) ** 2)
        
        # Edge loss
        sr_edges = jax.lax.conv(sr, self.edge_kernel, window_strides=(1, 1), padding='SAME')
        hr_edges = jax.lax.conv(hr, self.edge_kernel, window_strides=(1, 1), padding='SAME')
        edge_loss = jnp.mean((sr_edges - hr_edges) ** 2)
        
        # Structural Similarity (SSIM) loss
        ssim_loss = 1 - ssim(sr, hr)
        
        # Combine losses
        total_loss = (self.alpha * pixel_loss +
                      self.beta * perceptual_loss +
                      self.gamma * edge_loss +
                      self.delta * ssim_loss)
        
        return total_loss, pixel_loss, perceptual_loss, edge_loss, ssim_loss

@partial(jax.jit, static_argnums=(2, 3))
def ssim(img1, img2, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    def avg_pool(x):
        return jax.lax.reduce_window(x, 0., jax.lax.add, (1, window_size, window_size, 1), (1, 1, 1, 1), 'SAME') / (window_size ** 2)

    mu1 = avg_pool(img1)
    mu2 = avg_pool(img2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = avg_pool(img1 ** 2) - mu1_sq
    sigma2_sq = avg_pool(img2 ** 2) - mu2_sq
    sigma12 = avg_pool(img1 * img2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return jnp.mean(ssim_map)
    else:
        return jnp.mean(ssim_map, axis=(1, 2, 3))

def custom_sr_loss():
    loss_module = CustomSRLoss()
    
    @jax.jit
    def loss_fn(sr, hr):
        return loss_module(sr, hr)[0]  # Return only the total loss
    
    return loss_fn
