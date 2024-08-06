import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import scipy

@jit
def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)"""
    mse = jnp.mean((img1 - img2) ** 2)
    return 20 * jnp.log10(1.0 / jnp.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM (Structural Similarity Index)"""
    img1 = np.array(img1.squeeze())
    img2 = np.array(img2.squeeze())
    return ssim(img1, img2, data_range=img2.max() - img2.min(), channel_axis=-1)

@jit
def calculate_mse(img1, img2):
    """Calculate MSE (Mean Squared Error)"""
    return jnp.mean((img1 - img2) ** 2)

@jit
def calculate_mae(img1, img2):
    """Calculate MAE (Mean Absolute Error)"""
    return jnp.mean(jnp.abs(img1 - img2))

def calculate_lpips(img1, img2, lpips_fn):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)"""
    # Note: This function might need to be implemented differently depending on the LPIPS library used
    return lpips_fn(img1, img2).item()

def calculate_fid(real_features, fake_features):
    """Calculate FID (Fréchet Inception Distance)"""
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

@jit
def calculate_edge_psnr(img1, img2):
    """Calculate PSNR on image edges"""
    edge_kernel = jnp.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]], dtype=jnp.float32).reshape(1, 1, 3, 3)
    
    def conv2d(img, kernel):
        return jax.lax.conv(img, kernel, window_strides=(1, 1), padding='SAME')
    
    img1_edges = conv2d(img1, edge_kernel)
    img2_edges = conv2d(img2, edge_kernel)
    
    return calculate_psnr(img1_edges, img2_edges)

def calculate_metrics(sr_image, hr_image, lpips_fn=None):
    """Calculate all metrics"""
    metrics = {}
    metrics['PSNR'] = calculate_psnr(sr_image, hr_image).item()
    metrics['SSIM'] = calculate_ssim(sr_image, hr_image)
    metrics['MSE'] = calculate_mse(sr_image, hr_image).item()
    metrics['MAE'] = calculate_mae(sr_image, hr_image).item()
    metrics['Edge_PSNR'] = calculate_edge_psnr(sr_image, hr_image).item()
    
    if lpips_fn is not None:
        metrics['LPIPS'] = calculate_lpips(sr_image, hr_image, lpips_fn)
    
    return metrics
