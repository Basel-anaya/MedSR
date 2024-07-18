import scipy
import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)"""
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    """Calculate SSIM (Structural Similarity Index)"""
    img1 = img1.cpu().numpy().transpose(1, 2, 0)
    img2 = img2.cpu().numpy().transpose(1, 2, 0)
    return ssim(img1, img2, data_range=img2.max() - img2.min(), multichannel=True)

def calculate_mse(img1, img2):
    """Calculate MSE (Mean Squared Error)"""
    return F.mse_loss(img1, img2).item()

def calculate_mae(img1, img2):
    """Calculate MAE (Mean Absolute Error)"""
    return F.l1_loss(img1, img2).item()

def calculate_lpips(img1, img2, lpips_fn):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)"""
    return lpips_fn(img1, img2).item()

def calculate_fid(real_features, fake_features):
    """Calculate FID (Fréchet Inception Distance)"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_edge_psnr(img1, img2):
    """Calculate PSNR on image edges"""
    edge_kernel = torch.tensor([[-1, -1, -1],
                                [-1,  8, -1],
                                [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3).to(img1.device)
    
    img1_edges = F.conv2d(img1, edge_kernel, padding=1)
    img2_edges = F.conv2d(img2, edge_kernel, padding=1)
    
    return calculate_psnr(img1_edges, img2_edges)

def calculate_metrics(sr_image, hr_image, lpips_fn=None):
    """Calculate all metrics"""
    metrics = {}
    metrics['PSNR'] = calculate_psnr(sr_image, hr_image).item()
    metrics['SSIM'] = calculate_ssim(sr_image.squeeze(0), hr_image.squeeze(0))
    metrics['MSE'] = calculate_mse(sr_image, hr_image)
    metrics['MAE'] = calculate_mae(sr_image, hr_image)
    metrics['Edge_PSNR'] = calculate_edge_psnr(sr_image, hr_image).item()
    
    if lpips_fn is not None:
        metrics['LPIPS'] = calculate_lpips(sr_image, hr_image, lpips_fn)
    
    return metrics

# Usage example
if __name__ == "__main__":
    # Simulating super-resolved and high-res images
    sr_image = torch.rand(1, 1, 256, 256)
    hr_image = torch.rand(1, 1, 256, 256)
    
    # If you want to use LPIPS, you need to install and import it
    # import lpips
    # loss_fn_alex = lpips.LPIPS(net='alex')
    
    metrics = calculate_metrics(sr_image, hr_image)  # , lpips_fn=loss_fn_alex)
    
    for metric, value in metrics.items():
        print(f"{metric}: {value}")