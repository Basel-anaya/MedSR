import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CustomSRLoss(nn.Module):
    def __init__(self, vgg_model=None, alpha=1.0, beta=0.1, gamma=0.001, delta=0.1, chunk_size=4):
        super(CustomSRLoss, self).__init__()
        self.alpha = alpha  # Weight for pixel-wise loss
        self.beta = beta    # Weight for perceptual loss
        self.gamma = gamma  # Weight for edge loss
        self.delta = delta  # Weight for structural similarity loss
        self.chunk_size = chunk_size  # Number of images to process at once
        
        # VGG model for perceptual loss
        if vgg_model is None:
            vgg = models.vgg19(pretrained=True).features[:36].eval()
            for param in vgg.parameters():
                param.requires_grad = False
            self.vgg = vgg.cuda() if torch.cuda.is_available() else vgg
        else:
            self.vgg = vgg_model
        
        # Edge detection kernel
        self.edge_kernel = torch.tensor([[-1, -1, -1],
                                         [-1,  8, -1],
                                         [-1, -1, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        if torch.cuda.is_available():
            self.edge_kernel = self.edge_kernel.cuda()

    def forward(self, sr, hr):
        # Ensure inputs are in the correct format
        if sr.size(1) != 3:
            sr = sr.repeat(1, 3, 1, 1)
        if hr.size(1) != 3:
            hr = hr.repeat(1, 3, 1, 1)

        # Initialize loss components
        pixel_loss = 0
        perceptual_loss = 0
        edge_loss = 0
        ssim_loss = 0

        # Process images in chunks
        for i in range(0, sr.size(0), self.chunk_size):
            sr_chunk = sr[i:i+self.chunk_size]
            hr_chunk = hr[i:i+self.chunk_size]

            # Pixel-wise loss (L1 loss)
            pixel_loss += F.l1_loss(sr_chunk, hr_chunk)
            
            # Perceptual loss
            sr_features = self.vgg(sr_chunk)
            hr_features = self.vgg(hr_chunk)
            perceptual_loss += F.mse_loss(sr_features, hr_features)
            
            # Edge loss
            sr_edges = F.conv2d(sr_chunk[:, 0:1], self.edge_kernel, padding=1)
            hr_edges = F.conv2d(hr_chunk[:, 0:1], self.edge_kernel, padding=1)
            edge_loss += F.mse_loss(sr_edges, hr_edges)
            
            # Structural Similarity (SSIM) loss
            ssim_loss += 1 - ssim(sr_chunk[:, 0:1], hr_chunk[:, 0:1])

        # Average the losses
        batch_size = sr.size(0)
        pixel_loss /= batch_size
        perceptual_loss /= batch_size
        edge_loss /= batch_size
        ssim_loss /= batch_size
        
        # Combine losses
        total_loss = (self.alpha * pixel_loss +
                      self.beta * perceptual_loss +
                      self.gamma * edge_loss +
                      self.delta * ssim_loss)
        
        return total_loss, pixel_loss, perceptual_loss, edge_loss, ssim_loss

# SSIM function (unchanged)
def ssim(img1, img2, window_size=11, size_average=True):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
