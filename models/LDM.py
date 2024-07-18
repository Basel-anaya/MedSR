import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x):
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super(VAE, self).__init__()
        
        # Encoder
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.encoder_linear = None
        self.fc_mu = None
        self.fc_logvar = None
        
        self.latent_dim = latent_dim
        
        # Decoder
        self.decoder_input = None
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        
    def encode(self, x):
        x = self.encoder_conv(x)
        b, c, h, w = x.shape
        x = x.view(b, c * h * w)
        
        if self.encoder_linear is None:
            self.encoder_linear = nn.Linear(x.shape[1], 256).to(x.device)
            self.fc_mu = nn.Linear(256, self.latent_dim * h * w).to(x.device)
            self.fc_logvar = nn.Linear(256, self.latent_dim * h * w).to(x.device)
        
        x = F.relu(self.encoder_linear(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar, (h, w)
    
    def decode(self, z, shape):
        h, w = shape
        z = z.view(-1, self.latent_dim, h, w)
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar, shape = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, shape), mu, logvar

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=256):
        super(UNet, self).__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        self.inc = self.double_conv(in_channels, 64)
        self.down1 = self.down(64, 128)
        self.down2 = self.down(128, 256)
        
        self.bot1 = ResidualBlock(256, 256)
        self.bot2 = ResidualBlock(256, 256)
        
        self.up1 = self.up(256, 128)
        self.up2 = self.up(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
    
    def down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
    
    def up(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
    
    def forward(self, x, t):
        t = self.time_mlp(t)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        
        x3 = self.bot1(x3)
        x3 = self.bot2(x3)
        
        x = self.up1(x3)
        x = self.up2(x)
        x = self.outc(x)
        return x

class LDM(nn.Module):
    def __init__(self, latent_dim=4, time_steps=1000):
        super(LDM, self).__init__()
        self.vae = VAE(in_channels=3, latent_dim=latent_dim)
        self.unet = UNet(in_channels=latent_dim, out_channels=latent_dim)
        self.time_steps = time_steps
        self.latent_dim = latent_dim
        
    def forward(self, x, t):
        # Encode the input
        mu, _, (h, w) = self.vae.encode(x)
        
        # Reshape the latent representation to match UNet input
        z = mu.view(-1, self.latent_dim, h, w)
        
        # Pass through UNet
        noise_pred = self.unet(z, t)
        return noise_pred