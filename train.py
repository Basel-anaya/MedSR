import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import pickle
from PIL import Image

# Import your models (assuming these are in a 'models' directory)
from models.ESRGAN import RealESRGANGenerator
from models.LDM import LDM
from models.MAXIM import MAXIM
from models.SwinIR import SwinIR

# Import your custom loss and metrics
from scripts.custom_loss import CustomSRLoss
from scripts.sr_metrics import calculate_metrics
from scripts.prepare_data import TrainDataset, ValDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train Super Resolution Models on NIH Chest X-ray Dataset")
    parser.add_argument('--model', type=str, required=True, choices=['ESRGAN', 'LDM', 'MAXIM', 'SwinIR'],
                        help='Model to train')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to NIH Chest X-ray dataset')
    parser.add_argument('--output_dir', type=str, default='./output', help='Output directory for saved models')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda/cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every n epochs')
    return parser.parse_args()

def get_model(model_name):
    if model_name == 'ESRGAN':
        return RealESRGANGenerator()
    elif model_name == 'LDM':
        return LDM()
    elif model_name == 'MAXIM':
        return MAXIM()
    elif model_name == 'SwinIR':
        return SwinIR()
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train(args):
    # Set up device
    device = torch.device(args.device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Set up data transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load train and val image lists
    with open(os.path.join(args.data_dir, "train_images.pkl"), "rb") as f:
        train_images = pickle.load(f)
    with open(os.path.join(args.data_dir, "val_images.pkl"), "rb") as f:
        val_images = pickle.load(f)

    # Create datasets and data loaders
    train_dataset = TrainDataset(train_images)
    val_dataset = ValDataset(val_images)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Create model
    model = get_model(args.model).to(device)

    # Set up loss function and optimizer
    criterion = CustomSRLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        for i, (lr_images, hr_images) in enumerate(train_loader):
            lr_images, hr_images = lr_images.to(device), hr_images.to(device)

            optimizer.zero_grad()
            
            if args.model == 'LDM':
                # For LDM, we need to provide a time step
                batch_size = lr_images.shape[0]
                t = torch.randint(0, model.time_steps, (batch_size,), device=device).long()
                sr_images = model(lr_images, t)
            else:
                sr_images = model(lr_images)
            
            loss = criterion(sr_images, hr_images)
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")


        # Validation
        model.eval()
        val_metrics = {}
        with torch.no_grad():
            for lr_images, hr_images in val_loader:
                lr_images, hr_images = lr_images.to(device), hr_images.to(device)
                
                if args.model == 'LDM':
                    # For LDM, we need to provide a time step
                    batch_size = lr_images.shape[0]
                    t = torch.randint(0, model.time_steps, (batch_size,), device=device).long()
                    sr_images = model(lr_images, t)
                else:
                    sr_images = model(lr_images)
                
                metrics = calculate_metrics(sr_images, hr_images)
                for key, value in metrics.items():
                    val_metrics[key] = val_metrics.get(key, 0) + value

        # Average the metrics
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        print(f"Epoch [{epoch+1}/{args.epochs}] Validation Metrics:")
        for key, value in val_metrics.items():
            print(f"{key}: {value:.4f}")

        # Save model
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}_epoch_{epoch+1}.pth"))

    # Save final model
    torch.save(model.state_dict(), os.path.join(args.output_dir, f"{args.model}_final.pth"))

if __name__ == "__main__":
    args = parse_args()
    train(args)