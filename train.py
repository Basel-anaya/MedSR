import jax
import jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax
from flax import linen as nn
from flax.training import train_state
import numpy as np
from tqdm import tqdm
import argparse
import os
import pickle
import matplotlib.pyplot as plt

# Import your models (now implemented in Flax)
from models.ESRGAN import RealESRGANGenerator
from models.LDM import LDM
from models.MAXIM import MAXIM
from models.SwinIR import SwinIR

# Import your custom loss and metrics (reimplemented in JAX)
from scripts.custom_loss import custom_sr_loss
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
    parser.add_argument('--save_interval', type=int, default=10, help='Save model every n epochs')
    parser.add_argument('--log_interval', type=int, default=100, help='Log training progress every n steps')
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

def create_train_state(rng, model, learning_rate, input_shape):
    params = model.init(rng, jnp.ones(input_shape), train=True)['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx), custom_sr_loss()

@jit
def train_step(state, loss_fn, batch, rng):
    def loss_fn_wrapped(params):
        outputs = state.apply_fn({'params': params}, batch['lr'], train=True, rngs={'dropout': rng})
        loss = loss_fn(outputs, batch['hr'])
        return loss, outputs

    grad_fn = value_and_grad(loss_fn_wrapped, has_aux=True)
    (loss, outputs), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, outputs

@jit
def eval_step(state, loss_fn, batch):
    outputs = state.apply_fn({'params': state.params}, batch['lr'], train=False)
    loss = loss_fn(outputs, batch['hr'])
    metrics = calculate_metrics(outputs, batch['hr'])
    return loss, metrics

def train_epoch(state, loss_fn, train_loader, rng):
    epoch_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        rng, step_rng = random.split(rng)
        state, loss, _ = train_step(state, loss_fn, batch, step_rng)
        epoch_loss += loss
    return state, epoch_loss / len(train_loader)

def eval_model(state, loss_fn, val_loader):
    val_loss = 0
    val_metrics = {
        'PSNR': 0,
        'SSIM': 0,
        'MSE': 0,
        'MAE': 0,
        'Edge_PSNR': 0
    }
    for batch in tqdm(val_loader, desc="Evaluating"):
        loss, metrics = eval_step(state, loss_fn, batch)
        val_loss += loss
        for k, v in metrics.items():
            val_metrics[k] += v
    
    val_loss /= len(val_loader)
    for k in val_metrics:
        val_metrics[k] /= len(val_loader)
    return val_loss, val_metrics

def train(args):
    # Set up JAX random key
    rng = random.PRNGKey(0)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Load train and val image lists
    with open(os.path.join(args.data_dir, "train_images.pkl"), "rb") as f:
        train_images = pickle.load(f)
    with open(os.path.join(args.data_dir, "val_images.pkl"), "rb") as f:
        val_images = pickle.load(f)

    # Create datasets and data loaders
    train_dataset = TrainDataset(train_images)
    val_dataset = ValDataset(val_images)
    
    train_loader = list(train_dataset)  # Convert to list for JAX compatibility
    val_loader = list(val_dataset)
    
    # Create model
    model = get_model(args.model)

    # Initialize the training state
    rng, init_rng = random.split(rng)
    state, loss_fn = create_train_state(init_rng, model, args.lr, (args.batch_size, 64, 64, 3))  # Note the NHWC format

    # Training loop
    train_losses = []
    val_losses = []
    val_metrics_history = {
        'PSNR': [],
        'SSIM': [],
        'MSE': [],
        'MAE': [],
        'Edge_PSNR': []
    }
    
    for epoch in range(args.epochs):
        rng, epoch_rng = random.split(rng)
        state, train_loss = train_epoch(state, loss_fn, train_loader, epoch_rng)
        train_losses.append(train_loss)
        
        val_loss, val_metrics = eval_model(state, loss_fn, val_loader)
        val_losses.append(val_loss)

        # Store the metrics for this epoch
        for k, v in val_metrics.items():
            val_metrics_history[k].append(v)

        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        for k, v in val_metrics.items():
            print(f"{k}: {v:.4f}")

        # Save model
        if (epoch + 1) % args.save_interval == 0:
            model_path = os.path.join(args.output_dir, f"{args.model}_epoch_{epoch+1}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(state, f)
            print(f"Model saved to {model_path}")

    # Save final model
    final_model_path = os.path.join(args.output_dir, f"{args.model}_final.pkl")
    with open(final_model_path, 'wb') as f:
        pickle.dump(state, f)
    print(f"Final model saved to {final_model_path}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_plot.png'))
    plt.close()

    # Plot metrics
    plt.figure(figsize=(15, 10))
    for i, (metric, values) in enumerate(val_metrics_history.items(), 1):
        plt.subplot(2, 3, i)
        plt.plot(values)
        plt.title(metric)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'metrics_plot.png'))
    plt.close()

if __name__ == "__main__":
    args = parse_args()
    train(args)
