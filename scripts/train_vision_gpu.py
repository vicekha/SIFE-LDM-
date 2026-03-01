#!/usr/bin/env python3
"""
SIFE-Vision: GPU-Optimized CIFAR-100 Training Loop
====================================================

GPU-accelerated version of train_vision.py that accepts CLI arguments
for batch_size, embed_dim, base_features, and max_steps — auto-scaled
by the Colab GPU detection cell.
"""

import os
import argparse
import warnings
import traceback
import jax
import jax.numpy as jnp
import numpy as np
import optax

import sife

# Suppress JAX complex casting warnings during backprop
try:
    warnings.filterwarnings('ignore', category=np.ComplexWarning)
except AttributeError:
    pass

from sife.model import (
    SIFELDMConfig, SIFELDM, create_train_state, train_step,
    save_checkpoint, ImageEncoder, LabelEncoder
)
from sife.field import SIFEConfig
from sife.multiscale import create_multiscale_config
from sife.diffusion import DiffusionConfig


def parse_args():
    parser = argparse.ArgumentParser(description='SIFE-Vision GPU Training')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Model embedding dimension')
    parser.add_argument('--base_features', type=int, default=64,
                        help='Base features for multiscale config')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='Maximum training steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/vision',
                        help='Checkpoint save directory')
    parser.add_argument('--checkpoint_every', type=int, default=5000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--log_every', type=int, default=100,
                        help='Print loss every N steps')
    return parser.parse_args()


def load_cifar_data():
    """Load preprocessed CIFAR-100 arrays."""
    data_dir = './data/cifar100'
    if not os.path.exists(os.path.join(data_dir, 'train_images.npy')):
        raise FileNotFoundError("CIFAR-100 data not found. Run download_cifar.py first.")

    images = np.load(os.path.join(data_dir, 'train_images.npy'))
    labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    return images, labels


def main():
    args = parse_args()

    # Print GPU info
    devices = jax.devices()
    print(f"JAX devices: {devices}")

    # 1. Configuration
    sife_config = SIFEConfig(k=1.0, dt=0.005, gamma=0.1, eta_damping=0.05)
    ms_config = create_multiscale_config(num_levels=3, base_features=args.base_features)
    diff_config = DiffusionConfig(num_timesteps=1000)

    config = SIFELDMConfig(
        sife=sife_config, diffusion=diff_config, multiscale=ms_config,
        is_image=True, image_size=(32, 32),
        embed_dim=args.embed_dim, batch_size=args.batch_size,
        learning_rate=args.learning_rate, max_steps=args.max_steps,
        num_classes=100
    )

    print(f"\nGPU-Optimized Config:")
    print(f"    batch_size    = {config.batch_size}")
    print(f"    embed_dim     = {config.embed_dim}")
    print(f"    max_steps     = {config.max_steps}")

    print(f"\nInitializing SIFE-Vision model...")
    key = jax.random.PRNGKey(42)
    model, state, diffusion = create_train_state(config, key)
    from sife.model import create_optimizer
    optimizer = create_optimizer(config)

    # 3. Load Data
    print("Loading CIFAR-100 dataset...")
    images, labels = load_cifar_data()
    num_samples = len(images)
    print(f"Loaded {num_samples} samples")

    # 4. Training Loop
    print(f"\nStarting training for {config.max_steps} steps...")
    best_loss = float('inf')

    try:
        for step in range(config.max_steps):
            # Sample batch
            idx = np.random.randint(0, num_samples, config.batch_size)
            batch_images_raw = images[idx]
            batch_labels = jnp.array(labels[idx], dtype=jnp.int32)
            
            # Data Augmentation: Random Horizontal Flip
            key, subkey = jax.random.split(key)
            do_flip = jax.random.uniform(subkey, (config.batch_size,)) > 0.5
            batch_images = jnp.array(batch_images_raw, dtype=jnp.float32)
            flipped_images = batch_images[:, :, ::-1, :]
            batch_images = jnp.where(do_flip[:, jnp.newaxis, jnp.newaxis, jnp.newaxis], flipped_images, batch_images)

            # CFG Dropout
            key, subkey = jax.random.split(key)
            use_context = jax.random.uniform(subkey, (config.batch_size,)) > 0.15
            
            batch = {
                'images': batch_images,
                'labels': batch_labels,
                'use_context_mask': use_context
            }

            # Train step
            state, metrics = train_step(model, state, batch, diffusion, optimizer)
            loss = float(metrics['loss'])

            if step % args.log_every == 0:
                marker = " *" if loss < best_loss else ""
                print(f"Step {step:6d}/{config.max_steps}: Loss = {loss:.4f}{marker}", flush=True)
                if loss < best_loss:
                    best_loss = loss

            if step > 0 and step % args.checkpoint_every == 0:
                save_checkpoint(state, args.checkpoint_dir, step)
                print(f"    Checkpoint saved at step {step}")

    except Exception as e:
        with open("error.log", "w") as f:
            f.write(str(e) + "\n")
            f.write(traceback.format_exc())
        print(f"CRITICAL ERROR: {e}")
        print("Detail written to error.log")
        raise e

    print(f"\nTraining Complete! Best loss: {best_loss:.4f}")
    save_checkpoint(state, args.checkpoint_dir, config.max_steps, name="final_vision_model")
    print(f"    Final model saved to {args.checkpoint_dir}/final_vision_model.pkl")


if __name__ == "__main__":
    main()
