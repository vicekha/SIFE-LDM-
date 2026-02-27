#!/usr/bin/env python3
"""
SIFE-Vision: GPU-Optimized CIFAR-100 Training Loop
====================================================

GPU-accelerated version of train_vision.py that accepts CLI arguments
for batch_size, embed_dim, base_features, and max_steps — auto-scaled
by the Colab GPU detection cell.

Usage:
    python scripts/train_vision_gpu.py \
        --batch_size 64 --embed_dim 256 --base_features 128 --max_steps 100000
"""

import os
import argparse
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Suppress JAX complex casting warnings during backprop
try:
    warnings.filterwarnings('ignore', category=np.ComplexWarning)
except AttributeError:
    pass  # NumPy >= 2.0 removed ComplexWarning from the main namespace

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
                        help='Training batch size (auto-scaled by GPU detection)')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Model embedding dimension (auto-scaled by GPU detection)')
    parser.add_argument('--base_features', type=int, default=64,
                        help='Base features for multiscale config (auto-scaled by GPU detection)')
    parser.add_argument('--max_steps', type=int, default=50000,
                        help='Maximum training steps (auto-scaled by GPU detection)')
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
    for d in devices:
        if d.platform == 'gpu':
            print(f"🖥️  Training on GPU: {d}")
            try:
                stats = d.memory_stats()
                vram_gb = stats.get('bytes_limit', 0) / (1024**3)
                print(f"💾  VRAM available: {vram_gb:.1f} GB")
            except Exception:
                pass

    # 1. Configuration — GPU-scaled parameters
    sife_config = SIFEConfig(
        k=1.0,
        dt=0.005,
        gamma=0.1,
        eta_damping=0.05
    )

    ms_config = create_multiscale_config(
        num_levels=3,
        base_features=args.base_features
    )

    diff_config = DiffusionConfig(num_timesteps=1000)

    config = SIFELDMConfig(
        sife=sife_config,
        diffusion=diff_config,
        multiscale=ms_config,
        is_image=True,
        image_size=(32, 32),
        embed_dim=args.embed_dim,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        num_classes=100  # CIFAR-100 classes for CFG conditioning
    )

    print(f"\n📋  GPU-Optimized Config:")
    print(f"    batch_size    = {config.batch_size}")
    print(f"    embed_dim     = {config.embed_dim}")
    print(f"    base_features = {args.base_features}")
    print(f"    max_steps     = {config.max_steps}")
    print(f"    learning_rate = {config.learning_rate}")

    print(f"\nInitializing SIFE-Vision model (is_image={config.is_image})...")
    key = jax.random.PRNGKey(42)
    model, state, diffusion = create_train_state(config, key)
    from sife.model import create_optimizer
    optimizer = create_optimizer(config)

    # 2. Initialize ImageEncoder and LabelEncoder
    key, subkey = jax.random.split(key)
    img_enc = ImageEncoder(features=config.embed_dim)
    lbl_enc = LabelEncoder(num_classes=100, features=config.embed_dim)

    # Initialize encoder params with dummy inputs
    dummy_img = jnp.zeros((1, 32, 32, 3), dtype=jnp.float32)
    dummy_lbl = jnp.zeros((1,), dtype=jnp.int32)
    img_enc_params = img_enc.init(subkey, dummy_img)
    key, subkey = jax.random.split(key)
    lbl_enc_params = lbl_enc.init(subkey, dummy_lbl)

    # 3. Load Data
    print("Loading CIFAR-100 dataset...")
    images, labels = load_cifar_data()
    num_samples = len(images)
    print(f"Loaded {num_samples} samples")

    # 4. Training Loop with CFG class conditioning
    print(f"\n🚀  Starting GPU-accelerated training for {config.max_steps} steps...")
    print(f"    Logging every {args.log_every} steps, checkpointing every {args.checkpoint_every} steps\n")

    best_loss = float('inf')

    for step in range(config.max_steps):
        # Sample batch
        idx = np.random.randint(0, num_samples, config.batch_size)
        batch_images = jnp.array(images[idx], dtype=jnp.float32)
        batch_labels = jnp.array(labels[idx], dtype=jnp.int32)

        # Project RGB images → complex latent field via ImageEncoder
        complex_x = img_enc.apply(img_enc_params, batch_images)

        # Project labels → complex context embeddings via LabelEncoder
        # 15% CFG dropout: randomly null the context so model learns unconditional too
        key, subkey = jax.random.split(key)
        use_context = jax.random.uniform(subkey, (config.batch_size,)) > 0.15
        label_context = lbl_enc.apply(lbl_enc_params, batch_labels)
        label_context = label_context[:, jnp.newaxis, :]
        cfg_mask = use_context[:, jnp.newaxis, jnp.newaxis].astype(jnp.complex64)
        label_context = label_context * cfg_mask

        batch = {
            'complex_embedding': complex_x,
            'context': label_context,
            'labels': batch_labels
        }

        state, metrics = train_step(model, state, batch, diffusion, optimizer)
        loss = float(metrics['loss'])

        if step % args.log_every == 0:
            marker = " ⭐" if loss < best_loss else ""
            print(f"Step {step:>6d}/{config.max_steps}: Loss = {loss:.4f}{marker}", flush=True)
            if loss < best_loss:
                best_loss = loss

        if step % args.checkpoint_every == 0 and step > 0:
            save_checkpoint(state, args.checkpoint_dir, step)
            print(f"    💾  Checkpoint saved at step {step}")

    print(f"\n✅  Training Complete! Best loss: {best_loss:.4f}")
    save_checkpoint(state, args.checkpoint_dir, config.max_steps, name="final_vision_model")
    print(f"    💾  Final model saved to {args.checkpoint_dir}/final_vision_model.pkl")


if __name__ == "__main__":
    main()
