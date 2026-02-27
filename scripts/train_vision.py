#!/usr/bin/env python3
"""
SIFE-Vision: CIFAR-100 Training Loop
====================================

Trains the SIFE-LDM on the CIFAR-100 dataset using 2D hierarchical fields.
Supports class-conditional generation via Phase-Coherent CFG.
"""

import os
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Suppress JAX complex casting warnings during backprop
try:
    warnings.filterwarnings('ignore', category=np.ComplexWarning)
except AttributeError:
    pass # NumPy >= 2.0 removed ComplexWarning from the main namespace
from sife.model import (
    SIFELDMConfig, SIFELDM, create_train_state, train_step,
    save_checkpoint, ImageEncoder, LabelEncoder
)
from sife.field import SIFEConfig
from sife.multiscale import create_multiscale_config
from sife.diffusion import DiffusionConfig

def load_cifar_data():
    """Load preprocessed CIFAR-100 arrays."""
    data_dir = './data/cifar100'
    if not os.path.exists(os.path.join(data_dir, 'train_images.npy')):
        raise FileNotFoundError("CIFAR-100 data not found. Run download_cifar.py first.")
    
    images = np.load(os.path.join(data_dir, 'train_images.npy'))
    labels = np.load(os.path.join(data_dir, 'train_labels.npy'))
    return images, labels

def main():
    # 1. Configuration for 2D Vision
    sife_config = SIFEConfig(
        k=1.0, 
        dt=0.005, 
        gamma=0.1,
        eta_damping=0.05
    )
    
    ms_config = create_multiscale_config(
        num_levels=3,
        base_features=64
    )
    
    diff_config = DiffusionConfig(num_timesteps=1000)
    
    config = SIFELDMConfig(
        sife=sife_config,
        diffusion=diff_config,
        multiscale=ms_config,
        is_image=True,
        image_size=(32, 32),
        embed_dim=128, # Compact for CIFAR
        batch_size=8,  # Reduced from 32 to prevent OOM
        learning_rate=2e-4,
        max_steps=50000,
        num_classes=100  # CIFAR-100 classes for CFG conditioning
    )
    
    print(f"Initializing SIFE-Vision model (is_image={config.is_image})...")
    key = jax.random.PRNGKey(42)
    model, state, diffusion = create_train_state(config, key)
    from sife.model import create_optimizer
    optimizer = create_optimizer(config)
    
    # 2. Initialize ImageEncoder and LabelEncoder separately for projection
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
    
    # 4. Training Loop with CFG class conditioning
    print(f"Starting training for {config.max_steps} steps...")
    
    for step in range(config.max_steps):
        # Sample batch
        idx = np.random.randint(0, num_samples, config.batch_size)
        batch_images = jnp.array(images[idx], dtype=jnp.float32)
        batch_labels = jnp.array(labels[idx], dtype=jnp.int32)
        
        # Project RGB images → complex latent field via ImageEncoder
        # ImageEncoder: (B, H, W, 3) → (B, H, W, embed_dim) complex
        complex_x = img_enc.apply(img_enc_params, batch_images)
        
        # Project labels → complex context embeddings via LabelEncoder
        # 15% CFG dropout: randomly null the context so model learns unconditional too
        key, subkey = jax.random.split(key)
        use_context = jax.random.uniform(subkey, (config.batch_size,)) > 0.15
        label_context = lbl_enc.apply(lbl_enc_params, batch_labels)
        # Shape: (B, 1, embed_dim) for transformer cross-attention
        label_context = label_context[:, jnp.newaxis, :]
        # Zero out context for dropout samples
        cfg_mask = use_context[:, jnp.newaxis, jnp.newaxis].astype(jnp.complex64)
        label_context = label_context * cfg_mask
        
        batch = {
            'complex_embedding': complex_x,
            'context': label_context,
            'labels': batch_labels
        }
        
        state, metrics = train_step(model, state, batch, diffusion, optimizer)
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {metrics['loss']:.4f}", flush=True)
            
        if step % 5000 == 0 and step > 0:
            save_checkpoint(state, './checkpoints/vision', step)
            
    print("Training Complete!")
    save_checkpoint(state, './checkpoints/vision', config.max_steps, name="final_vision_model")

if __name__ == "__main__":
    main()

