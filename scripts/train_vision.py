#!/usr/bin/env python3
"""
SIFE-Vision: CIFAR-100 Training Loop
====================================

Trains the SIFE-LDM on the CIFAR-100 dataset using 2D hierarchical fields.
"""

import os
import warnings
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
import optax

# Suppress JAX complex casting warnings during backprop
try:
    warnings.filterwarnings('ignore', category=np.ComplexWarning)
except AttributeError:
    pass # NumPy >= 2.0 removed ComplexWarning from the main namespace
from sife.model import (
    SIFELDMConfig, SIFELDM, create_train_state, train_step,
    save_checkpoint, ImageEncoder
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
        max_steps=50000
    )
    
    print(f"Initializing SIFE-Vision model (is_image={config.is_image})...")
    key = jax.random.PRNGKey(42)
    model, state, diffusion = create_train_state(config, key)
    from sife.model import create_optimizer
    optimizer = create_optimizer(config)
    
    # 2. Load Data
    print("Loading CIFAR-100 dataset...")
    images, labels = load_cifar_data()
    num_samples = len(images)
    
    # 3. Training Loop
    print(f"Starting training for {config.max_steps} steps...")
    
    for step in range(config.max_steps):
        # Sample batch
        idx = np.random.randint(0, num_samples, config.batch_size)
        batch_images = jnp.array(images[idx])
        
        # 4. Integrate ImageEncoder directly into the batch dict
        # The model's get_loss expects 'complex_embedding'
        # We'll use the ImageEncoder to project RGB -> Latent Complex
        
        # Manual projection if params not yet loaded/properly nested
        # R -> Amplitude, G/B -> Phase components (simplified but physically grounded)
        amp = batch_images[..., 0] 
        phase = jnp.arctan2(batch_images[..., 1] - 0.5, batch_images[..., 2] - 0.5)
        
        # Expand across feature dimension
        amp = jnp.repeat(amp[..., jnp.newaxis], config.embed_dim, axis=-1)
        phase = jnp.repeat(phase[..., jnp.newaxis], config.embed_dim, axis=-1)
        complex_x = amp * jnp.exp(1j * phase)
        
        batch = {
            'complex_embedding': complex_x,
            'labels': labels[idx]
        }
        
        state, metrics = train_step(model, state, batch, diffusion, optimizer)
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {metrics['loss']:.4f}")
            
        if step % 5000 == 0 and step > 0:
            save_checkpoint(state, './checkpoints/vision', step)
            
    print("Training Complete!")
    save_checkpoint(state, './checkpoints/vision', config.max_steps, name="final_vision_model")

if __name__ == "__main__":
    main()
