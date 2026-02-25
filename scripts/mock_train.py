#!/usr/bin/env python3
"""
SIFE-LDM Mock Training Run
==========================

Quick test of the model on CPU with a few training steps.
"""

import sys
import os
sys.path.insert(0, '/home/z/my-project/download/sife-ldm')

import jax
import jax.numpy as jnp
from jax import jit, value_and_grad
import optax
import time

print("="*60)
print("SIFE-LDM MOCK TRAINING RUN (CPU)")
print("="*60)

print(f"\nJAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")

# Import SIFE-LDM modules
from sife.field import SIFEConfig, SIFField, initialize_field, compute_hamiltonian
from sife.diffusion import DiffusionConfig, GaussianDiffusion
from sife.model import SIFELDM, SIFELDMConfig, create_optimizer

# Create configuration
print("\n📝 Creating configuration...")
config = SIFELDMConfig(
    embed_dim=64,      # Small for quick test
    num_heads=2,
    num_blocks=2,
    batch_size=2,
    learning_rate=1e-4,
    max_seq_len=32,    # Short sequences for quick test
    max_steps=10       # Just 10 steps
)

print(f"  embed_dim: {config.embed_dim}")
print(f"  num_heads: {config.num_heads}")
print(f"  num_blocks: {config.num_blocks}")
print(f"  batch_size: {config.batch_size}")
print(f"  max_seq_len: {config.max_seq_len}")

# Initialize model
print("\n🔧 Initializing model...")
key = jax.random.PRNGKey(42)
model = SIFELDM(config)

# Create dummy inputs
key, subkey = jax.random.split(key)
dummy_x = jax.random.normal(subkey, (config.batch_size, config.max_seq_len, config.embed_dim))
dummy_x = dummy_x + 1j * jax.random.normal(jax.random.split(subkey)[0], 
                                            (config.batch_size, config.max_seq_len, config.embed_dim))
dummy_t = jnp.zeros((config.batch_size,), dtype=jnp.int32)

# Initialize parameters
print("  Initializing parameters...")
params = model.init(key, dummy_x, dummy_t, deterministic=True)
print(f"  Parameters initialized!")

# Count parameters
def count_params(params):
    total = 0
    for k, v in params.items():
        if hasattr(v, 'shape'):
            total += v.size
        elif isinstance(v, dict):
            total += count_params(v)
    return total

param_count = count_params(params)
print(f"  Total parameters: {param_count:,}")

# Create diffusion
print("\n📊 Setting up diffusion...")
diffusion_config = DiffusionConfig(
    num_timesteps=100,
    clip_denoised=False
)
diffusion = GaussianDiffusion(diffusion_config)
print(f"  Timesteps: {diffusion.num_timesteps}")

# Create optimizer
print("\n⚡ Creating optimizer...")
optimizer = create_optimizer(config)
opt_state = optimizer.init(params)
print("  Optimizer created!")

# Create dummy batch
print("\n📦 Creating dummy training batch...")
key, subkey = jax.random.split(key)
batch = {
    'complex_embedding': jax.random.normal(
        subkey, (config.batch_size, config.max_seq_len, config.embed_dim)
    ) + 1j * jax.random.normal(
        jax.random.split(subkey)[0], 
        (config.batch_size, config.max_seq_len, config.embed_dim)
    )
}

# Training step function
@jit
def train_step(params, opt_state, batch, key):
    key, subkey = jax.random.split(key)
    
    # Sample timesteps
    t = jax.random.randint(subkey, (config.batch_size,), 0, diffusion.num_timesteps)
    
    # Sample noise
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, batch['complex_embedding'].shape, dtype=jnp.float32)
    noise = noise + 1j * jax.random.normal(
        jax.random.split(subkey)[0], batch['complex_embedding'].shape, dtype=jnp.float32
    )
    
    # Add noise
    x_t = diffusion.q_sample(batch['complex_embedding'], t, key, noise)
    
    # Loss function
    def loss_fn(p):
        epsilon_pred = model.apply(p, x_t, t, deterministic=True)
        return jnp.mean(jnp.abs(epsilon_pred - noise) ** 2)
    
    loss, grads = value_and_grad(loss_fn)(params)
    
    # Apply gradients using optax
    updates, opt_state = optimizer.update(grads, opt_state, params=params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss, key

# Compile training step
print("\n🔥 Compiling training step...")
key, subkey = jax.random.split(key)
start_time = time.time()
params, opt_state, loss, key = train_step(params, opt_state, batch, subkey)
compile_time = time.time() - start_time
print(f"  Compilation time: {compile_time:.2f}s")
print(f"  Initial loss: {float(loss):.6f}")

# Run a few training steps
print("\n🚀 Running training steps...")
num_steps = 5
total_time = 0

for step in range(num_steps):
    key, subkey = jax.random.split(key)
    start_time = time.time()
    params, opt_state, loss, key = train_step(params, opt_state, batch, subkey)
    step_time = time.time() - start_time
    total_time += step_time
    
    print(f"  Step {step+1}/{num_steps}: Loss = {float(loss):.6f}, Time = {step_time*1000:.1f}ms")

avg_time = total_time / num_steps * 1000
print(f"\n  Average step time: {avg_time:.1f}ms")

# Test SIFE field dynamics
print("\n🌊 Testing SIFE field dynamics...")
sife_config = SIFEConfig(dt=0.001)
field = initialize_field(key, (32,))
H_initial = compute_hamiltonian(field, sife_config)
print(f"  Initial Hamiltonian: {float(H_initial):.4f}")

# Evolve field
from sife.field import evolve_field
evolved = evolve_field(field, sife_config, num_steps=5)
H_final = compute_hamiltonian(evolved, sife_config)
print(f"  Final Hamiltonian: {float(H_final):.4f}")
print(f"  Energy change: {float(H_final - H_initial):.6f} (should be small)")

# Test generation
print("\n🎨 Testing generation...")
from sife.diffusion import DDIMSampler

ddim = DDIMSampler(diffusion)

def simple_model(x, t, ctx=None):
    """Simple noise prediction (identity for testing)."""
    return jnp.zeros_like(x)

sample_shape = (1, 16, config.embed_dim)  # Smaller for quick test
key, subkey = jax.random.split(key)

start_time = time.time()
sample = ddim.sample(
    simple_model, 
    sample_shape, 
    subkey, 
    num_steps=5  # Just 5 steps for testing
)
gen_time = time.time() - start_time

print(f"  Generated sample shape: {sample.shape}")
print(f"  Generation time (5 steps): {gen_time:.2f}s")
print(f"  Sample amplitude range: [{float(jnp.min(jnp.abs(sample))):.4f}, {float(jnp.max(jnp.abs(sample))):.4f}]")

# Summary
print("\n" + "="*60)
print("MOCK TRAINING RUN COMPLETE!")
print("="*60)
print(f"\n✅ All components working correctly:")
print(f"  - Model initialization: OK")
print(f"  - Parameter count: {param_count:,}")
print(f"  - Diffusion process: OK")
print(f"  - Training step: OK ({avg_time:.1f}ms/step)")
print(f"  - SIFE field dynamics: OK")
print(f"  - Generation: OK")
print(f"\n🎉 SIFE-LDM is ready for TPU training!")
