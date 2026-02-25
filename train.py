#!/usr/bin/env python3
"""
SIFE-LDM Training Script
========================

Main training script for SIFE-LDM on TPU.

Usage:
    python train.py --config configs/base.yaml --data /path/to/data

For TPU training:
    python train.py --config configs/tpu_large.yaml --data /path/to/data --tpu

Author: SIFE-LDM Research Team
License: MIT
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sife.model import (
    SIFELDMConfig, SIFELDM, create_train_state, 
    train_tpu, generate, save_checkpoint, load_checkpoint
)
from sife.field import SIFEConfig
from sife.diffusion import DiffusionConfig
from sife.multiscale import create_multiscale_config
from sife.tokenizer import Vocabulary, SIFETokenizer, DataPipeline, create_training_data


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SIFE-LDM')
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                        help='Path to training data (text file or code directory)')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to validation data')
    parser.add_argument('--vocab', type=str, default=None,
                        help='Path to pre-built vocabulary')
    
    # Model arguments
    parser.add_argument('--config', type=str, default='configs/base.json',
                        help='Path to configuration file')
    parser.add_argument('--embed_dim', type=int, default=256,
                        help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_blocks', type=int, default=4,
                        help='Number of transformer blocks')
    parser.add_argument('--max_seq_len', type=int, default=2048,
                        help='Maximum sequence length')
    
    # MoE arguments
    parser.add_argument('--num_experts', type=int, default=0,
                        help='Number of experts for MoE')
    parser.add_argument('--num_experts_per_token', type=int, default=1,
                        help='Number of experts to route each token to')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup_steps', type=int, default=1000,
                        help='Number of warmup steps')
    parser.add_argument('--max_steps', type=int, default=100000,
                        help='Maximum training steps')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    
    # Diffusion arguments
    parser.add_argument('--num_timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='Starting beta value')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='Ending beta value')
    parser.add_argument('--noise_schedule', type=str, default='cosine',
                        choices=['linear', 'cosine', 'sqrt'],
                        help='Noise schedule type')
    
    # SIFE arguments
    parser.add_argument('--sife_mass', type=float, default=1.0,
                        help='SIFE mass parameter')
    parser.add_argument('--sife_coupling', type=float, default=1.0,
                        help='SIFE coupling strength')
    parser.add_argument('--sife_alpha', type=float, default=0.25,
                        help='SIFE quartic coefficient')
    parser.add_argument('--sife_beta', type=float, default=1.0,
                        help='SIFE quadratic coefficient')
    parser.add_argument('--sife_gamma', type=float, default=0.1,
                        help='SIFE truth potential strength')
    parser.add_argument('--phase_coupling_lambda', type=float, default=0.1,
                        help='Phase neighbor coupling loss weight')
    
    # Hardware arguments
    parser.add_argument('--tpu', action='store_true',
                        help='Use TPU')
    parser.add_argument('--tpu_address', type=str, default=None,
                        help='TPU address (for remote TPU)')
    parser.add_argument('--num_devices', type=int, default=None,
                        help='Number of devices (auto-detected if not specified)')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Log directory')
    parser.add_argument('--save_every', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--eval_every', type=int, default=500,
                        help='Evaluate every N steps')
    
    # Resume arguments
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def create_config_from_args(args) -> SIFELDMConfig:
    """Create SIFELDMConfig from command line arguments."""
    sife_config = SIFEConfig(
        m=args.sife_mass,
        k=args.sife_coupling,
        alpha=args.sife_alpha,
        beta=args.sife_beta,
        gamma=args.sife_gamma
    )
    
    diffusion_config = DiffusionConfig(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        schedule=args.noise_schedule
    )
    
    multiscale_config = create_multiscale_config(
        num_levels=3,
        base_features=args.embed_dim
    )
    
    return SIFELDMConfig(
        sife=sife_config,
        diffusion=diffusion_config,
        multiscale=multiscale_config,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_blocks=args.num_blocks,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        max_seq_len=args.max_seq_len,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        phase_coupling_lambda=args.phase_coupling_lambda
    )


def setup_tpu(args):
    """Setup TPU for training."""
    if args.tpu:
        print("Setting up TPU...")
        
        if args.tpu_address:
            # Connect to remote TPU
            jax.distributed.initialize(
                coordinator_address=args.tpu_address,
                num_processes=args.num_devices,
                process_id=0
            )
        else:
            # Local TPU
            jax.distributed.initialize()
        
        devices = jax.devices()
        print(f"Available devices: {devices}")
        print(f"Number of devices: {len(devices)}")
        
        return len(devices)
    
    return 1


def load_data(args):
    """Load and prepare training data."""
    print(f"Loading data from {args.data}...")
    
    # Check if data is a file or directory
    if os.path.isfile(args.data):
        # Text file
        with open(args.data, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    elif os.path.isdir(args.data):
        # Code directory
        texts = []
        import glob
        for ext in ['*.py', '*.js', '*.java', '*.cpp', '*.ts', '*.go', '*.rs']:
            for filepath in glob.glob(os.path.join(args.data, '**', ext), recursive=True):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        texts.append(f.read())
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}")
    else:
        raise ValueError(f"Data path {args.data} does not exist")
    
    print(f"Loaded {len(texts)} samples")
    
    # Build or load vocabulary
    if args.vocab and os.path.exists(args.vocab):
        print(f"Loading vocabulary from {args.vocab}")
        vocab = Vocabulary.load(args.vocab)
    else:
        print("Building vocabulary...")
        vocab = Vocabulary(min_freq=2, max_size=32000)
        vocab.build_from_texts(texts[:min(100000, len(texts))])
        print(f"Vocabulary size: {len(vocab)}")
        
        # Save vocabulary
        vocab_path = os.path.join(args.output_dir, 'vocab.json')
        vocab.save(vocab_path)
        print(f"Saved vocabulary to {vocab_path}")
    
    # Create tokenizer
    key = jax.random.PRNGKey(42)
    tokenizer = SIFETokenizer(
        vocab=vocab,
        embed_dim=args.embed_dim,
        max_seq_len=args.max_seq_len,
        key=key
    )
    
    # Create datasets
    print("Creating datasets...")
    train_texts = texts[:int(0.95 * len(texts))]
    val_texts = texts[int(0.95 * len(texts)):]
    
    key, subkey = jax.random.split(key)
    train_dataset = create_training_data(
        tokenizer, text_path=args.data if os.path.isfile(args.data) else None,
        code_dir=args.data if os.path.isdir(args.data) else None,
        batch_size=args.batch_size, key=subkey
    )[0]
    
    return tokenizer, train_dataset, val_texts


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Setup TPU
    num_devices = setup_tpu(args)
    
    # Create config
    config = create_config_from_args(args)
    
    # Save config
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'embed_dim': config.embed_dim,
            'num_heads': config.num_heads,
            'num_blocks': config.num_blocks,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'max_steps': config.max_steps,
            'max_seq_len': config.max_seq_len
        }, f, indent=2)
    
    # Load data
    tokenizer, train_dataset, val_texts = load_data(args)
    
    # Create model and training state
    print("Initializing model...")
    key = jax.random.PRNGKey(42)
    
    model, state, diffusion = create_train_state(config, key)
    if args.resume:
        print(f"Resuming from {args.resume}")
        state = load_checkpoint(args.resume, state)
    
    # Training loop
    print(f"\nStarting training for {args.num_epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of devices: {num_devices}")
    
    global_step = 0
    best_val_loss = float('inf')
    
    from sife.model import create_optimizer, train_step
    import optax
    
    optimizer = create_optimizer(config)
    
    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"{'='*50}")
        
        for batch in train_dataset:
            state, metrics = train_step(model, state, batch, diffusion, optimizer)
            epoch_loss += metrics['loss']
            num_batches += 1
            global_step += 1
            
            # Log progress
            if global_step % 100 == 0:
                elapsed = time.time() - epoch_start
                samples_per_sec = (num_batches * args.batch_size) / elapsed
                print(f"Step {global_step}: Loss = {metrics['loss']:.4f}, "
                      f"Samples/sec = {samples_per_sec:.2f}")
            
            # Save checkpoint
            if global_step % args.save_every == 0:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, f'checkpoint_{global_step}'
                )
                save_checkpoint(state, args.checkpoint_dir, global_step)
        
        # End of epoch
        avg_loss = epoch_loss / num_batches
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        
        # Validation
        if val_texts and (epoch + 1) % 1 == 0:
            val_loss = validate_model(model, state, tokenizer, val_texts, args)
            print(f"  Validation Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(state, args.checkpoint_dir, global_step, name='best')
                print(f"  New best model saved!")
    
    # Save final model
    save_checkpoint(state, args.checkpoint_dir, global_step, name='final')
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {final_path}")
    print("="*50)


def validate_model(model, state, tokenizer, val_texts, args):
    """Validate model on validation set."""
    total_loss = 0.0
    num_samples = 0
    
    from sife.diffusion import GaussianDiffusion, compute_loss
    diffusion = GaussianDiffusion(DiffusionConfig())
    
    # Process validation texts in batches
    batch_size = min(args.batch_size, len(val_texts))
    
    for i in range(0, len(val_texts), batch_size):
        batch_texts = val_texts[i:i + batch_size]
        batch = tokenizer.batch_encode(batch_texts)
        
        key = jax.random.PRNGKey(i)
        t = jax.random.randint(key, (len(batch_texts),), 0, diffusion.num_timesteps)
        
        loss = model.get_loss(state.params, batch, key, diffusion)
        total_loss += float(loss) * len(batch_texts)
        num_samples += len(batch_texts)
    
    return total_loss / num_samples


def generate_samples(model, state, tokenizer, diffusion, args):
    """Generate sample outputs."""
    print("\nGenerating samples...")
    
    key = jax.random.PRNGKey(0)
    shape = (1, args.max_seq_len, args.embed_dim)
    
    samples = generate(
        model, state.params, diffusion, key, shape,
        num_steps=50, use_sife_guidance=True
    )
    
    # Convert to text (simplified)
    print("Sample generated!")
    
    return samples


if __name__ == '__main__':
    main()
