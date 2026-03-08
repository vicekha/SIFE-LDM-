#!/usr/bin/env python3
"""
SIFE-LDM Inference Script
=========================

Script for generating text/code using a trained SIFE-LDM model.

Usage:
    python inference.py --checkpoint checkpoints/best --prompt "def fibonacci(n):"
    python inference.py --checkpoint checkpoints/best --interactive

Author: SIFE-LDM Research Team
License: MIT
"""

import argparse
import json

print("\n--- SIFE-LDM INFERENCE v2.2 CL (BOS/In-painting Fix) ---")
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sife.model import SIFELDM, SIFELDMConfig, load_checkpoint
from sife.field import SIFEConfig
from sife.diffusion import DiffusionConfig, GaussianDiffusion, DDIMSampler, SIFEDiffusion
from sife.multiscale import create_multiscale_config
from sife.tokenizer import Vocabulary, SIFETokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate text with SIFE-LDM')
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (uses checkpoint config if not provided)')
    parser.add_argument('--vocab', type=str, default=None,
                        help='Path to vocabulary file')
    
    # Generation arguments
    parser.add_argument('--prompt', type=str, default=None,
                        help='Text prompt for generation')
    parser.add_argument('--max_length', type=int, default=256,
                        help='Maximum generation length')
    parser.add_argument('--num_steps', type=int, default=50,
                        help='Number of diffusion steps')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature')
    parser.add_argument('--eta', type=float, default=0.0,
                        help='DDIM eta parameter (0=deterministic, 1=DDPM)')
    
    # Guidance arguments
    parser.add_argument('--use_sife_guidance', action='store_true',
                        help='Use SIFE field guidance')
    parser.add_argument('--hamiltonian_scale', type=float, default=0.1,
                        help='Hamiltonian guidance scale')
    parser.add_argument('--truth_scale', type=float, default=0.1,
                        help='Truth potential guidance scale')
    
    # Mode arguments
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    parser.add_argument('--batch', type=str, default=None,
                        help='Path to file with prompts (one per line)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for generated text')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, config_path: Optional[str] = None):
    """Load a trained model from checkpoint."""
    # Load config
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), '..', 'config.json')
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}. Using defaults.")
        config_dict = {}
    else:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    
    # Create configuration with robust parsing (handles both flat and nested JSON)
    model_cfg = config_dict.get('model', config_dict)
    
    config = SIFELDMConfig(
        sife=SIFEConfig(**config_dict.get('sife', {})),
        diffusion=DiffusionConfig(**config_dict.get('diffusion', {})),
        multiscale=create_multiscale_config(),
        embed_dim=model_cfg.get('embed_dim', 256),
        num_heads=model_cfg.get('num_heads', 8),
        num_blocks=model_cfg.get('num_blocks', 4),
        max_seq_len=model_cfg.get('max_seq_len', 1024),
        vocab_size=model_cfg.get('vocab_size', 32000)
    )
    
    # Initialize a dummy state to define the structure for flax loading
    from sife.model import create_train_state
    temp_key = jax.random.PRNGKey(0)
    model, init_state, diffusion = create_train_state(config, temp_key)
    
    # Load checkpoint parameters into the state
    print(f"Restoring parameters from {checkpoint_path}...")
    state = load_checkpoint(checkpoint_path, init_state)
    
    return model, state, diffusion, config


def load_tokenizer(vocab_path: str, config: SIFELDMConfig):
    """Load tokenizer."""
    vocab = Vocabulary.load(vocab_path)
    
    key = jax.random.PRNGKey(0)
    tokenizer = SIFETokenizer(
        vocab=vocab,
        embed_dim=config.embed_dim,
        max_seq_len=config.max_seq_len,
        key=key
    )
    
    return tokenizer


def apply_attractor_relaxation(x: jnp.ndarray, config: SIFEConfig, eta: float = 0.2, num_steps: int = 5) -> Tuple[jnp.ndarray, bool]:
    """
    Apply SIFE attractor relaxation to the complex field.
    Enforces neighbor phase coherence (grammar physics).
    
    Returns:
        x: Relaxed field
        stable: Whether the field has fundamentally stabilized
    """
    from sife.field import is_field_stable, SIFField
    
    def phase_potential(theta):
        # We want to minimize the difference between adjacent phases
        diff = theta[:, 1:, :] - theta[:, :-1, :]
        return jnp.mean(1 - jnp.cos(diff))
    
    grad_fn = jax.grad(phase_potential)
    
    for _ in range(num_steps):
        amp = jnp.abs(x)
        theta = jnp.angle(x)
        
        # Descent along the phase potential gradient
        g = grad_fn(theta)
        theta = theta - eta * g
        
        x = amp * jnp.exp(1j * theta)
        
    # Check physical stability (AGI System 2 stopping criteria)
    # We must construct a dummy SIFField to pass to the stability checker
    field_for_check = SIFField(
        amplitude=jnp.abs(x),
        phase=jnp.angle(x),
        fluctuation=jnp.angle(x),
        velocity_amp=jnp.zeros_like(jnp.abs(x)),
        velocity_phi=jnp.zeros_like(jnp.angle(x))
    )
    
    # Check if the field energy is stable (threshold 0.1 for high coherence)
    is_stable = is_field_stable(field_for_check, config.sife, threshold=0.1)
    
    return x, bool(is_stable)


def generate_from_prompt(
    model: SIFELDM,
    params: dict,
    diffusion: GaussianDiffusion,
    tokenizer: SIFETokenizer,
    prompt: str,
    config: SIFELDMConfig,
    args
) -> str:
    """Generate text from a prompt."""
    key = jax.random.PRNGKey(42)
    
    # 1. Encode prompt to ensure BOS but NO EOS
    # tokenizer.encode adds BOS/EOS by default. Let's do it manually.
    inner_ids = tokenizer.vocab.encode(prompt)
    prompt_ids = [tokenizer.vocab.bos_id] + inner_ids
    prompt_len = len(prompt_ids)
    
    # Get token embeddings (unweighted by space)
    token_emb = tokenizer.token_embedding(jnp.array(prompt_ids)[jnp.newaxis, :])[0]
    # Add positional embeddings for these first N tokens
    pos_emb = tokenizer.positional_embedding(prompt_len)
    context = (token_emb * pos_emb)[jnp.newaxis, :, :]
    
    if prompt_len >= args.max_length:
        print(f"Warning: Prompt too long ({prompt_len} tokens).")
        prompt_len = args.max_length - 1
        context = context[:, :prompt_len, :]

    # 2. Setup Generation Field
    key, subkey = jax.random.split(key)
    noise_shape = (1, args.max_length, config.embed_dim)
    
    # Start from noise
    x = jax.random.normal(subkey, noise_shape, dtype=jnp.float32)
    x = x + 1j * jax.random.normal(jax.random.split(subkey)[0], noise_shape, dtype=jnp.float32)
    
    # 3. Sampling with In-painting
    ddim = DDIMSampler(diffusion)
    c = diffusion.num_timesteps // args.num_steps
    ddim_timesteps = jnp.arange(0, diffusion.num_timesteps, c)[::-1]
    
    def model_fn(x, t, ctx=None):
        return model.apply(params, x, t, context=ctx, deterministic=True)

    print(f"Generating ({args.max_length} tokens total, prompt is {prompt_len})...")
    
    # Phase 1: Standard Diffusion
    for i, t in enumerate(ddim_timesteps):
        t_prev = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else -1
        
        # Denoising step
        key, subkey = jax.random.split(key)
        x = ddim.ddim_step(model_fn, x, int(t), int(t_prev), subkey)
        
        # In-painting: Fix the prefix to match the current denoising noise level
        if t_prev >= 0:
            # We use t_prev because x now represents the state at the next (lower) noise level
            alpha_prev = diffusion.alphas_cumprod[int(t_prev)]
            key, subkey1 = jax.random.split(key)
            noise = jax.random.normal(subkey1, context.shape, dtype=jnp.float32)
            noise = noise + 1j * jax.random.normal(jax.random.split(subkey1)[0], context.shape, dtype=jnp.float32)
            
            # Repopulate the prefix with a noisy version of the prompt at the correct alpha level
            noisy_prompt = jnp.sqrt(alpha_prev) * context + jnp.sqrt(1 - alpha_prev) * noise
            x = x.at[:, :prompt_len, :].set(noisy_prompt)
        else:
            # At final t=0, set to clean context
            x = x.at[:, :prompt_len, :].set(context)
            
    # Phase 2: AGI Dynamic Deliberative Reasoning (System 2)
    # The model "thinks" internally by running basic physical relaxation until the
    # grammar field's Hamiltonian energy fundamentally stabilizes.
    print(f"Diffusion complete. Starting AGI System-2 Physical Relaxation...")
    max_thought_steps = 100
    thought_steps = 0
    is_stable = False
    
    while not is_stable and thought_steps < max_thought_steps:
        # Apply physical relaxation and evaluate stability
        x, is_stable = apply_attractor_relaxation(x, config, eta=0.1, num_steps=1)
        
        # Enforce prompt constraint so the model doesn't drift away from context
        x = x.at[:, :prompt_len, :].set(context)
        
        thought_steps += 1
        
    print(f"Thought Time: {thought_steps} internal physics cycles "
          f"{'(Stable)' if is_stable else '(Max steps reached)'}")
            
    return decode_embedding(x[0], tokenizer)


def decode_embedding(
    embedding: jnp.ndarray,
    tokenizer: SIFETokenizer
) -> str:
    """
    Decode complex embedding back to text using full complex distance.
    Factoring out positional embeddings first.
    """
    seq_len = embedding.shape[0]
    
    # 1. Get token embeddings components
    token_amp = tokenizer.token_embedding.amplitude_embeddings
    token_phase = tokenizer.token_embedding.phase_embeddings
    rotation = tokenizer.token_embedding.phase_rotation
    
    # Apply rotation to token phases to match forward logic
    token_phase_rot = token_phase @ rotation
    token_complex = token_amp * jnp.exp(1j * token_phase_rot)
    
    # 2. Get positional embeddings to factor them out
    pos_complex = tokenizer.positional_embedding(seq_len)
    
    # Factor out positional embedding: token_emb = embedding / pos_emb
    # (In complex space, we multiply by the conjugate and divide by amp^2)
    # Or just multiply by the reciprocal if amp is not zero
    # pos_complex has amp + 0.1, so it's safe
    token_latents = embedding / pos_complex
    
    decoded_ids = []
    print(f"DEBUG: Decoding {seq_len} tokens...")
    
    for i in range(seq_len):
        emb = token_latents[i]
        
        # Compute L2 distance in complex space to all vocab tokens
        diff = jnp.abs(emb[None, :] - token_complex)
        dist = jnp.sum(diff**2, axis=-1)
        
        best_token = jnp.argmin(dist)
        decoded_ids.append(int(best_token))
    
    # Decode tokens
    text = tokenizer.decode(jnp.array(decoded_ids), skip_special_tokens=True)
    return text


def interactive_mode(model, state, diffusion, tokenizer, config, args):
    """Run interactive generation mode."""
    print("\n" + "="*60)
    print("SIFE-LDM Interactive Mode")
    print("="*60)
    print("Enter prompts to generate text. Type 'quit' to exit.")
    print("="*60 + "\n")
    
    while True:
        try:
            prompt = input("Prompt> ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not prompt:
                continue
            
            # Generate
            output = generate_from_prompt(
                model, state.params, diffusion, tokenizer, prompt, config, args
            )
            
            print(f"\nGenerated:\n{output}\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(model, state, diffusion, tokenizer, config, args):
    """Process batch of prompts from file."""
    with open(args.batch, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    results = []
    for prompt in prompts:
        output = generate_from_prompt(
            model, state.params, diffusion, tokenizer, prompt, config, args
        )
        results.append({
            'prompt': prompt,
            'generated': output
        })
        print(f"\nPrompt: {prompt}")
        print(f"Generated: {output}\n")
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            for r in results:
                f.write(f"Prompt: {r['prompt']}\n")
                f.write(f"Generated: {r['generated']}\n")
                f.write("-" * 60 + "\n")
        print(f"Results saved to {args.output}")


def main():
    """Main inference function."""
    args = parse_args()
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, state, diffusion, config = load_model(args.checkpoint, args.config)
    
    # Load tokenizer
    if args.vocab is None:
        args.vocab = os.path.join(os.path.dirname(args.checkpoint), '..', 'vocab.json')
    
    print(f"Loading vocabulary from {args.vocab}...")
    tokenizer = load_tokenizer(args.vocab, config)
    
    # Run in appropriate mode
    if args.interactive:
        interactive_mode(model, state, diffusion, tokenizer, config, args)
    elif args.batch:
        batch_mode(model, state, diffusion, tokenizer, config, args)
    elif args.prompt:
        output = generate_from_prompt(
            model, state.params, diffusion, tokenizer, args.prompt, config, args
        )
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {output}")
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(f"Prompt: {args.prompt}\n")
                f.write(f"Generated: {output}\n")
            print(f"Result saved to {args.output}")
    else:
        print("Please provide --prompt, --interactive, or --batch")


if __name__ == '__main__':
    main()
