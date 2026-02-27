#!/usr/bin/env python3
"""
SIFE-LDM Image Generation Script
==================================
Generates CIFAR-100 style images from a trained vision checkpoint.

Usage (V1 unconditional from Kaggle):
    python scripts/generate_images.py --checkpoint checkpoints/vision/final_vision_model.pkl --num_images 16

Usage (V2 class-conditional with CFG):
    python scripts/generate_images.py --checkpoint checkpoints/vision/final_vision_model.pkl --class_id 14 --guidance_scale 7.5 --num_images 16

CIFAR-100 class IDs (examples):
    0=apple, 14=caterpillar, 19=cow, 35=girl, 46=lamp, 55=otter, 72=rocket
"""

import os
import sys
import argparse
import pickle
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sife.model import SIFELDM, SIFELDMConfig, ImageDecoder, create_model
from sife.field import SIFEConfig
from sife.diffusion import DiffusionConfig, GaussianDiffusion, EulerMaruyamaSampler
from sife.multiscale import create_multiscale_config

# CIFAR-100 class names for display
CIFAR100_CLASSES = [
    'apple','aquarium_fish','baby','bear','beaver','bed','bee','beetle','bicycle','bottle',
    'bowl','boy','bridge','bus','butterfly','camel','can','castle','caterpillar','cattle',
    'chair','chimpanzee','clock','cloud','cockroach','couch','crab','crocodile','cup','dinosaur',
    'dolphin','elephant','flatfish','forest','fox','girl','hamster','house','kangaroo','keyboard',
    'lamp','lawn_mower','leopard','lion','lizard','lobster','man','maple_tree','motorcycle','mountain',
    'mouse','mushroom','oak_tree','orange','orchid','otter','palm_tree','pear','pickup_truck','pine_tree',
    'plain','plate','poppy','porcupine','possum','rabbit','raccoon','ray','road','rocket',
    'rose','sea','seal','shark','shrew','skunk','skyscraper','snail','snake','spider',
    'squirrel','streetcar','sunflower','sweet_pepper','table','tank','telephone','television','tiger','tractor',
    'train','trout','tulip','turtle','wardrobe','whale','willow_tree','wolf','woman','worm'
]


def parse_args():
    parser = argparse.ArgumentParser(description='Generate images from SIFE-LDM vision model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained checkpoint .pkl file')
    parser.add_argument('--output_dir', type=str, default='./generated_images',
                        help='Directory to save generated images')
    parser.add_argument('--num_images', type=int, default=16,
                        help='Number of images to generate')
    parser.add_argument('--num_steps', type=int, default=100,
                        help='Number of SDE sampling steps (fewer = faster but lower quality)')
    parser.add_argument('--class_id', type=int, default=None,
                        help='CIFAR-100 class ID for conditional generation (0-99). None = unconditional.')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='CFG guidance strength (only used with --class_id). Higher = more class-faithful.')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Model embedding dim (128 for V1/V2 CIFAR, 256 for larger models)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--attractor_stop', action='store_true',
                        help='Enable attractor-based early stopping (V2 only)')
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, init_state) -> dict:
    """Load model parameters from a flax serialization checkpoint."""
    from sife.model import load_checkpoint as sife_load_ckpt
    print(f"Loading checkpoint from {checkpoint_path}...")
    
    # Check if the path exists, if not see if it has a .pkl extension and try without
    if not os.path.exists(checkpoint_path) and checkpoint_path.endswith('.pkl'):
        alt_path = checkpoint_path[:-4]
        if os.path.exists(alt_path):
            checkpoint_path = alt_path
            
    try:
        # Load using sife.model.load_checkpoint
        state = sife_load_ckpt(checkpoint_path, init_state)
        return state.params
    except Exception as e:
        # Fallback to pickle just in case it is an older model format
        import pickle
        print(f"Flax loading failed, trying pickle fallback: {e}")
        with open(checkpoint_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and 'params' in data:
            return data['params']
        elif hasattr(data, 'params'):
            return data.params
        else:
            return data


def complex_field_to_rgb(field: jnp.ndarray) -> np.ndarray:
    """
    Decode a complex SIFE field (B, H, W, embed_dim) → RGB images (B, H, W, 3).
    
    Uses ImageDecoder for learned decoding, or falls back to physics-based projection:
    - Amplitude → brightness
    - Phase coherence → color hue
    """
    # Project amplitude + phase → RGB using physics-based mapping
    amp = jnp.abs(field)           # (B, H, W, embed_dim)
    phase = jnp.angle(field)       # (B, H, W, embed_dim)
    
    # Average across feature dim to get per-pixel amplitude and phase
    mean_amp = jnp.mean(amp, axis=-1, keepdims=True)         # (B, H, W, 1)
    mean_phase = jnp.mean(phase, axis=-1, keepdims=True)     # (B, H, W, 1)
    
    # Map phase to color channels using 3 phase offsets (RGB = 0°, 120°, 240°)
    r = mean_amp * (0.5 + 0.5 * jnp.cos(mean_phase))
    g = mean_amp * (0.5 + 0.5 * jnp.cos(mean_phase + 2 * jnp.pi / 3))
    b = mean_amp * (0.5 + 0.5 * jnp.cos(mean_phase + 4 * jnp.pi / 3))
    
    rgb = jnp.concatenate([r, g, b], axis=-1)               # (B, H, W, 3)
    rgb = jnp.clip(rgb, 0.0, 1.0)
    
    return np.array(rgb)


def save_image_grid(images: np.ndarray, output_path: str, class_name: str = None):
    """Save a grid of images as a PNG file."""
    try:
        from PIL import Image
    except ImportError:
        print("PIL not available. Saving as .npy instead.")
        np.save(output_path.replace('.png', '.npy'), images)
        print(f"Saved raw arrays to {output_path.replace('.png', '.npy')}")
        return
    
    B, H, W, C = images.shape
    cols = int(np.ceil(np.sqrt(B)))
    rows = int(np.ceil(B / cols))
    
    # Scale up for visibility (CIFAR is 32x32, scale to 128x128)
    scale = 4
    grid = Image.new('RGB', (cols * W * scale, rows * H * scale), color=(20, 20, 20))
    
    for i, img_arr in enumerate(images):
        row, col = divmod(i, cols)
        img_uint8 = (img_arr * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8).resize((W * scale, H * scale), Image.NEAREST)
        grid.paste(img, (col * W * scale, row * H * scale))
    
    grid.save(output_path)
    title = f"Class: {class_name}" if class_name else "Unconditional"
    print(f"Saved {B} images ({title}) → {output_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build config matching the training setup
    config = SIFELDMConfig(
        sife=SIFEConfig(),
        diffusion=DiffusionConfig(num_timesteps=1000),
        multiscale=create_multiscale_config(num_levels=3, base_features=64),
        is_image=True,
        image_size=(32, 32),
        embed_dim=args.embed_dim,
        batch_size=args.num_images,
    )
    
    print(f"Initializing model architecture...")
    key = jax.random.PRNGKey(args.seed)
    model, _ = create_model(config, key)
    
    # Load trained weights
    from sife.model import create_train_state
    _, init_state, _ = create_train_state(config, key)
    params = load_checkpoint(args.checkpoint, init_state)
    
    # Build diffusion + sampler
    diffusion = GaussianDiffusion(DiffusionConfig(num_timesteps=1000))
    sampler = EulerMaruyamaSampler(diffusion)
    
    # Build model function for sampling
    def model_fn(x, t, context=None):
        return model.apply(params, x, t, context=context, deterministic=True)
    
    # Build optional label context for CFG
    context = None
    class_name = None
    
    if args.class_id is not None:
        try:
            from sife.model import LabelEncoder
            class_name = CIFAR100_CLASSES[args.class_id]
            print(f"Generating class: [{args.class_id}] {class_name} with guidance scale {args.guidance_scale}")
            
            lbl_enc = LabelEncoder(num_classes=100, features=args.embed_dim)
            key, subkey = jax.random.split(key)
            labels = jnp.full((args.num_images,), args.class_id, dtype=jnp.int32)
            lbl_params = lbl_enc.init(subkey, labels)
            label_emb = lbl_enc.apply(lbl_params, labels)
            context = label_emb[:, jnp.newaxis, :]  # (B, 1, embed_dim)
        except ImportError:
            print("LabelEncoder not available (V1 checkpoint). Running unconditional.")
    else:
        print("Running unconditional generation...")
    
    print(f"Sampling {args.num_images} images with {args.num_steps} steps...")
    
    shape = (args.num_images, 32, 32, args.embed_dim)
    sife_cfg = config.sife if args.attractor_stop else None
    
    # Run sampling
    if context is not None and args.class_id is not None:
        field, steps_used = sampler.cfg_guided_sample(
            model_fn, shape, key, context=context,
            guidance_scale=args.guidance_scale,
            num_steps=args.num_steps,
            sife_config=sife_cfg
        )
    else:
        field, steps_used = sampler.sample(
            model_fn, shape, key, context=context,
            num_steps=args.num_steps,
            sife_config=sife_cfg
        )
    
    print(f"Sampling complete — used {steps_used}/{args.num_steps} steps" +
          (" (attractor early stop!)" if steps_used < args.num_steps else ""))
    
    # Decode complex field → RGB
    print("Decoding complex field → RGB images...")
    rgb_images = complex_field_to_rgb(field)
    
    # Save grid
    suffix = f"_class{args.class_id}_{class_name}" if args.class_id is not None else "_unconditional"
    output_path = os.path.join(args.output_dir, f"generated{suffix}.png")
    save_image_grid(rgb_images, output_path, class_name)
    
    # Also save raw numpy arrays for analysis
    raw_path = os.path.join(args.output_dir, f"field{suffix}.npy")
    np.save(raw_path, np.array(field))
    print(f"Saved raw complex field → {raw_path}")
    
    print("\nDone! Generated outputs:")
    print(f"  Images:      {output_path}")
    print(f"  Raw fields:  {raw_path}")


if __name__ == '__main__':
    main()
