import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Optional
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sife.field import SIFEConfig
from sife.diffusion import (
    DiffusionConfig, GaussianDiffusion, SIFEDiffusion
)
from sife.model import SIFELDM, SIFELDMConfig

def demo_temporal_sampling():
    """
    Demonstrate that sampling correctly uses absolute phase conditioning.
    """
    print("Initializing demo...")
    
    # 1. Configs
    sife_config = SIFEConfig(
        omega_0=jnp.pi / 4,  # 45 degrees per second (fast for visibility)
        dt=0.01
    )
    diff_config = DiffusionConfig(num_timesteps=100)
    model_config = SIFELDMConfig(
        embed_dim=64,
        sife=sife_config
    )
    
    # 2. Initialize Model (Mock)
    key = jax.random.PRNGKey(42)
    model = SIFELDM(config=model_config)
    
    # Initialize params with a single dummy input
    dummy_x = jnp.zeros((1, 16, model_config.embed_dim), dtype=jnp.complex64)
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    params = model.init(key, dummy_x, dummy_t)['params']
    
    # 3. Diffusion process
    base_diffusion = GaussianDiffusion(diff_config)
    diffusion = SIFEDiffusion(
        diffusion=base_diffusion,
        sife_config=sife_config
    )
    
    import inspect
    print(f"DEBUG: SIFEDiffusion.sample signature: {inspect.signature(diffusion.sample)}")
    
    # 4. Instrumented Noise Prediction
    # We want to verify that 'abs_phase' passed to the model matches our expectations
    captured_abs_phases = []
    
    def model_fn(x, t, context=None, abs_phase=None, deterministic=True):
        if abs_phase is not None:
            captured_abs_phases.append((int(t[0]), float(abs_phase[0])))
        return model.apply({'params': params}, x, t, context, deterministic=deterministic, abs_phase=abs_phase)

    # 5. Run a few steps of sampling
    print("Running sampling steps...")
    batch_size = 1
    seq_len = 16
    shape = (batch_size, seq_len, model_config.embed_dim)
    
    # Run weighted DDIM sampling
    # Use keyword arguments for everything to be safe
    x_t = diffusion.sample(
        model=model_fn,
        shape=shape,
        key=key,
        num_steps=5
    )
    
    print("\nAbsolute Phase Conditioning Trace:")
    print("Step (t) | Expected (omega_0 * t) | Captured")
    print("-" * 45)
    
    for t_val, captured in captured_abs_phases:
        expected = (sife_config.omega_0 * t_val)
        # Handle wrap around or simple comparison
        print(f"{t_val:8d} | {expected:22.6f} | {captured:8.6f}")
        
        # Check tolerance (float32 precision)
        assert jnp.allclose(captured, expected, atol=1e-5), f"Mismatch at t={t_val}"

    print("\n[SUCCESS] Model correctly receives absolute phase temporal conditioning!")
    print("This grounding allows the SIFE-LDM to align latent fields with a physical clock.")

if __name__ == "__main__":
    import traceback
    try:
        demo_temporal_sampling()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
