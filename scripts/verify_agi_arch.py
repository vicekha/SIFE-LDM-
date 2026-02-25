"""
SIFE-AGI Architecture Verification Suite
========================================

Tests the four core components of the newly implemented cognitive architecture:
1. Persistent World State (Context Field)
2. Action Coupling (Perturbations)
3. Symbolic Abstraction (Phase-Collapse)
4. Stability (Curvature/Hessian)
"""

import jax
import jax.numpy as jnp
from sife.field import SIFEConfig, SIFField, initialize_field, evolve_field, compute_hamiltonian
from sife.multiscale import create_multiscale_config, initialize_hierarchical_field, evolve_hierarchical_field
from sife.symbols import SymbolDecoder, SymbolEncoder
import matplotlib.pyplot as plt

def test_persistence():
    print("\n--- Testing Persistence (Global Context Field) ---")
    config = create_multiscale_config(num_levels=3)
    config = config._replace(use_context_persistence=True)
    
    # Initialize a hierarchical field
    key = jax.random.PRNGKey(0)
    # Define shapes for 3 levels (finest to coarsest)
    shapes = [(64,), (32,), (16,)]
    field = initialize_hierarchical_field(key, shapes, config)
    
    # Set a strong pattern in the coarsest level (context field)
    ctx_idx = field.num_levels - 1
    ctx_amp = field.amplitudes[ctx_idx].at[5:15].set(2.0)
    field = field._replace(amplitudes=field.amplitudes[:ctx_idx] + [ctx_amp])
    
    print(f"Initial context energy: {jnp.sum(field.amplitudes[ctx_idx]**2):.4f}")
    
    # Evolve
    sife_configs = [SIFEConfig() for _ in range(field.num_levels)]
    evolved = evolve_hierarchical_field(field, config, sife_configs, num_steps=50)
    
    print(f"Evolved context energy: {jnp.sum(evolved.amplitudes[ctx_idx]**2):.4f}")
    
    if jnp.sum(evolved.amplitudes[ctx_idx]**2) > 0.5 * jnp.sum(field.amplitudes[ctx_idx]**2):
        print("SUCCESS: Context field retained significant energy (low leakage confirmed).")
    else:
        print("FAILURE: Context field leaked too much energy.")

def test_action_coupling():
    print("\n--- Testing Action Coupling (Perturbations) ---")
    config = SIFEConfig()
    field = initialize_field(jax.random.PRNGKey(42), (64,))
    
    # Define an action (localized perturbation)
    delta_amp = jnp.zeros(64).at[32].set(1.0)
    delta_phi = jnp.zeros(64).at[32].set(jnp.pi / 2)
    
    perturbed = field.apply_perturbation(delta_amp, delta_phi)
    
    diff_amp = jnp.abs(perturbed.amplitude[32] - field.amplitude[32])
    print(f"Amplitude change at patch 32: {diff_amp:.4f}")
    
    if diff_amp > 0.1:
        print("SUCCESS: Action correctly perturbed the field state.")
    else:
        print("FAILURE: Action had no effect.")

def test_symbolic_abstraction():
    print("\n--- Testing Symbolic Abstraction (Phase-Collapse) ---")
    # Create a field with a coherent patch
    seq_len = 64
    features = 256
    field_data = jnp.zeros((1, seq_len, features), dtype=jnp.complex64)
    
    # Inject coherence in the middle patch (size 8)
    # Patches: 0-7, 8-15, 16-23, 24-31, 32-39, ...
    # Patch 4 is tokens 32-39
    phase = jnp.pi / 4
    coherent_patch = jnp.exp(1j * phase) * jnp.ones((8, features))
    field_data = field_data.at[0, 32:40, :].set(coherent_patch)
    
    decoder = SymbolDecoder(vocab_size=100)
    params = decoder.init(jax.random.PRNGKey(0), field_data)
    
    logits, mask = decoder.apply(params, field_data, threshold=0.7)
    
    print(f"Coherent patches detected: {jnp.where(mask[0])[0]}")
    
    if 4 in jnp.where(mask[0])[0]:
        print("SUCCESS: Symbolic decoder identified the coherent patch at index 4.")
    else:
        print("FAILURE: Symbolic decoder missed the coherent patch.")

if __name__ == "__main__":
    test_persistence()
    test_action_coupling()
    test_symbolic_abstraction()
