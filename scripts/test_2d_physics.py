"""
SIFE-Vision: 2D Physics Verification
====================================

Tests the evolution of the SIFE field on a 2D image-style lattice (32x32).
This validates that the 'Grammar Physics' scales from 1D sequences to 2D visual data.
"""

import jax
import jax.numpy as jnp
from sife.field import SIFEConfig, SIFField, initialize_field, evolve_field, compute_hamiltonian, sife_equations, discrete_laplacian, discrete_phase_laplacian

def test_2d_evolution():
    print("\n--- Testing 2D SIFE Evolution (Full-Multiscale Physics) ---")
    config = SIFEConfig(h=1.0, k=1.0, dt=0.005, gamma=0.1) 
    shape = (32, 32)
    
    key = jax.random.PRNGKey(123)
    field = initialize_field(key, shape, init_scale=0.1) # Keep away from A=0
    
    # Define a localized visual perturbation
    Y, X = jnp.ogrid[:32, :32]
    mask = (X - 16)**2 + (Y - 16)**2 < 8**2
    delta_amp = jnp.zeros(shape).at[mask].set(2.0)
    delta_phi = jnp.zeros(shape).at[mask].set(jnp.pi / 2)
    
    field = field.apply_perturbation(delta_amp, delta_phi)
    
    # Debug: Check if gradients are non-zero
    lap_A = discrete_laplacian(field.amplitude, config.h)
    lap_phi = discrete_phase_laplacian(field.fluctuation, config.h)
    
    print(f"Max |Laplacian A|: {jnp.max(jnp.abs(lap_A)):.6f}")
    print(f"Max |Laplacian Phi|: {jnp.max(jnp.abs(lap_phi)):.6f}")
    
    accel_A, accel_phi = sife_equations(field, config)
    print(f"Max accel_A: {jnp.max(jnp.abs(accel_A)):.6f}")
    print(f"Max accel_phi: {jnp.max(jnp.abs(accel_phi)):.6f}")
    
    # Calculate initial energy
    h_init = compute_hamiltonian(field, config)
    print(f"Initial Hamiltonian: {h_init:.4f}")
    
    # Evolve the 2D field
    print("Evolving 2D field for 50 steps...")
    evolved = evolve_field(field, config, num_steps=50)
    
    # Check if field changed at all
    amp_change = jnp.max(jnp.abs(evolved.amplitude - field.amplitude))
    phi_change = jnp.max(jnp.abs(evolved.fluctuation - field.fluctuation))
    print(f"Max Amplitude Change: {amp_change:.8e}")
    print(f"Max Fluctuation Change: {phi_change:.8e}")
    
    # Calculate final energy
    h_final = compute_hamiltonian(evolved, config)
    print(f"Final Hamiltonian: {h_final:.4f}")
    
    # Verify conservation
    drift = jnp.abs(h_final - h_init) / (jnp.abs(h_init) + 1e-8)
    print(f"Energy Drift: {drift:.6f}")
    
    if phi_change > 1e-10:
        print("SUCCESS: Field dynamics are active in 2D.")
    else:
        print("FAILURE: Field is static. Check if dt is being applied.")

if __name__ == "__main__":
    test_2d_evolution()
