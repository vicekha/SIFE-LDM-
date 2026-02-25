import jax
import jax.numpy as jnp
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sife.field import (
    SIFEConfig, SIFField, initialize_field,
    compute_hamiltonian, leapfrog_step, evolve_field
)

def test_background_rotation():
    """Verify that total phase theta integrates omega_0 correctly."""
    print("Testing background rotation...")
    omega_0 = 1.0 # 1 rad/s
    dt = 0.01
    num_steps = 100
    config = SIFEConfig(omega_0=omega_0, dt=dt)
    
    key = jax.random.PRNGKey(42)
    shape = (10,)
    field = SIFField(
        amplitude=jnp.ones(shape),
        phase=jnp.zeros(shape),
        fluctuation=jnp.zeros(shape),
        velocity_amp=jnp.zeros(shape),
        velocity_phi=jnp.zeros(shape)
    )
    
    evolved = evolve_field(field, config, num_steps)
    
    total_time = num_steps * dt
    expected_delta = (omega_0 * total_time) % (2 * jnp.pi)
    actual_delta = (evolved.phase - evolved.fluctuation) % (2 * jnp.pi)
    mean_actual_delta = jnp.mean(actual_delta)
    
    diff = jnp.abs(mean_actual_delta - expected_delta)
    diff = jnp.minimum(diff, 2 * jnp.pi - diff)
    
    print(f"  Final Mean(theta - phi): {mean_actual_delta:.6f}")
    print(f"  Expected omega_0 * t:   {expected_delta:.6f}")
    
    assert diff < 1e-5, f"Phase invariant mismatch: {diff}"
    print("[DONE] Background rotation test passed")

def test_phase_difference_invariance():
    """Verify that global rotation doesn't affect fluctuation dynamics."""
    print("\nTesting phase difference invariance...")
    omega_0 = 0.5
    dt = 0.005 # Smaller dt for stability
    config = SIFEConfig(omega_0=omega_0, dt=dt)
    
    key = jax.random.PRNGKey(42)
    shape = (10,)
    
    # Use stable initialization: uniform amplitude, random phase
    field1 = SIFField(
        amplitude=jnp.ones(shape),
        phase=jax.random.uniform(key, shape) * 2 * jnp.pi,
        fluctuation=jnp.zeros(shape),
        velocity_amp=jnp.zeros(shape),
        velocity_phi=jnp.zeros(shape)
    )
    # Ensure fluctuation is theta at t=0
    field1 = field1._replace(fluctuation=field1.phase)
    
    rotation = 1.0
    field2 = field1._replace(
        phase=(field1.phase + rotation) % (2 * jnp.pi),
        fluctuation=(field1.fluctuation + rotation)
    )
    
    num_steps = 50
    evolved1 = evolve_field(field1, config, num_steps)
    evolved2 = evolve_field(field2, config, num_steps)
    
    amp_diff = jnp.abs(evolved1.amplitude - evolved2.amplitude).max()
    print(f"  Amplitude max diff: {amp_diff:.6e}")
    assert jnp.isfinite(amp_diff)
    assert amp_diff < 1e-5
    
    phi_diff = (evolved2.fluctuation - evolved1.fluctuation)
    phi_diff = ((phi_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    phi_diff_error = jnp.abs(phi_diff - rotation).max()
    print(f"  Fluctuation shift error: {phi_diff_error:.6e}")
    assert phi_diff_error < 1e-5
    
    print("[DONE] Phase difference invariance test passed")

def test_hamiltonian_conservation():
    """Verify Hamiltonian conservation with new terms."""
    print("\nTesting Hamiltonian conservation...")
    omega_0 = 2.0
    dt = 0.0001
    config = SIFEConfig(omega_0=omega_0, dt=dt)
    
    key = jax.random.PRNGKey(42)
    shape = (10,)
    # Use field from initialize_field but manually ensure stability if needed
    field = initialize_field(key, shape)
    
    H_initial = compute_hamiltonian(field, config)
    
    num_steps = 200
    evolved = evolve_field(field, config, num_steps)
    H_final = compute_hamiltonian(evolved, config)
    
    rel_diff = jnp.abs(H_final - H_initial) / (jnp.abs(H_initial) + 1e-8)
    print(f"  Initial H: {H_initial:.6f}")
    print(f"  Final H:   {H_final:.6f}")
    print(f"  Relative H diff: {rel_diff:.6e}")
    
    assert jnp.isfinite(rel_diff)
    assert rel_diff < 1e-2, f"Hamiltonian drift too large: {rel_diff}"
    print("[DONE] Hamiltonian conservation test passed")

if __name__ == "__main__":
    test_background_rotation()
    test_phase_difference_invariance()
    test_hamiltonian_conservation()
