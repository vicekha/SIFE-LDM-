#!/usr/bin/env python3
"""
SIFE-LDM Test Suite
===================

Basic tests to verify the SIFE-LDM implementation.

Run with: pytest tests/test_sife.py -v

Author: SIFE-LDM Research Team
License: MIT
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sife.field import (
    SIFEConfig, SIFField, initialize_field,
    compute_hamiltonian, truth_potential, leapfrog_step, evolve_field
)
from sife.diffusion import (
    DiffusionConfig, GaussianDiffusion, DDIMSampler,
    cosine_lr_schedule
)
from sife.multiscale import (
    MultiScaleConfig, HierarchicalField,
    create_multiscale_config, initialize_hierarchical_field
)


class TestSIFEField:
    """Tests for SIFE field dynamics."""

    def test_config_creation(self):
        """Test SIFE config creation."""
        config = SIFEConfig(
            m=1.0,
            k=1.0,
            alpha=0.25,
            beta=1.0,
            gamma=0.1
        )
        assert config.m == 1.0
        assert config.k == 1.0
        assert config.gamma == 0.1

    def test_field_initialization(self):
        """Test field initialization."""
        key = jax.random.PRNGKey(0)
        shape = (128,)

        field = initialize_field(key, shape)

        assert field.amplitude.shape == shape
        assert field.phase.shape == shape
        assert field.velocity_amp.shape == shape
        assert field.velocity_phase.shape == shape

        # Amplitude should be positive
        assert jnp.all(field.amplitude >= 0)

        # Phase should be in [0, 2π)
        assert jnp.all(field.phase >= 0)
        assert jnp.all(field.phase < 2 * jnp.pi)

    def test_complex_field_conversion(self):
        """Test conversion to complex field."""
        key = jax.random.PRNGKey(0)
        shape = (64,)

        field = initialize_field(key, shape)
        complex_field = field.complex_field

        assert complex_field.shape == shape
        assert jnp.allclose(jnp.abs(complex_field), field.amplitude)

    def test_hamiltonian_computation(self):
        """Test Hamiltonian computation."""
        key = jax.random.PRNGKey(0)
        shape = (32,)
        config = SIFEConfig()

        field = initialize_field(key, shape)
        H = compute_hamiltonian(field, config)

        # Hamiltonian should be a scalar
        assert H.shape == ()

        # Hamiltonian should be finite
        assert jnp.isfinite(H)

    def test_truth_potential(self):
        """Test truth potential computation."""
        key = jax.random.PRNGKey(0)
        shape = (32,)

        amplitude = jnp.abs(jax.random.normal(key, shape)) + 0.1
        phase = 2 * jnp.pi * jax.random.uniform(jax.random.split(key)[0], shape)

        phi_T = truth_potential(amplitude, phase)

        # Truth potential should be a scalar
        assert phi_T.shape == ()

        # For coherent phase (all same), truth potential should be positive
        coherent_amplitude = jnp.ones(shape)
        coherent_phase = jnp.zeros(shape)
        phi_T_coherent = truth_potential(coherent_amplitude, coherent_phase)
        assert phi_T_coherent > 0

    def test_leapfrog_step(self):
        """Test leapfrog integration step."""
        key = jax.random.PRNGKey(0)
        shape = (32,)
        config = SIFEConfig(dt=0.001)

        field = initialize_field(key, shape)
        new_field = leapfrog_step(field, config)

        # Shape should be preserved
        assert new_field.amplitude.shape == shape
        assert new_field.phase.shape == shape

        # Energy should be approximately conserved
        H_initial = compute_hamiltonian(field, config)
        H_final = compute_hamiltonian(new_field, config)

        # Allow for small numerical errors
        assert jnp.abs(H_final - H_initial) < 1.0

    def test_field_evolution(self):
        """Test field evolution over multiple steps."""
        key = jax.random.PRNGKey(0)
        shape = (64,)
        config = SIFEConfig(dt=0.001)

        field = initialize_field(key, shape)
        evolved = evolve_field(field, config, num_steps=10)

        assert evolved.amplitude.shape == shape
        assert jnp.all(jnp.isfinite(evolved.amplitude))
        assert jnp.all(jnp.isfinite(evolved.phase))


class TestDiffusion:
    """Tests for diffusion process."""

    def test_diffusion_config(self):
        """Test diffusion configuration."""
        config = DiffusionConfig(
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            schedule='cosine'
        )

        assert config.num_timesteps == 1000
        assert config.schedule == 'cosine'

    def test_gaussian_diffusion_creation(self):
        """Test Gaussian diffusion creation."""
        config = DiffusionConfig(num_timesteps=100)
        diffusion = GaussianDiffusion(config)

        assert diffusion.num_timesteps == 100
        assert len(diffusion.betas) == 100
        assert len(diffusion.alphas_cumprod) == 100

    def test_q_sample(self):
        """Test forward diffusion sampling."""
        config = DiffusionConfig(num_timesteps=100)
        diffusion = GaussianDiffusion(config)

        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(key, (4, 32, 64))
        x_0 = x_0 + 1j * jax.random.normal(jax.random.split(key)[0], (4, 32, 64))

        t = jnp.array([0, 25, 50, 99])

        x_t = diffusion.q_sample(x_0, t, key)

        assert x_t.shape == x_0.shape

        # At t=0, x_t should be close to x_0
        assert jnp.allclose(x_t[0], x_0[0], atol=0.1)

    def test_predict_x0(self):
        """Test x_0 prediction from noise."""
        # Use config without clipping to verify the math
        config = DiffusionConfig(num_timesteps=100, clip_denoised=False)
        diffusion = GaussianDiffusion(config)

        key = jax.random.PRNGKey(0)
        x_0 = jax.random.normal(key, (1, 32, 64))
        x_0 = x_0 + 1j * jax.random.normal(jax.random.split(key)[0], (1, 32, 64))

        t = jnp.array([50])

        # Sample noise
        noise = jax.random.normal(key, x_0.shape)
        noise = noise + 1j * jax.random.normal(jax.random.split(key)[0], x_0.shape)

        # Get x_t
        x_t = diffusion.q_sample(x_0, t, key, noise)

        # Predict x_0
        x_0_pred = diffusion.predict_x0_from_epsilon(x_t, noise, t)

        # Should recover x_0
        assert jnp.allclose(x_0_pred, x_0, atol=0.1)

    def test_cosine_lr_schedule(self):
        """Test cosine learning rate schedule."""
        total_steps = 10000
        warmup_steps = 1000
        lr_max = 1e-4

        # During warmup
        lr_warmup = cosine_lr_schedule(500, total_steps, warmup_steps, lr_max)
        assert lr_warmup == lr_max * 500 / warmup_steps

        # After warmup
        lr_after = cosine_lr_schedule(5000, total_steps, warmup_steps, lr_max)
        assert 0 < lr_after < lr_max

        # At the end
        lr_end = cosine_lr_schedule(total_steps, total_steps, warmup_steps, lr_max)
        assert lr_end >= 0


class TestMultiscale:
    """Tests for multi-scale architecture."""

    def test_multiscale_config(self):
        """Test multi-scale configuration."""
        config = create_multiscale_config(num_levels=3, base_features=64)

        assert config.num_levels == 3
        assert len(config.feature_multipliers) == 3
        assert len(config.lattice_spacings) == 3

    def test_hierarchical_field_initialization(self):
        """Test hierarchical field initialization."""
        key = jax.random.PRNGKey(0)
        config = create_multiscale_config(num_levels=3)
        shapes = [(128, 64), (64, 64), (32, 64)]

        field = initialize_hierarchical_field(key, shapes, config)

        assert field.num_levels == 3
        assert len(field.amplitudes) == 3
        assert len(field.phases) == 3

        for i, shape in enumerate(shapes):
            assert field.amplitudes[i].shape == shape

    def test_hierarchical_complex_fields(self):
        """Test complex field extraction from hierarchy."""
        key = jax.random.PRNGKey(0)
        config = create_multiscale_config(num_levels=2)
        shapes = [(64, 32), (32, 32)]

        field = initialize_hierarchical_field(key, shapes, config)
        complex_fields = field.get_complex_fields()

        assert len(complex_fields) == 2

        for i, cf in enumerate(complex_fields):
            assert cf.shape == shapes[i]


class TestIntegration:
    """Integration tests."""

    def test_full_pipeline(self):
        """Test full SIFE-LDM pipeline."""
        # Initialize
        key = jax.random.PRNGKey(42)

        # Create configs
        sife_config = SIFEConfig(dt=0.001)
        diffusion_config = DiffusionConfig(num_timesteps=100)
        multiscale_config = create_multiscale_config(num_levels=2)

        # Initialize field
        field = initialize_field(key, (64,))

        # Compute Hamiltonian
        H = compute_hamiltonian(field, sife_config)
        assert jnp.isfinite(H)

        # Evolve field
        evolved = evolve_field(field, sife_config, num_steps=5)
        assert jnp.all(jnp.isfinite(evolved.amplitude))

        # Test diffusion
        diffusion = GaussianDiffusion(diffusion_config)

        # Create complex sample
        key, subkey = jax.random.split(key)
        x_0 = jax.random.normal(subkey, (1, 32, 64))
        x_0 = x_0 + 1j * jax.random.normal(jax.random.split(subkey)[0], (1, 32, 64))

        t = jnp.array([50])

        # Forward diffusion
        x_t = diffusion.q_sample(x_0, t, key)
        assert x_t.shape == x_0.shape

        print("Full pipeline test passed!")


def test_all():
    """Run all tests."""
    print("Running SIFE-LDM tests...")

    # Field tests
    print("\nTesting SIFE field...")
    test = TestSIFEField()
    test.test_config_creation()
    test.test_field_initialization()
    test.test_complex_field_conversion()
    test.test_hamiltonian_computation()
    test.test_truth_potential()
    test.test_leapfrog_step()
    test.test_field_evolution()
    print("✓ SIFE field tests passed")

    # Diffusion tests
    print("\nTesting diffusion...")
    test = TestDiffusion()
    test.test_diffusion_config()
    test.test_gaussian_diffusion_creation()
    test.test_q_sample()
    test.test_predict_x0()
    test.test_cosine_lr_schedule()
    print("✓ Diffusion tests passed")

    # Multiscale tests
    print("\nTesting multi-scale...")
    test = TestMultiscale()
    test.test_multiscale_config()
    test.test_hierarchical_field_initialization()
    test.test_hierarchical_complex_fields()
    print("✓ Multi-scale tests passed")

    # Integration tests
    print("\nRunning integration test...")
    test = TestIntegration()
    test.test_full_pipeline()
    print("✓ Integration test passed")

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50)


if __name__ == '__main__':
    test_all()
