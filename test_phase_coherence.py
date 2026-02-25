import jax
import jax.numpy as jnp
from sife.model import SIFELDM, SIFELDMConfig, create_train_state
from sife.diffusion import GaussianDiffusion, DiffusionConfig
from inference import apply_attractor_relaxation

def test_phase_coherence():
    print("Testing Phase Coherence Features...")
    config = SIFELDMConfig(
        embed_dim=128,
        num_heads=4,
        num_blocks=2,
        phase_coupling_lambda=0.5,
        vocab_size=1000
    )
    
    key = jax.random.PRNGKey(42)
    model = SIFELDM(config)
    diffusion = GaussianDiffusion(DiffusionConfig())
    
    # 1. Test Positional Phase Encoding (Inside model)
    batch_size = 2
    seq_len = 16
    x = jax.random.normal(key, (batch_size, seq_len, config.embed_dim), dtype=jnp.float32)
    x = x + 1j * jax.random.normal(jax.random.split(key)[0], (batch_size, seq_len, config.embed_dim), dtype=jnp.float32)
    t = jnp.zeros((batch_size,), dtype=jnp.int32)
    
    print("Initializing Model...")
    variables = model.init(key, x, t)
    params = variables['params']
    
    print("Running Forward Pass (with Positional Phase Encoding)...")
    out = model.apply({'params': params}, x, t)
    assert out.shape == x.shape
    print("✅ Forward Pass Successful!")
    
    # 2. Test Neighbor Coupling Loss
    print("Testing Loss Function (with Neighbor Coupling)...")
    batch = {'complex_embedding': x}
    loss = model.get_loss(params, batch, key, diffusion)
    print(f"Computed Loss: {loss:.4f}")
    assert not jnp.isnan(loss)
    print("✅ Loss Computation Successful!")
    
    # 3. Test Attractor Relaxation
    print("Testing Attractor Relaxation...")
    relaxed_x = apply_attractor_relaxation(x, eta=0.1, num_steps=5)
    assert relaxed_x.shape == x.shape
    
    # Check if potential decreased
    def potential(val):
        theta = jnp.angle(val)
        diff = theta[:, 1:, :] - theta[:, :-1, :]
        return jnp.mean(1 - jnp.cos(diff))
    
    p_orig = potential(x)
    p_relaxed = potential(relaxed_x)
    print(f"Potential: {p_orig:.4f} -> {p_relaxed:.4f}")
    assert p_relaxed <= p_orig
    print("✅ Attractor Relaxation Successful!")

if __name__ == "__main__":
    try:
        test_phase_coherence()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
