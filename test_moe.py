import jax
import jax.numpy as jnp
from sife.model import SIFELDM, SIFELDMConfig
from sife.tokenizer import SIFETokenizer

def test_moe_forward():
    print("Initializing MoE Model...")
    config = SIFELDMConfig(
        embed_dim=128,
        num_heads=4,
        num_blocks=2,
        num_experts=4, # Enable MoE
        vocab_size=1000
    )
    
    model = SIFELDM(config)
    key = jax.random.PRNGKey(42)
    
    # Dummy inputs
    batch_size = 2
    seq_len = 16
    x = jax.random.normal(key, (batch_size, seq_len, config.embed_dim), dtype=jnp.float32)
    x = x + 1j * jax.random.normal(jax.random.split(key)[0], (batch_size, seq_len, config.embed_dim), dtype=jnp.float32)
    
    t = jnp.array([10, 50])
    
    # Initialize params
    print("Initializing Parameters...")
    variables = model.init(key, x, t, deterministic=True)
    params = variables['params']
    
    # Forward pass
    print("Running Forward Pass...")
    out = model.apply({'params': params}, x, t, deterministic=True)
    
    print(f"Output Shape: {out.shape}")
    assert out.shape == x.shape
    print("✅ MoE Forward Pass Successful!")

if __name__ == "__main__":
    try:
        test_moe_forward()
    except Exception as e:
        print(f"❌ Test Failed: {e}")
        import traceback
        traceback.print_exc()
