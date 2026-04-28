import jax
import jax.numpy as jnp
import optax
from sife.model import SIFELDM, SIFELDMConfig, get_loss
from sife.diffusion import GaussianDiffusion, MaskedDiffusion
import flax.linen as nn

jax.config.update("jax_debug_nans", True)

class MultiInitModel(nn.Module):
    model: SIFELDM
    @nn.compact
    def __call__(self, img, t, tokens, mask):
        x0_img = self.model.image_encoder(img)
        x0 = self.model.patch_encoder(x0_img)
        v = self.model(x0, t, mode='vision')
        hw = (img.shape[1], img.shape[2])
        img_out = self.model.image_decoder(v, hw)
        x_t = self.model.text_encoder(tokens, mask)
        t_out = self.model(x_t, t, mode='text')
        logits = self.model.symbol_decoder(t_out)
        return img_out, logits

def debug_nan():
    config = SIFELDMConfig(
        vocab_size=100,
        max_seq_len=16,
        embed_dim=128,
        num_layers=4,
        num_heads=4,
        mlp_dim=256,
        patch_size=8,
        image_size=32
    )
    
    rng = jax.random.PRNGKey(42)
    model = SIFELDM(config)
    
    dummy_img = jnp.ones((1, 32, 32, 3))
    dummy_t = jnp.zeros((1,), dtype=jnp.int32)
    dummy_tokens = jnp.zeros((1, 16), dtype=jnp.int32)
    dummy_mask = jnp.zeros((1, 16), dtype=jnp.bool_)
    
    rng, init_rng = jax.random.split(rng)
    init_model = MultiInitModel(model=model)
    variables = init_model.init(init_rng, dummy_img, dummy_t, dummy_tokens, dummy_mask)
    params = variables['params']['model']
    
    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(params)
    
    diffusion = GaussianDiffusion()
    
    print("Starting diagnostic training loop...")
    
    for step in range(100):
        rng, step_rng = jax.random.split(rng)
        
        batch = {
            'image': jax.random.normal(step_rng, (4, 32, 32, 3)),
            'tokens': jax.random.randint(step_rng, (4, 16), 0, 100),
            'mask': jnp.zeros((4, 16), dtype=jnp.bool_)
        }
        
        def loss_fn(p):
            # Test Vision
            v_loss = get_loss(model, p, batch, step_rng, diffusion, config, mode='vision')
            # Test Text
            t_loss = get_loss(model, p, batch, step_rng, MaskedDiffusion(), config, mode='text')
            return v_loss + t_loss
            
        loss, grads = jax.value_and_grad(loss_fn)(params)
        
        if jnp.isnan(loss):
            print(f"NaN detected at step {step}!")
            break
            
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss:.4f}")

    print("Diagnostic complete.")

if __name__ == "__main__":
    debug_nan()
