import jax
import jax.numpy as jnp
from sife.model import SIFELDM
from sife.diffusion import DDIMSampler, GaussianDiffusion

def generate_vision(model, params, config, rng, batch_size=1):
    """DDIM sampling for vision generation"""
    diffusion = GaussianDiffusion(timesteps=1000)
    sampler = DDIMSampler(diffusion)
    
    rng_A, rng_theta = jax.random.split(rng)
    H, W = config.image_size, config.image_size
    P = config.patch_size
    L = (H // P) * (W // P)
    
    noise_A = jnp.abs(jax.random.normal(rng_A, (batch_size, L, config.embed_dim)))
    noise_theta = jax.random.normal(rng_theta, (batch_size, L, config.embed_dim)) * jnp.pi
    x_t = noise_A * jnp.exp(1j * noise_theta)
    
    steps = 50
    times = jnp.linspace(1000 - 1, 0, steps, dtype=jnp.int32)
    
    for i in range(steps):
        t = int(times[i])
        t_prev = int(times[i+1]) if i < steps - 1 else -1
        t_batch = jnp.full((batch_size,), t)
        
        pred_noise = model.apply({'params': params}, x_t, t_batch, mode='vision', deterministic=True)
        x_t = sampler.sample_step(x_t, pred_noise, t, t_prev, eta=0.0)
        
    images = model.apply({'params': params}, x_t, hw_shape=(H, W), method=model.decode_image)
    return images

def generate_text(model, params, config, rng, start_tokens, max_new_tokens=50):
    """Masked diffusion inference loop for text"""
    B = start_tokens.shape[0]
    tokens = jnp.pad(start_tokens, ((0,0), (0, max_new_tokens)), constant_values=0)
    mask = (tokens == 0)
    
    for step in range(max_new_tokens):
        x_t = model.apply({'params': params}, tokens, mask=mask, method=model.encode_text)
        
        t_dummy = jnp.zeros((B,), dtype=jnp.int32)
        pred_x0 = model.apply({'params': params}, x_t, t_dummy, mode='text', deterministic=True)
        logits = model.apply({'params': params}, pred_x0, method=model.decode_symbol)
        
        pred_tokens = jnp.argmax(logits, axis=-1)
        
        first_masked = jnp.argmax(mask, axis=1)
        
        for b in range(B):
            idx = first_masked[b]
            if mask[b, idx]:
                tokens = tokens.at[b, idx].set(pred_tokens[b, idx])
                mask = mask.at[b, idx].set(False)
                
    return tokens
