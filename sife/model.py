import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Optional, Dict, Any, Tuple
import optax

from .unet import UnifiedSIFETransformer, ComplexPatchEncoder
from .field import SIFEConfig, SIFField, compute_hamiltonian

class SIFELDMConfig(NamedTuple):
    vocab_size: int = 50000
    max_seq_len: int = 1024
    embed_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    patch_size: int = 16
    dropout_rate: float = 0.1
    physics_config: SIFEConfig = SIFEConfig()
    guidance_scale: float = 0.1
    image_size: int = 256
    channels: int = 3

class ImageEncoder(nn.Module):
    @nn.compact
    def __call__(self, images):
        r, g, b = images[..., 0], images[..., 1], images[..., 2]
        max_c = jnp.max(images, axis=-1)
        min_c = jnp.min(images, axis=-1)
        delta = max_c - min_c
        
        v = max_c
        
        rc = (max_c - r) / (delta + 1e-6)
        gc = (max_c - g) / (delta + 1e-6)
        bc = (max_c - b) / (delta + 1e-6)
        
        h_r = jnp.where(r == max_c, bc - gc, 0.0)
        h_g = jnp.where(g == max_c, 2.0 + rc - bc, h_r)
        h_b = jnp.where(b == max_c, 4.0 + gc - rc, h_g)
        h = jnp.where(delta == 0.0, 0.0, h_b)
        h = (h / 6.0) % 1.0
        
        phase = h * 2 * jnp.pi - jnp.pi
        amp = v
        
        amp_mean = jnp.mean(amp, axis=(1,2), keepdims=True)
        amp_std = jnp.std(amp, axis=(1,2), keepdims=True) + 1e-6
        amp = (amp - amp_mean) / amp_std
        amp = nn.softplus(amp)
        
        complex_img = amp[..., None] * jnp.exp(1j * phase[..., None])
        return jnp.tile(complex_img, (1, 1, 1, images.shape[-1]))

class TextEncoder(nn.Module):
    vocab_size: int
    embed_dim: int
    
    @nn.compact
    def __call__(self, input_ids, mask=None):
        phase_emb = nn.Embed(self.vocab_size, self.embed_dim)(input_ids)
        phase = jnp.tanh(phase_emb) * jnp.pi
        
        amp = jnp.ones_like(phase)
        if mask is not None:
            amp = jnp.where(mask[..., None], 0.0, 1.0)
            
        return amp * jnp.exp(1j * phase)

class ImageDecoder(nn.Module):
    channels: int
    patch_size: int
    
    @nn.compact
    def __call__(self, complex_sequence, hw_shape):
        B, L, D = complex_sequence.shape
        H, W = hw_shape
        
        amp = jnp.abs(complex_sequence)
        phase = jnp.angle(complex_sequence)
        
        feats = jnp.concatenate([amp, jnp.cos(phase), jnp.sin(phase)], axis=-1)
        x = nn.Dense(self.patch_size * self.patch_size * self.channels)(feats)
        x = nn.sigmoid(x)
        
        P = self.patch_size
        x = x.reshape(B, H // P, W // P, P, P, self.channels)
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape(B, H, W, self.channels)
        return x

class SymbolDecoder(nn.Module):
    vocab_size: int
    
    @nn.compact
    def __call__(self, complex_sequence):
        amp = jnp.abs(complex_sequence)
        phase = jnp.angle(complex_sequence)
        
        feats = jnp.concatenate([jnp.cos(phase), jnp.sin(phase), amp], axis=-1)
        logits = nn.Dense(self.vocab_size)(feats)
        return logits

class SIFELDM(nn.Module):
    config: SIFELDMConfig
    
    def setup(self):
        self.transformer = UnifiedSIFETransformer(
            num_layers=self.config.num_layers,
            num_heads=self.config.num_heads,
            head_dim=self.config.embed_dim // self.config.num_heads,
            mlp_dim=self.config.mlp_dim,
            out_dim=self.config.embed_dim,
            dropout_rate=self.config.dropout_rate
        )
        self.patch_encoder = ComplexPatchEncoder(
            patch_size=self.config.patch_size,
            out_dim=self.config.embed_dim
        )
        self.text_encoder = TextEncoder(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim
        )
        self.image_decoder = ImageDecoder(
            channels=self.config.channels,
            patch_size=self.config.patch_size
        )
        self.symbol_decoder = SymbolDecoder(
            vocab_size=self.config.vocab_size
        )
        self.image_encoder = ImageEncoder()

    def __call__(self, x, t, context=None, mode='vision', mask=None, deterministic=False):
        if mode == 'vision':
            return self.transformer(x, t, context=context, deterministic=deterministic)
        elif mode == 'text':
            return self.transformer(x, t, context=context, deterministic=deterministic)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
    def encode_image(self, x):
        return self.image_encoder(x)
        
    def encode_patch(self, x):
        return self.patch_encoder(x)
        
    def decode_image(self, x, hw_shape):
        return self.image_decoder(x, hw_shape)
        
    def encode_text(self, x, mask):
        return self.text_encoder(x, mask)
        
    def decode_symbol(self, x):
        return self.symbol_decoder(x)

def get_loss(model, params, batch, rng, diffusion_obj, config, mode='vision'):
    if mode == 'vision':
        images = batch['image']
        B = images.shape[0]
        t = jax.random.randint(rng, (B,), 0, diffusion_obj.timesteps)
        
        # Encode image
        x0_img = model.apply({'params': params}, images, method=model.encode_image)
        x0 = model.apply({'params': params}, x0_img, method=model.encode_patch)
        
        noise_A = jax.random.normal(rng, x0.shape)
        noise_theta = jax.random.normal(rng, x0.shape)
        noise = noise_A + 1j * noise_theta
        
        x_t = diffusion_obj.q_sample(x0, t, noise)
        
        rng, dropout_rng = jax.random.split(rng)
        pred_noise = model.apply({'params': params}, x_t, t, mode='vision', deterministic=False, rngs={'dropout': dropout_rng})
        
        loss = jnp.mean(jnp.abs(pred_noise - noise)**2)
        
        # Physics regularization
        field = SIFField(
            amplitude=jnp.abs(x_t), phase=jnp.angle(x_t), fluctuation=jnp.angle(x_t),
            velocity_amp=jnp.zeros_like(x_t.real), velocity_phi=jnp.zeros_like(x_t.real)
        )
        phys_loss = compute_hamiltonian(field, config.physics_config)
        loss = loss + 0.01 * jnp.mean(phys_loss)
        return loss
        
    elif mode == 'text':
        tokens = batch['tokens']
        mask = batch['mask'] # Assume binary mask provided
        B = tokens.shape[0]
        t = jnp.zeros((B,)) # Timesteps unused for text but required by signature
        
        x0 = model.apply({'params': params}, tokens, mask=mask, method=model.encode_text)
        
        rng, dropout_rng = jax.random.split(rng)
        pred_x0 = model.apply({'params': params}, x0, t, mode='text', deterministic=False, rngs={'dropout': dropout_rng})
        logits = model.apply({'params': params}, pred_x0, method=model.decode_symbol)
        
        # CrossEntropy on masked positions
        one_hot = jax.nn.one_hot(tokens, config.vocab_size)
        ce_loss = optax.softmax_cross_entropy(logits, one_hot)
        loss = jnp.sum(ce_loss * mask) / (jnp.sum(mask) + 1e-8)
        
        field = SIFField(
            amplitude=jnp.abs(pred_x0), phase=jnp.angle(pred_x0), fluctuation=jnp.angle(pred_x0),
            velocity_amp=jnp.zeros_like(pred_x0.real), velocity_phi=jnp.zeros_like(pred_x0.real)
        )
        phys_loss = compute_hamiltonian(field, config.physics_config)
        loss = loss + 0.01 * jnp.mean(phys_loss)
        return loss
