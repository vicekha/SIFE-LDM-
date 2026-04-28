import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple

def CoherenceMeasure(complex_sequence: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """
    Computes the phase coherence of a complex patch or sequence.
    C = | 1/N \sum e^{i \theta} |
    """
    phase = jnp.angle(complex_sequence)
    phase_factors = jnp.exp(1j * phase)
    mean_phase_factor = jnp.mean(phase_factors, axis=axis)
    return jnp.abs(mean_phase_factor)

class SymbolDecoder(nn.Module):
    vocab_size: int
    
    @nn.compact
    def __call__(self, complex_patch):
        """
        Maps coherent phase patches to discrete token logits.
        Expects complex_patch to have shape (B, num_patches, patch_size_or_dim)
        """
        amp = jnp.abs(complex_patch)
        phase = jnp.angle(complex_patch)
        
        mean_amp = jnp.mean(amp, axis=-1, keepdims=False)
        phase_factors = jnp.exp(1j * phase)
        mean_phase_factor = jnp.mean(phase_factors, axis=-1, keepdims=False)
        
        cos_theta = jnp.real(mean_phase_factor)
        sin_theta = jnp.imag(mean_phase_factor)
        
        feats = jnp.stack([cos_theta, sin_theta, mean_amp], axis=-1)
        logits = nn.Dense(self.vocab_size)(feats)
        return logits

class SymbolEncoder(nn.Module):
    vocab_size: int
    embed_dim: int
    patch_size: int
    
    @nn.compact
    def __call__(self, token_ids):
        """
        Maps discrete tokens back to phase patches.
        """
        phase_emb = nn.Embed(self.vocab_size, self.embed_dim)(token_ids)
        phase = jnp.tanh(phase_emb) * jnp.pi
        
        amp = jnp.ones_like(phase)
        base_complex = amp * jnp.exp(1j * phase)
        
        # Tile over patch dimension: (B, seq_len, patch_size, embed_dim)
        complex_patches = jnp.tile(base_complex[..., None, :], (1, 1, self.patch_size, 1))
        return complex_patches
