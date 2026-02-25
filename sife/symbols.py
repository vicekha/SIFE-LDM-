"""
SIFE-LDM: Symbolic Phase-Collapse Decoder
==========================================

Implements the bridge between continuous SIFE fields and discrete symbols.
Uses phase-coherence to identify "proto-concepts" in the field and
maps them to a symbolic vocabulary.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Optional, List, Any
from .field import Array, PRNGKey

class CoherenceMeasure(nn.Module):
    """
    Computes local phase coherence across spatial patches.
    $c_p = | \frac{1}{N_p} \sum_{n \in p} e^{i\theta_n} |$
    """
    patch_size: int = 8
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Complex field of shape (batch, seq_len, features)
        Returns:
            Coherence map of shape (batch, seq_len // patch_size, features)
        """
        batch, seq_len, features = x.shape
        num_patches = seq_len // self.patch_size
        
        # Reshape to separate patches
        # (batch, num_patches, patch_size, features)
        patches = x.reshape(batch, num_patches, self.patch_size, features)
        
        # Phase coherence: magnitude of the average phase vector
        # Normalize by amplitude to focus on phase agreement
        normalised = patches / jnp.where(jnp.abs(patches) > 1e-8, jnp.abs(patches), 1.0)
        coherence = jnp.abs(jnp.mean(normalised, axis=2))
        
        return coherence

class SymbolDecoder(nn.Module):
    """
    Decodes symbols from coherent field patches.
    """
    vocab_size: int
    patch_size: int = 8
    features: int = 256
    
    @nn.compact
    def __call__(self, x: Array, threshold: float = 0.7) -> Tuple[Array, Array]:
        """
        Args:
            x: Complex field (batch, seq_len, features)
        Returns:
            - logits: (batch, num_patches, vocab_size)
            - mask: boolean mask of coherent patches (batch, num_patches)
        """
        batch, seq_len, features = x.shape
        num_patches = seq_len // self.patch_size
        
        # 1. Compute coherence
        coherence = CoherenceMeasure(patch_size=self.patch_size)(x)
        
        # 2. Extract patch features (average complex value)
        patch_features = x.reshape(batch, num_patches, self.patch_size, features).mean(axis=2)
        
        # 3. Predict symbols from features
        # Flatten features (real, imag) for standard MLP
        flat_features = jnp.concatenate([jnp.real(patch_features), jnp.imag(patch_features)], axis=-1)
        
        h = nn.Dense(self.features)(flat_features)
        h = nn.gelu(h)
        logits = nn.Dense(self.vocab_size)(h)
        
        # 4. Create mask based on coherence threshold
        # We take the mean coherence across feature dimension or max?
        # Max coherence across features indicates a strong signal in at least one frequency.
        mask = jnp.max(coherence, axis=-1) > threshold
        
        return logits, mask

class SymbolEncoder(nn.Module):
    """
    Encodes symbolic sequences back into the complex SIFE field.
    Provides bidirectional consistency.
    """
    vocab_size: int
    features: int = 256
    patch_size: int = 8
    
    @nn.compact
    def __call__(self, symbols: Array, seq_len: int) -> Array:
        """
        Args:
            symbols: Symbol indices (batch, num_patches)
            seq_len: Desired field sequence length
        Returns:
            Complex field (batch, seq_len, features)
        """
        batch, num_patches = symbols.shape
        
        # 1. Embed symbols
        # Real and imaginary parts as separate embeddings
        embed_r = nn.Embed(self.vocab_size, self.features)(symbols)
        embed_i = nn.Embed(self.vocab_size, self.features)(symbols)
        
        patch_complex = embed_r + 1j * embed_i
        
        # 2. Expand patches to full sequence
        # (batch, num_patches, 1, features) -> (batch, num_patches, patch_size, features)
        field = jnp.repeat(patch_complex[:, :, jnp.newaxis, :], self.patch_size, axis=2)
        
        # 3. Reshape to sequence
        field = field.reshape(batch, seq_len, self.features)
        
        return field
