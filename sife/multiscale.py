import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, List, Tuple
from .unet import ComplexConv1D

class MultiScaleConfig(NamedTuple):
    num_scales: int = 3
    downsample_factors: Tuple[int, ...] = (2, 2, 2)
    features: Tuple[int, ...] = (256, 512, 1024)

class HierarchicalField(NamedTuple):
    fields_at_scales: List[jnp.ndarray]

class MultiScaleEncoder(nn.Module):
    config: MultiScaleConfig
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> HierarchicalField:
        fields = [x]
        current = x
        for i in range(self.config.num_scales - 1):
            factor = self.config.downsample_factors[i]
            
            B, L, C = current.shape
            padded_L = L + (factor - L % factor) % factor
            if padded_L != L:
                pad_width = ((0, 0), (0, padded_L - L), (0, 0))
                current = jnp.pad(current, pad_width)
                L = padded_L
                
            pooled = current.reshape(B, L // factor, factor, C).mean(axis=2)
            current = pooled
            fields.append(current)
            
        return HierarchicalField(fields_at_scales=fields)

class MultiScaleDecoder(nn.Module):
    config: MultiScaleConfig
    
    @nn.compact
    def __call__(self, hierarchical_field: HierarchicalField) -> jnp.ndarray:
        fields = hierarchical_field.fields_at_scales
        current = fields[-1]
        
        for i in reversed(range(self.config.num_scales - 1)):
            factor = self.config.downsample_factors[i]
            target_feat = self.config.features[i]
            
            B, L, C = current.shape
            upsampled = jnp.repeat(current, factor, axis=1)
            
            skip = fields[i]
            
            # Crop upsampled if it's larger than skip (due to padding in encoder)
            if upsampled.shape[1] > skip.shape[1]:
                upsampled = upsampled[:, :skip.shape[1], :]
                
            merged = jnp.concatenate([upsampled, skip], axis=-1)
            current = ComplexConv1D(features=target_feat, kernel_size=(3,), padding='SAME')(merged)
            
        return current
