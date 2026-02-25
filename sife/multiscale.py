"""
SIFE-LDM: Multi-Scale Nested Architecture
=========================================

Implements the hierarchical SIFE architecture where multiple
concentric rings (or nested tori) operate at different scales.

This enables:
- Microscopic patterns: Fine-grained token-level features
- Mesoscopic patterns: Phrase and sentence structure
- Macroscopic patterns: Document-level semantics

Each level has its own field Ψ^(ℓ) with different lattice spacing h_ℓ
and time steps Δt_ℓ. Rings are coupled through interpolation kernels.

Author: SIFE-LDM Research Team
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax
import flax.linen as nn
from typing import Tuple, Optional, List, Sequence, NamedTuple, Dict, Any
from functools import partial
import math

# Type aliases
Array = jnp.ndarray
PRNGKey = jnp.ndarray


class MultiScaleConfig(NamedTuple):
    """Configuration for multi-scale SIFE architecture."""
    num_levels: int = 3
    base_features: int = 64
    feature_multipliers: Sequence[int] = (1, 2, 4)
    lattice_spacings: Sequence[float] = (1.0, 2.0, 4.0)
    time_scales: Sequence[float] = (0.01, 0.02, 0.04)
    coupling_strengths: Sequence[float] = (0.1, 0.1, 0.1)
    attention_levels: Sequence[bool] = (False, True, True)
    use_context_persistence: bool = True # Enable Global Context Field


class HierarchicalField(NamedTuple):
    """
    Hierarchical SIFE field across multiple scales.
    
    Each level ℓ has:
    - amplitude^(ℓ): Amplitude at scale ℓ
    - phase^(ℓ): Phase at scale ℓ
    - velocity_amp^(ℓ): Velocity of amplitude at scale ℓ
    - velocity_phase^(ℓ): Velocity of phase at scale ℓ
    """
    amplitudes: Sequence[Array]      # List of amplitude fields
    phases: Sequence[Array]          # List of phase fields
    fluctuations: Sequence[Array]    # List of fluctuation fields
    velocities_amp: Sequence[Array]  # List of amplitude velocities
    velocities_phi: Sequence[Array]  # List of fluctuation velocities
    
    @property
    def num_levels(self) -> int:
        return len(self.amplitudes)
    
    def get_level(self, level: int) -> 'SIFField':
        """Get the field at a specific level."""
        from .field import SIFField
        return SIFField(
            amplitude=self.amplitudes[level],
            phase=self.phases[level],
            fluctuation=self.fluctuations[level],
            velocity_amp=self.velocities_amp[level],
            velocity_phi=self.velocities_phi[level]
        )
    
    def get_complex_fields(self) -> List[Array]:
        """Get complex field representation at all levels."""
        return [
            amp * jnp.exp(1j * phase)
            for amp, phase in zip(self.amplitudes, self.phases)
        ]


def initialize_hierarchical_field(
    key: PRNGKey,
    shapes: Sequence[Tuple[int, ...]],
    config: MultiScaleConfig,
    init_scale: float = 0.1
) -> HierarchicalField:
    """
    Initialize a hierarchical field across all scales.
    
    Args:
        key: Random key
        shapes: Shapes for each level (from finest to coarsest)
        config: Multi-scale configuration
        init_scale: Initialization scale
    
    Returns:
        HierarchicalField with initialized state at all levels
    """
    from .field import initialize_field, SIFEConfig
    
    amplitudes = []
    phases = []
    fluctuations = []
    velocities_amp = []
    velocities_phi = []
    
    for level, shape in enumerate(shapes):
        key, subkey = jax.random.split(key)
        
        # Initialize field
        field = initialize_field(subkey, shape, init_scale)
        
        amplitudes.append(field.amplitude)
        phases.append(field.phase)
        fluctuations.append(field.fluctuation)
        velocities_amp.append(field.velocity_amp)
        velocities_phi.append(field.velocity_phi)
    
    return HierarchicalField(
        amplitudes=amplitudes,
        phases=phases,
        fluctuations=fluctuations,
        velocities_amp=velocities_amp,
        velocities_phi=velocities_phi
    )


def create_interpolation_kernel(
    source_size: int,
    target_size: int,
    kernel_type: str = 'linear'
) -> Array:
    """
    Create interpolation kernel for coupling between scales.
    
    Args:
        source_size: Size of source field
        target_size: Size of target field
        kernel_type: Type of interpolation ('linear', 'cubic', 'nearest')
    
    Returns:
        Interpolation kernel matrix of shape (target_size, source_size)
    """
    if kernel_type == 'nearest':
        # Nearest neighbor
        scale = source_size / target_size
        indices = jnp.arange(target_size)
        source_indices = jnp.minimum((indices * scale).astype(int), source_size - 1)
        kernel = jnp.zeros((target_size, source_size))
        kernel = kernel.at[jnp.arange(target_size), source_indices].set(1.0)
    
    elif kernel_type == 'linear':
        # Bilinear interpolation
        scale = source_size / target_size
        indices = jnp.arange(target_size)
        
        # Source positions (floating point)
        source_pos = indices * scale
        
        # Integer positions
        pos_low = jnp.floor(source_pos).astype(int)
        pos_high = jnp.minimum(pos_low + 1, source_size - 1)
        
        # Interpolation weights
        weight_high = source_pos - pos_low
        weight_low = 1 - weight_high
        
        # Build kernel
        kernel = jnp.zeros((target_size, source_size))
        kernel = kernel.at[jnp.arange(target_size), pos_low].set(weight_low)
        kernel = kernel.at[jnp.arange(target_size), pos_high].set(weight_high)
    
    elif kernel_type == 'cubic':
        # Bicubic interpolation (simplified)
        scale = source_size / target_size
        indices = jnp.arange(target_size)
        source_pos = indices * scale
        
        kernel = jnp.zeros((target_size, source_size))
        
        for i in range(target_size):
            center = source_pos[i]
            
            # Cubic kernel spans 4 points
            for j in range(max(0, int(center) - 1), min(source_size, int(center) + 3)):
                dist = abs(j - center)
                if dist < 1:
                    weight = (1.5 * dist - 2.5) * dist * dist + 1
                elif dist < 2:
                    weight = ((-0.5 * dist + 2.5) * dist - 4) * dist + 2
                else:
                    weight = 0
                
                kernel = kernel.at[i, j].set(weight)
    
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    return kernel


def interpolate_field(
    field: Array,
    source_size: int,
    target_size: int,
    kernel_type: str = 'linear'
) -> Array:
    """
    Interpolate a field between scales.
    
    Args:
        field: Complex field to interpolate
        source_size: Current size
        target_size: Target size
        kernel_type: Interpolation type
    
    Returns:
        Interpolated field
    """
    kernel = create_interpolation_kernel(source_size, target_size, kernel_type)
    
    # Apply to real and imaginary parts separately
    if field.ndim == 1:
        real_interp = kernel @ jnp.real(field)
        imag_interp = kernel @ jnp.imag(field)
    elif field.ndim == 2:
        # 2D interpolation (apply kernel to both dimensions)
        real_interp = kernel @ jnp.real(field) @ kernel.T
        imag_interp = kernel @ jnp.imag(field) @ kernel.T
    else:
        raise ValueError(f"Unsupported field dimension: {field.ndim}")
    
    return real_interp + 1j * imag_interp


def compute_cross_scale_coupling(
    field: HierarchicalField,
    config: MultiScaleConfig
) -> List[Tuple[Array, Array]]:
    """
    Compute the coupling between adjacent scales.
    
    The coupling term from level ℓ' to level ℓ is:
    (λ_{ℓ,ℓ'} / M) Σ_{n_ℓ, n_{ℓ'}} κ(n_ℓ, n_{ℓ'}) e^{i(θ_{n_{ℓ'}} - θ_{n_ℓ})} Ψ^{(ℓ')}_{n_{ℓ'}}
    
    Args:
        field: Hierarchical field
        config: Multi-scale configuration
    
    Returns:
        List of (amplitude_coupling, phase_coupling) tuples for each level
    """
    complex_fields = field.get_complex_fields()
    
    couplings = []
    
    for level in range(field.num_levels):
        amp_coupling = jnp.zeros_like(field.amplitudes[level])
        phase_coupling = jnp.zeros_like(field.phases[level])
        
        for other_level in range(field.num_levels):
            if other_level == level:
                continue
            
            # Get coupling strength
            idx = min(level, other_level)
            lambda_coupling = config.coupling_strengths[idx]
            
            # Get sizes
            source_size = field.amplitudes[other_level].shape[0]
            target_size = field.amplitudes[level].shape[0]
            
            # Interpolate other field to current scale
            other_field_interp = interpolate_field(
                complex_fields[other_level],
                source_size,
                target_size
            )
            
            # Compute phase difference
            phase_diff = jnp.angle(other_field_interp) - field.phases[level]
            
            # Coupling contribution
            coupling = lambda_coupling * jnp.exp(1j * phase_diff) * other_field_interp
            
            amp_coupling = amp_coupling + jnp.abs(coupling)
            phase_coupling = phase_coupling + jnp.angle(coupling)
        
        couplings.append((amp_coupling, phase_coupling))
    
    return couplings


def evolve_hierarchical_field(
    field: HierarchicalField,
    config: MultiScaleConfig,
    sife_configs: List['SIFEConfig'],
    num_steps: int
) -> HierarchicalField:
    """
    Evolve the hierarchical field for multiple time steps.
    
    Each level evolves according to its own SIFE dynamics,
    with coupling between levels computed at each step.
    
    Args:
        field: Initial hierarchical field
        config: Multi-scale configuration
        sife_configs: SIFE config for each level
        num_steps: Number of time steps
    
    Returns:
        Evolved hierarchical field
    """
    from .field import evolve_field, SIFField
    
    # Identify context field (index of coarsest level)
    ctx_idx = field.num_levels - 1
    ctx_complex = field.get_complex_fields()[ctx_idx]
    
    # Evolve each level
    new_amplitudes = []
    new_phases = []
    new_fluctuations = []
    new_velocities_amp = []
    new_velocities_phi = []
    
    total_ctx_contribution = jnp.zeros_like(ctx_complex)
    
    for level in range(field.num_levels):
        # Create field for this level
        level_field = field.get_level(level)
        
        # SIFEConfig for this level
        sife_config = sife_configs[level]
        
        # AGI: Apply specialized context dynamics to the coarsest level
        if config.use_context_persistence and level == ctx_idx:
            sife_config = sife_config._replace(
                m=sife_config.m_ctx,
                k=sife_config.k_ctx,
                eta_leak=sife_config.eta_leak
            )
            
        # Interpolate context field to current level resolution
        ctx_interp = None
        if config.use_context_persistence and level != ctx_idx:
            ctx_interp = interpolate_field(
                ctx_complex, 
                ctx_complex.shape[0], 
                level_field.amplitude.shape[0]
            )
            
        # Evolve with context coupling
        evolved = evolve_field(level_field, sife_config, num_steps, context_field=ctx_interp)
        
        # Collect contribution for context field update
        if config.use_context_persistence and level != ctx_idx:
            contrib = evolved.complex_field
            contrib_up = interpolate_field(
                contrib,
                contrib.shape[0],
                ctx_complex.shape[0]
            )
            total_ctx_contribution = total_ctx_contribution + contrib_up
            
        new_amplitudes.append(evolved.amplitude)
        new_phases.append(evolved.phase)
        new_fluctuations.append(evolved.fluctuation)
        new_velocities_amp.append(evolved.velocity_amp)
        new_velocities_phi.append(evolved.velocity_phi)
        
    # Apply combined contributions to the context field
    if config.use_context_persistence:
        ctx_amp = new_amplitudes[ctx_idx] + 0.1 * jnp.abs(total_ctx_contribution)
        # Phase pressure from all scales
        ctx_phase = new_phases[ctx_idx] + 0.1 * jnp.angle(total_ctx_contribution)
        
        new_amplitudes[ctx_idx] = ctx_amp
        new_phases[ctx_idx] = ctx_phase % (2 * jnp.pi)
    
    return HierarchicalField(
        amplitudes=new_amplitudes,
        phases=new_phases,
        fluctuations=new_fluctuations,
        velocities_amp=new_velocities_amp,
        velocities_phi=new_velocities_phi
    )


class MultiScaleEncoder(nn.Module):
    """
    Multi-scale encoder that processes input at multiple resolutions.
    
    The encoder progressively downsamples the input and creates
    representations at each scale.
    """
    features: Sequence[int] = (64, 128, 256)
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action_emb: Optional[Array] = None,
        deterministic: bool = False
    ) -> List[Array]:
        """
        Args:
            x: Input complex field of shape (batch, seq_len, features)
            t: Timestep embedding
            context: Optional context
            deterministic: Whether to use dropout
        
        Returns:
            List of embeddings at each scale
        """
        from .unet import ComplexLinear, ComplexLayerNorm, ComplexModReLU
        from .unet import ComplexSelfAttention
        
        embeddings = []
        h = x
        
        for i, features in enumerate(self.features):
            # Project to current features
            h = ComplexLinear(features)(h)
            h = ComplexLayerNorm()(h)
            h = ComplexModReLU()(h)
            
            # Self-attention
            h = ComplexSelfAttention(features, self.num_heads)(h, deterministic=deterministic)
            h = ComplexLayerNorm()(h)
            
            if abs_phase is not None:
                # abs_phase is (batch,) 
                # This encoder handles spatial downsampling, so we rotate consistently
                h = h * jnp.exp(1j * abs_phase)[:, jnp.newaxis, jnp.newaxis]
            
            embeddings.append(h)
            
            # Downsample for next level (if not last)
            if i < len(self.features) - 1:
                # Average pooling in complex space
                seq_len = h.shape[1]
                new_len = seq_len // 2
                
                # Reshape and average
                h_real = jnp.real(h).reshape(h.shape[0], new_len, 2, -1).mean(axis=2)
                h_imag = jnp.imag(h).reshape(h.shape[0], new_len, 2, -1).mean(axis=2)
                h = h_real + 1j * h_imag
        
        return embeddings


class MultiScaleDecoder(nn.Module):
    """
    Multi-scale decoder that combines representations from multiple scales.
    """
    features: Sequence[int] = (256, 128, 64)
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        embeddings: List[Array],
        t: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action_emb: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        """
        Args:
            embeddings: List of embeddings from encoder
            t: Timestep embedding
            context: Optional context
            deterministic: Whether to use dropout
        
        Returns:
            Decoded complex field
        """
        from .unet import ComplexLinear, ComplexLayerNorm, ComplexModReLU
        from .unet import ComplexSelfAttention
        
        # Start from coarsest
        h = embeddings[-1]
        
        for i in range(len(embeddings) - 2, -1, -1):
            # Upsample
            target_len = embeddings[i].shape[1]
            
            # Bilinear upsampling
            h_real = jax.image.resize(
                jnp.real(h)[..., jnp.newaxis],
                (h.shape[0], target_len, h.shape[-1], 1),
                method='bilinear'
            )[..., 0]
            h_imag = jax.image.resize(
                jnp.imag(h)[..., jnp.newaxis],
                (h.shape[0], target_len, h.shape[-1], 1),
                method='bilinear'
            )[..., 0]
            h = h_real + 1j * h_imag
            
            # Process
            features = self.features[len(embeddings) - 1 - i]
            
            # Project h to the target feature dimension
            h = ComplexLinear(features)(h)
            
            # Project skip connection to same dimension
            skip = ComplexLinear(features)(embeddings[i])
            
            # Skip connection
            h = h + skip
            
            h = ComplexLayerNorm()(h)
            h = ComplexModReLU()(h)
            
            h = ComplexSelfAttention(features, self.num_heads)(h, deterministic=deterministic)
            h = ComplexLayerNorm()(h)
        
        return h


class MultiScaleSIFEUNet(nn.Module):
    """
    Multi-scale U-Net for SIFE-LDM.
    
    This architecture processes the input at multiple scales
    simultaneously, allowing the model to capture both
    fine-grained and coarse-grained patterns.
    """
    features: Sequence[int] = (64, 128, 256)
    num_heads: int = 8
    dropout_rate: float = 0.1
    output_features: int = 64
    num_experts: int = 0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        """
        Args:
            x: Input complex field
            t: Timestep
            context: Optional context
            deterministic: Whether to use dropout
        
        Returns:
            Predicted noise in complex field
        """
        from .unet import ComplexTimeEmbedding, ComplexLinear, ComplexLayerNorm
        
        # Time embedding
        t_emb = ComplexTimeEmbedding(self.features[-1])(t)
        
        # Action embedding
        action_emb = None
        if action is not None:
            action_emb = ComplexTimeEmbedding(dim=self.features[-1])(action)
        
        # Encode at multiple scales
        encoder = MultiScaleEncoder(
            features=self.features,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        embeddings = encoder(x, t_emb, context, abs_phase, action_emb, deterministic)
        
        # Process at each scale with cross-scale attention
        processed = []
        for i, emb in enumerate(embeddings):
            # Self-attention at this scale
            h = emb
            for _ in range(2):
                from .unet import ComplexTransformerBlock
                # Expand t_emb to 3D for cross-attention (batch, 1, features)
                t_emb_3d = t_emb[:, jnp.newaxis, :]
                h = ComplexTransformerBlock(
                    features=emb.shape[-1],
                    context_dim=t_emb.shape[-1],
                    num_heads=self.num_heads,
                    num_experts=self.num_experts
                )(h, t_emb_3d, deterministic=deterministic)
                
                if abs_phase is not None:
                    h = h * jnp.exp(1j * abs_phase)[:, jnp.newaxis, jnp.newaxis]
            processed.append(h)
        
        # Decode
        decoder = MultiScaleDecoder(
            features=self.features[::-1],
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )
        h = decoder(processed, t_emb, context, abs_phase, action_emb, deterministic)
        
        # Output projection
        h = ComplexLayerNorm()(h)
        h = ComplexLinear(self.output_features)(h)
        
        return h


class HierarchicalMemory:
    """
    Hierarchical memory system using multi-scale SIFE fields.
    
    Memory is stored as persistent attractors at multiple scales:
    - Fine-scale: Episodic details, specific tokens
    - Medium-scale: Concepts, relationships
    - Coarse-scale: Abstract knowledge, schemas
    """
    
    def __init__(
        self,
        config: MultiScaleConfig,
        shapes: Sequence[Tuple[int, ...]],
        key: PRNGKey
    ):
        self.config = config
        self.shapes = shapes
        
        # Initialize memory fields
        self.memory_field = initialize_hierarchical_field(
            key, shapes, config
        )
        
        # Stored attractors
        self.attractors: List[HierarchicalField] = []
    
    def store(self, field: HierarchicalField) -> None:
        """Store a field as a new attractor."""
        self.attractors.append(field)
    
    def recall(
        self,
        cue: HierarchicalField,
        num_relaxation_steps: int = 100
    ) -> HierarchicalField:
        """
        Recall from memory using a cue.
        
        The cue triggers relaxation to the nearest attractor.
        """
        if not self.attractors:
            return cue
        
        # Find nearest attractor by computing overlap at each scale
        best_distance = float('inf')
        best_attractor = self.attractors[0]
        
        for attractor in self.attractors:
            total_distance = 0.0
            for level in range(self.config.num_levels):
                # Phase distance at this level
                phase_diff = cue.phases[level] - attractor.phases[level]
                distance = jnp.mean(1 - jnp.cos(phase_diff))
                total_distance += distance
            
            if total_distance < best_distance:
                best_distance = total_distance
                best_attractor = attractor
        
        # Relax toward the best attractor
        relaxed = cue
        for _ in range(num_relaxation_steps):
            alpha = 0.1  # Relaxation rate
            new_amplitudes = [
                (1 - alpha) * relaxed.amplitudes[i] + alpha * best_attractor.amplitudes[i]
                for i in range(self.config.num_levels)
            ]
            new_phases = [
                relaxed.phases[i] + alpha * jnp.sin(
                    best_attractor.phases[i] - relaxed.phases[i]
                )
                for i in range(self.config.num_levels)
            ]
            
            relaxed = HierarchicalField(
                amplitudes=new_amplitudes,
                phases=new_phases,
                velocities_amp=relaxed.velocities_amp,
                velocities_phase=relaxed.velocities_phase
            )
        
        return relaxed
    
    def query_by_phase(
        self,
        target_phase: Array,
        level: int = 0
    ) -> List[Tuple[HierarchicalField, float]]:
        """
        Query memory by phase pattern at a specific level.
        """
        results = []
        
        for attractor in self.attractors:
            phase_similarity = jnp.mean(
                jnp.cos(attractor.phases[level] - target_phase)
            )
            results.append((attractor, float(phase_similarity)))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results


def create_multiscale_shapes(
    base_seq_len: int,
    num_levels: int,
    embed_dim: int
) -> List[Tuple[int, int]]:
    """
    Create shapes for each level of the hierarchy.
    
    Args:
        base_seq_len: Sequence length at finest scale
        num_levels: Number of hierarchy levels
        embed_dim: Embedding dimension
    
    Returns:
        List of (seq_len, embed_dim) tuples for each level
    """
    shapes = []
    seq_len = base_seq_len
    
    for level in range(num_levels):
        shapes.append((seq_len, embed_dim))
        seq_len = seq_len // 2  # Halve at each level
    
    return shapes


# Convenience function to create multi-scale config

def create_multiscale_config(
    num_levels: int = 3,
    base_features: int = 64
) -> MultiScaleConfig:
    """Create a default multi-scale configuration."""
    return MultiScaleConfig(
        num_levels=num_levels,
        base_features=base_features,
        feature_multipliers=tuple(2**i for i in range(num_levels)),
        lattice_spacings=tuple(2**i for i in range(num_levels)),
        time_scales=tuple(0.01 * 2**i for i in range(num_levels)),
        coupling_strengths=tuple(0.1 for _ in range(num_levels - 1)) + (0.05,),
        attention_levels=tuple(i >= num_levels // 2 for i in range(num_levels))
    )

