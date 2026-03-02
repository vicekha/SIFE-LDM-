"""
SIFE-LDM: Complex-Valued U-Net Architecture
============================================

Implements a complex-valued U-Net for processing the SIFE field.
The network operates on complex-valued tensors where:
- Real part represents amplitude information
- Imaginary part represents phase information

This architecture is designed to preserve the phase coherence
that is central to the SIFE framework.

Author: SIFE-LDM Research Team
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import flax.linen as nn
from flax.linen import initializers
from typing import Tuple, Optional, List, Sequence, Any, Dict
from functools import partial
import math

# Type aliases
Array = jnp.ndarray
PRNGKey = jnp.ndarray


# Utility functions (moved to top for class availability)

def complex_he_init(key: PRNGKey, shape: Tuple[int, ...]) -> Array:
    """
    He initialization adapted for complex weights.
    Returns real values scaled so that total complex variance is 2/fan_in.
    """
    fan_in = math.prod(shape[:-1])
    std = math.sqrt(1.0 / fan_in)  # Each component has 1/fan_in variance
    return jax.random.normal(key, shape) * std


def complex_xavier_init(key: PRNGKey, shape: Tuple[int, ...]) -> Array:
    """
    Xavier/Glorot initialization for complex weights.
    Returns real values scaled so that total complex variance is 2/(fan_in + fan_out).
    """
    fan_in = math.prod(shape[:-1])
    fan_out = shape[-1]
    std = math.sqrt(1.0 / (fan_in + fan_out)) # Each component has 1/(fan_in + fan_out) variance
    return jax.random.normal(key, shape) * std


class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer.
    
    Implements: y = (W_r + i*W_i) @ (x_r + i*x_i) + (b_r + i*b_i)
    
    This preserves phase relationships in the input and allows
    the network to learn rotation and scaling in the complex plane.
    """
    features: int
    use_bias: bool = True
    kernel_init: Any = complex_xavier_init
    bias_init: Any = initializers.zeros
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Complex input tensor of shape (..., in_features)
        
        Returns:
            Complex output tensor of shape (..., features)
        """
        # Get input features dimension
        in_features = x.shape[-1]
        
        # Real and imaginary weight matrices
        # Flax nn.compact handles unique key generation per param name during init
        Wr = self.param('Wr', self.kernel_init, (in_features, self.features))
        Wi = self.param('Wi', self.kernel_init, (in_features, self.features))
        
        # Complex multiplication: (Wr + i*Wi)(x_r + i*x_i)
        x_r, x_i = jnp.real(x), jnp.imag(x)
        
        # (Wr*x_r - Wi*x_i) + i(Wr*x_i + Wi*x_r)
        out_r = jnp.dot(x_r, Wr) - jnp.dot(x_i, Wi)
        out_i = jnp.dot(x_r, Wi) + jnp.dot(x_i, Wr)
        
        if self.use_bias:
            br = self.param('br', self.bias_init, (self.features,))
            bi = self.param('bi', self.bias_init, (self.features,))
            out_r = out_r + br
            out_i = out_i + bi
        
        return out_r + 1j * out_i


class ComplexConv(nn.Module):
    """
    Complex-valued convolution layer.
    
    Implements 2D convolution with complex weights, allowing
    the network to learn spatial phase transformations.
    """
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    strides: Tuple[int, int] = (1, 1)
    padding: str = 'SAME'
    use_bias: bool = True
    kernel_init: Any = initializers.lecun_normal()
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Complex input tensor of shape (H, W, in_features)
        
        Returns:
            Complex output tensor of shape (H', W', features)
        """
        in_features = x.shape[-1]
        kh, kw = self.kernel_size
        
        # Real and imaginary kernels
        # Each kernel uses lecun_normal which is real, ensuring Wr/Wi remain real
        Wr = self.param('Wr', self.kernel_init, (kh, kw, in_features, self.features))
        Wi = self.param('Wi', self.kernel_init, (kh, kw, in_features, self.features))
        
        # Separate real and imaginary parts
        x_r, x_i = jnp.real(x), jnp.imag(x)
        
        dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        
        # Convolution: (Wr * x_r - Wi * x_i) + i(Wr * x_i + Wi * x_r)
        out_r = (jax.lax.conv_general_dilated(
            x_r, Wr, window_strides=self.strides, padding=self.padding, dimension_numbers=dimension_numbers
        ) - jax.lax.conv_general_dilated(
            x_i, Wi, window_strides=self.strides, padding=self.padding, dimension_numbers=dimension_numbers
        ))
        out_i = (jax.lax.conv_general_dilated(
            x_r, Wi, window_strides=self.strides, padding=self.padding, dimension_numbers=dimension_numbers
        ) + jax.lax.conv_general_dilated(
            x_i, Wr, window_strides=self.strides, padding=self.padding, dimension_numbers=dimension_numbers
        ))
        
        if self.use_bias:
            br = self.param('br', initializers.zeros, (self.features,))
            bi = self.param('bi', initializers.zeros, (self.features,))
            out_r = out_r + br
            out_i = out_i + bi
        
        return out_r + 1j * out_i


class ComplexConv1D(nn.Module):
    """
    Complex-valued 1D convolution for sequence data.
    
    Used for NLP and coding tasks where the input is a sequence
    of tokens embedded in a complex field.
    """
    features: int
    kernel_size: int = 3
    strides: int = 1
    padding: str = 'SAME'
    use_bias: bool = True
    kernel_init: Any = initializers.lecun_normal()
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Args:
            x: Complex input tensor of shape (seq_len, in_features)
        
        Returns:
            Complex output tensor of shape (seq_len', features)
        """
        in_features = x.shape[-1]
        
        # Add batch and spatial dimensions for conv_general_dilated
        x_4d = x[jnp.newaxis, :, jnp.newaxis, :]  # (1, seq, 1, features)
        
        Wr = self.param('Wr', self.kernel_init, (self.kernel_size, 1, in_features, self.features))
        Wi = self.param('Wi', self.kernel_init, (self.kernel_size, 1, in_features, self.features))
        
        x_r, x_i = jnp.real(x_4d), jnp.imag(x_4d)
        
        # 1D convolution with kernel_size as dimension spec
        dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        
        out_r = (jax.lax.conv_general_dilated(
            x_r, Wr, window_strides=(self.strides, 1), padding=self.padding,
            dimension_numbers=dimension_numbers
        ) - jax.lax.conv_general_dilated(
            x_i, Wi, window_strides=(self.strides, 1), padding=self.padding,
            dimension_numbers=dimension_numbers
        ))
        out_i = (jax.lax.conv_general_dilated(
            x_r, Wi, window_strides=(self.strides, 1), padding=self.padding,
            dimension_numbers=dimension_numbers
        ) + jax.lax.conv_general_dilated(
            x_i, Wr, window_strides=(self.strides, 1), padding=self.padding,
            dimension_numbers=dimension_numbers
        ))
        
        if self.use_bias:
            br = self.param('br', initializers.zeros, (self.features,))
            bi = self.param('bi', initializers.zeros, (self.features,))
            out_r = out_r + br
            out_i = out_i + bi
        
        # Remove batch and spatial dimensions
        return (out_r + 1j * out_i)[0, :, 0, :]


class ComplexLayerNorm(nn.Module):
    """
    Complex-valued layer normalization.
    
    Normalizes the field energy (amplitude) while strictly preserving
    the logical phase structure (Logos).
    """
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Compute safe amplitude
        amp = jnp.sqrt(jnp.real(x)**2 + jnp.imag(x)**2 + self.epsilon)
        
        # Normalize by mean magnitude to preserve relative structure
        mean_amp = jnp.mean(amp, axis=-1, keepdims=True)
        # Note: We don't subtract the mean from the amplitude here because 
        # centered amplitudes can be negative, which flips the phase θ by π.
        # Instead, we perform a scale-only normalization.
        
        amp_norm = amp / (mean_amp + self.epsilon)
        
        # Learnable scale (Pathos intensity)
        gamma = self.param('gamma', initializers.ones, (x.shape[-1],))
        
        # Apply scaling
        amp_scaled = gamma * amp_norm
        
        # Reconstruct complex result: Ψ_norm = (A_norm * γ) * exp(i*θ)
        return amp_scaled * (x / (amp + self.epsilon))


class ComplexDropout(nn.Module):
    """
    Complex-valued dropout that applies the same mask to real and imaginary parts.
    """
    rate: float
    
    @nn.compact
    def __call__(self, x: Array, deterministic: bool = False) -> Array:
        if deterministic or self.rate == 0.0:
            return x
        
        # Apply dropout to both real and imaginary parts with same mask
        keep_prob = 1.0 - self.rate
        mask = nn.Dropout(rate=self.rate, name='dropout')(jnp.ones_like(jnp.real(x)), deterministic=False)
        
        return x * mask / keep_prob


class ComplexReLU(nn.Module):
    """
    Complex ReLU activation.
    
    Applies ReLU to the real part (modulus) while preserving phase.
    This ensures the output remains physically meaningful as a field.
    """
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Safe amplitude and phase preservation
        epsilon = 1e-10
        amplitude = jnp.sqrt(jnp.real(x)**2 + jnp.imag(x)**2 + epsilon)
        
        # Apply ReLU to amplitude (ensures non-negative)
        amplitude_relu = nn.relu(amplitude)
        
        # Reconstruct complex number: out = (x / amp) * amp_relu
        return (x / (amplitude + epsilon)) * amplitude_relu




class PhaseRouter(nn.Module):
    """
    Phase-Routed Mixture of Experts Gating Network.
    Routes tokens strictly mathematically based on their Complex Phase Angle (Logos).
    """
    num_experts: int
    
    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        # x shape: (batch, seq, features)
        
        # Calculate the phase angle (-pi to pi)
        phase = jnp.angle(x)
        
        # Calculate circular mean of phase across feature dimension for each token
        mean_cos = jnp.mean(jnp.cos(phase), axis=-1)
        mean_sin = jnp.mean(jnp.sin(phase), axis=-1)
        token_phase = jnp.arctan2(mean_sin, mean_cos)
        
        # Map Phase from (-pi, pi) to (0, 1)
        normalized_phase = (token_phase + jnp.pi) / (2 * jnp.pi)
        
        # Map to expert sectors
        expert_indices = jnp.floor(normalized_phase * self.num_experts).astype(jnp.int32)
        expert_indices = jnp.clip(expert_indices, 0, self.num_experts - 1)
        
        # Deterministic routing probability matrix
        probs = jax.nn.one_hot(expert_indices, self.num_experts)
        
        return probs, expert_indices


class ComplexMoELayer(nn.Module):
    """
    Phase-Routed Mixture of Experts layer for the SIFE complex field.
    """
    num_experts: int
    features: int
    mlp_ratio: int = 4
    num_experts_per_token: int = 1 # Kept for API compatibility
    
    @nn.compact
    def __call__(self, x: Array, deterministic: bool = False) -> Array:
        # 1. Physical Phase-based Gate
        probs, indices = PhaseRouter(
            num_experts=self.num_experts
        )(x)
        
        out = jnp.zeros_like(x)
        
        # 2. Execute sparse experts
        for i in range(self.num_experts):
            # Create the expert MLP
            expert_mlp = lambda inp: ComplexLinear(self.features)(
                ComplexModReLU()(
                    ComplexLinear(self.features * self.mlp_ratio)(inp)
                )
            )
            
            # Hard routing mask
            mask = (indices == i)
            expert_weight = probs[..., i:i+1]
            
            # Apply expert only to relevant tokens
            expert_out = expert_mlp(x)
            out = out + expert_out * expert_weight * mask[..., jnp.newaxis]
            
        return out


class ComplexGELU(nn.Module):
    """
    Complex GELU activation.
    
    Applies GELU to amplitude while preserving phase structure.
    """
    @nn.compact
    def __call__(self, x: Array) -> Array:
        epsilon = 1e-10
        amplitude = jnp.sqrt(jnp.real(x)**2 + jnp.imag(x)**2 + epsilon)
        
        # GELU on amplitude
        amplitude_gelu = nn.gelu(amplitude)
        
        return amplitude_gelu * (x / (amplitude + epsilon))


class ComplexModReLU(nn.Module):
    """
    ModReLU: Complex activation that applies ReLU with a learnable bias on the modulus.
    
    From "Deep Complex Networks" (Trabelsi et al., 2018):
    modReLU(z) = ReLU(|z| + b) * z / |z|
    
    This allows the network to learn which amplitudes to suppress.
    """
    @nn.compact
    def __call__(self, x: Array) -> Array:
        epsilon = 1e-10
        amplitude = jnp.sqrt(jnp.real(x)**2 + jnp.imag(x)**2 + epsilon)
        
        # Learnable bias for amplitude threshold
        b = self.param('bias', initializers.zeros, (x.shape[-1],))
        
        # ReLU with bias on amplitude
        amplitude_out = nn.relu(amplitude + b)
        
        return amplitude_out * (x / (amplitude + epsilon))


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timesteps.
    
    Encodes the timestep t into a high-dimensional representation
    that can be added to the field or used for conditioning.
    """
    dim: int
    
    @nn.compact
    def __call__(self, t: Array) -> Array:
        """
        Args:
            t: Timestep of shape (batch,) or scalar
        
        Returns:
            Embedding of shape (batch, dim) or (dim,)
        """
        half_dim = self.dim // 2
        emb_scale = jnp.log(10000.0) / (half_dim - 1)
        emb = jnp.arange(half_dim, dtype=jnp.float32)
        emb = jnp.exp(emb * (-emb_scale))
        
        # Ensure t is at least 1D
        t = jnp.atleast_1d(t).astype(jnp.float32)
        
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        return emb


class ComplexTimeEmbedding(nn.Module):
    """
    Complex-valued time embedding for conditioning on diffusion timesteps.
    
    Generates both amplitude and phase components for the embedding.
    """
    dim: int
    
    @nn.compact
    def __call__(self, t: Array) -> Array:
        # Get base sinusoidal embedding
        emb = SinusoidalPositionEmbedding(self.dim)(t)
        
        # Split into amplitude and phase components
        half_dim = self.dim // 2
        amp_part = emb[:, :half_dim]
        phase_part = emb[:, half_dim:]
        
        # Amplitude: positive values from sin/cos
        amplitude = nn.Dense(self.dim)(amp_part)
        amplitude = nn.relu(amplitude) + 0.1  # Ensure positive
        
        # Phase: normalize to [0, 2π)
        phase = nn.Dense(self.dim)(phase_part)
        phase = 2 * jnp.pi * jnp.tanh(phase)
        
        return amplitude * jnp.exp(1j * phase)


class ComplexResidualBlock(nn.Module):
    """
    Complex-valued residual block with time embedding.
    
    The time embedding is added to the field before the second convolution,
    allowing the network to modulate its behavior based on the diffusion timestep.
    """
    features: int
    kernel_size: int = 3
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: Array, t_emb: Array, abs_phase: Optional[Array] = None, 
                 action_emb: Optional[Array] = None, deterministic: bool = False) -> Array:
        # First convolution
        h = ComplexConv1D(features=self.features, kernel_size=self.kernel_size)(x)
        h = ComplexLayerNorm()(h)
        h = ComplexModReLU()(h)
        
        # Time embedding projection
        t_proj = ComplexLinear(self.features)(t_emb)
        
        # Add time embedding and global phase rotation
        h = h + t_proj
        
        # Action embedding projection
        if action_emb is not None:
            a_proj = ComplexLinear(self.features)(action_emb)
            h = h + a_proj
        
        if abs_phase is not None:
            # abs_phase provides a global temporal reference
            # conditioning via global rotation
            h = h * jnp.exp(1j * abs_phase)[:, jnp.newaxis, jnp.newaxis]
            
        # Dropout
        h = ComplexDropout(self.dropout_rate)(h, deterministic)
        
        # Second convolution
        h = ComplexConv1D(features=self.features, kernel_size=self.kernel_size)(h)
        h = ComplexLayerNorm()(h)
        
        # Residual connection with projection if needed
        if x.shape[-1] != self.features:
            x = ComplexLinear(self.features)(x)
        
        return ComplexModReLU()(h + x)


class ComplexSelfAttention(nn.Module):
    """
    Complex-valued self-attention mechanism.
    
    Computes attention in the complex domain, allowing the model to
    learn phase relationships between different positions in the sequence.
    
    The attention scores are computed as:
    attention(Q, K, V) = softmax(Re(Q* K^H) / sqrt(d)) * V
    
    Where * denotes complex conjugate and ^H denotes Hermitian transpose.
    """
    features: int
    num_heads: int = 8
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(self, x: Array, mask: Optional[Array] = None, deterministic: bool = False) -> Array:
        batch_size, seq_len, _ = x.shape
        head_dim = self.features // self.num_heads
        
        # Project to Q, K, V with complex linear layers
        Q = ComplexLinear(self.features)(x)
        K = ComplexLinear(self.features)(x)
        V = ComplexLinear(self.features)(x)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        K = K.reshape(batch_size, seq_len, self.num_heads, head_dim)
        V = V.reshape(batch_size, seq_len, self.num_heads, head_dim)
        
        # Transpose to (batch, num_heads, seq_len, head_dim)
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))
        
        # Compute attention scores using Hermitian inner product
        # score = Re(Q * K^H) / sqrt(d)
        # Re( (Qr + iQi) * (Kr - iKi)^T ) = Qr*Kr^T + Qi*Ki^T
        scale = 1.0 / jnp.sqrt(head_dim)
        
        Q_r, Q_i = jnp.real(Q), jnp.imag(Q)
        K_r, K_i = jnp.real(K), jnp.imag(K)
        
        # Directly compute real part of inner product to save memory
        scores = (jnp.einsum('bnqd,bnkd->bnqk', Q_r, K_r) + 
                  jnp.einsum('bnqd,bnkd->bnqk', Q_i, K_i)) * scale
        
        # Apply mask if provided
        if mask is not None:
            # mask has shape (batch, seq_len)
            scores = scores + (1.0 - mask[:, None, None, :]) * (-1e9)
        
        # Softmax
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        out = jnp.einsum('bnqk,bnkd->bnqd', attn_weights, V)
        
        # Reshape back
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch_size, seq_len, self.features)
        
        # Output projection
        out = ComplexLinear(self.features)(out)
        
        return out


class ComplexCrossAttention(nn.Module):
    """
    Complex-valued cross-attention for conditioning.
    
    Allows the model to attend to a context (e.g., conversation history,
    code context) while processing the main sequence.
    """
    features: int
    context_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        context: Array,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        batch_size, seq_len, _ = x.shape
        _, context_len, _ = context.shape
        head_dim = self.features // self.num_heads
        
        # Q from main sequence, K, V from context
        Q = ComplexLinear(self.features)(x)
        K = ComplexLinear(self.features)(context)
        V = ComplexLinear(self.features)(context)
        
        # Reshape for multi-head attention
        Q = Q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        K = K.reshape(batch_size, context_len, self.num_heads, head_dim)
        V = V.reshape(batch_size, context_len, self.num_heads, head_dim)
        
        # Transpose
        Q = jnp.transpose(Q, (0, 2, 1, 3))
        K = jnp.transpose(K, (0, 2, 1, 3))
        V = jnp.transpose(V, (0, 2, 1, 3))
        
        # Attention scores
        # Re( (Qr + iQi) * (Kr - iKi)^T ) = Qr*Kr^T + Qi*Ki^T
        Q_r, Q_i = jnp.real(Q), jnp.imag(Q)
        K_r, K_i = jnp.real(K), jnp.imag(K)
        
        scores = (jnp.einsum('bnqd,bnkd->bnqk', Q_r, K_r) + 
                  jnp.einsum('bnqd,bnkd->bnqk', Q_i, K_i)) / jnp.sqrt(head_dim)
        
        if mask is not None:
            scores = scores + (1.0 - mask[:, None, None, :]) * (-1e9)
        
        attn_weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum('bnqk,bnkd->bnqd', attn_weights, V)
        
        # Reshape back
        out = jnp.transpose(out, (0, 2, 1, 3))
        out = out.reshape(batch_size, seq_len, self.features)
        
        return ComplexLinear(self.features)(out)


class ComplexTransformerBlock(nn.Module):
    """
    Complex-valued Transformer block with self-attention and cross-attention.
    """
    features: int
    context_dim: int
    num_heads: int = 8
    dropout_rate: float = 0.1
    mlp_ratio: int = 4
    num_experts: int = 0  # 0 means don't use MoE
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        context: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        # Self-attention
        h = ComplexLayerNorm()(x)
        h = ComplexSelfAttention(
            features=self.features,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate
        )(h, mask, deterministic)
        x = x + h
        
        # Cross-attention (if context provided)
        if context is not None:
            h = ComplexLayerNorm()(x)
            h = ComplexCrossAttention(
                features=self.features,
                context_dim=self.context_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate
            )(h, context, mask, deterministic)
            x = x + h
        
        # MLP or MoE
        h = ComplexLayerNorm()(x)
        if self.num_experts > 0:
            h = ComplexMoELayer(
                num_experts=self.num_experts,
                features=self.features,
                mlp_ratio=self.mlp_ratio
            )(h, deterministic)
        else:
            h = ComplexLinear(self.features * self.mlp_ratio)(h)
            h = ComplexModReLU()(h)
            h = ComplexDropout(self.dropout_rate)(h, deterministic)
            h = ComplexLinear(self.features)(h)
        x = x + h
        
        return x


class ComplexDownBlock(nn.Module):
    """
    Downsampling block for the U-Net encoder path.
    
    Applies residual blocks followed by optional attention,
    then downsamples the sequence.
    """
    features: int
    num_layers: int = 2
    kernel_size: int = 3
    attention: bool = False
    dropout_rate: float = 0.1
    num_experts: int = 0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        t_emb: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action_emb: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Tuple[Array, Array]:
        """
        Args:
            x: Input field of shape (batch, seq_len, features)
            t_emb: Time embedding
            context: Optional conditioning context
            mask: Attention mask
        
        Returns:
            Tuple of (output, skip_connection for decoder)
        """
        for i in range(self.num_layers):
            x = ComplexResidualBlock(
                features=self.features,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate
            )(x, t_emb, abs_phase, action_emb, deterministic)
            
            if self.attention:
                x = ComplexTransformerBlock(
                    features=self.features,
                    context_dim=self.features if context is not None else self.features,
                    dropout_rate=self.dropout_rate,
                    num_experts=self.num_experts
                )(x, context, mask, deterministic)
        
        # Downsample by strided convolution or average pooling
        skip = x
        x = ComplexConv1D(features=self.features, kernel_size=3, strides=2)(x)
        
        return x, skip


class ComplexUpBlock(nn.Module):
    """
    Upsampling block for the U-Net decoder path.
    
    Upsamples the sequence and concatenates with skip connection,
    then applies residual blocks.
    """
    features: int
    num_layers: int = 2
    kernel_size: int = 3
    attention: bool = False
    dropout_rate: float = 0.1
    num_experts: int = 0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        skip: Array,
        t_emb: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action_emb: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        # Upsample
        seq_len = skip.shape[1]
        x = jax.image.resize(
            jnp.stack([jnp.real(x), jnp.imag(x)], axis=-1),
            (x.shape[0], seq_len, x.shape[2], 2),
            method='bilinear'
        )
        x = x[..., 0] + 1j * x[..., 1]
        
        # Concatenate with skip connection
        x = jnp.concatenate([x, skip], axis=-1)
        
        # Project back to desired features
        x = ComplexLinear(self.features)(x)
        
        for i in range(self.num_layers):
            x = ComplexResidualBlock(
                features=self.features,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate
            )(x, t_emb, abs_phase, action_emb, deterministic)
            
            if self.attention:
                x = ComplexTransformerBlock(
                    features=self.features,
                    context_dim=self.features if context is not None else self.features,
                    dropout_rate=self.dropout_rate,
                    num_experts=self.num_experts
                )(x, context, mask, deterministic)
        
        return x


class PositionalPhaseEncoding(nn.Module):
    """Injects positional information via phase rotation."""
    @nn.compact
    def __call__(self, x: Array) -> Array:
        seq_len = x.shape[1]
        features = x.shape[2]
        pos = jnp.arange(seq_len)[jnp.newaxis, :, jnp.newaxis]
        # Initialize omega to small values to start with smooth rotations
        omega = self.param('omega', jax.nn.initializers.normal(stddev=0.02), (1, 1, features))
        theta = pos * omega
        return x * jnp.exp(1j * theta)


class ComplexUNet1D(nn.Module):
    """
    Complex-valued 1D U-Net for sequence modeling.
    
    This is the core architecture for SIFE-LDM, processing
    sequences embedded in the complex field. The encoder-decoder
    structure allows the model to capture multi-scale patterns
    while preserving phase coherence.
    
    Architecture:
    - Encoder: Progressive downsampling with residual blocks
    - Bottleneck: Full attention over the entire sequence
    - Decoder: Progressive upsampling with skip connections
    
    The model is conditioned on:
    - Diffusion timestep t
    - Optional context (for conditional generation)
    """
    features: int = 256
    num_down_layers: int = 2
    num_up_layers: int = 2
    num_blocks: int = 4
    attention_levels: Sequence[bool] = (False, False, True, True)
    num_heads: int = 8
    dropout_rate: float = 0.1
    context_dim: Optional[int] = None
    num_experts: int = 0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        """
        Args:
            x: Input complex field of shape (batch, seq_len, in_features)
            t: Diffusion timestep of shape (batch,)
            context: Optional conditioning context of shape (batch, ctx_len, context_dim)
            mask: Attention mask for context
            deterministic: Whether to use deterministic mode (no dropout)
        
        Returns:
            Predicted noise in the complex field of shape (batch, seq_len, features)
        """
        # Project input to feature dimension
        h = ComplexLinear(self.features)(x)
        
        # Apply Positional Phase Encoding
        h = PositionalPhaseEncoding()(h)
        
        # Time embedding
        t_emb = ComplexTimeEmbedding(self.features)(t)
        
        # Action embedding
        action_emb = None
        if action is not None:
            action_emb = ComplexTimeEmbedding(dim=self.features)(action)
        
        # Store skip connections
        skips = []
        
        # Encoder path
        for i in range(self.num_blocks):
            features = self.features * (2 ** i)
            attention = self.attention_levels[i] if i < len(self.attention_levels) else True
            
            h, skip = ComplexDownBlock(
                features=features,
                num_layers=self.num_down_layers,
                attention=attention,
                dropout_rate=self.dropout_rate,
                num_experts=self.num_experts
            )(h, t_emb, context, abs_phase, action_emb, mask, deterministic)
            
            skips.append(skip)
        
        # Bottleneck with full attention
        h = ComplexTransformerBlock(
            features=h.shape[-1],
            context_dim=self.context_dim or self.features,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            num_experts=self.num_experts
        )(h, context, mask, deterministic)
        
        # Decoder path
        for i in reversed(range(self.num_blocks)):
            features = self.features * (2 ** i)
            attention = self.attention_levels[i] if i < len(self.attention_levels) else True
            
            h = ComplexUpBlock(
                features=features,
                num_layers=self.num_up_layers,
                attention=attention,
                dropout_rate=self.dropout_rate,
                num_experts=self.num_experts
            )(h, skips[i], t_emb, context, abs_phase, action_emb, mask, deterministic)
        
        # Output projection to predict noise
        h = ComplexLayerNorm()(h)
        h = ComplexLinear(self.features)(h)
        h = ComplexModReLU()(h)
        h = ComplexLinear(x.shape[-1])(h)
        
        return h


class SIFEUNet(nn.Module):
    """
    SIFE-specific U-Net that operates directly on amplitude and phase.
    
    This variant processes the amplitude and phase separately before
    combining them, which may better preserve the physical interpretation
    of the SIFE field.
    """
    features: int = 256
    num_blocks: int = 4
    num_heads: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(
        self,
        amplitude: Array,
        phase: Array,
        t: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        deterministic: bool = False
    ) -> Tuple[Array, Array]:
        """
        Args:
            amplitude: Amplitude field of shape (batch, seq_len, features)
            phase: Phase field of shape (batch, seq_len, features)
            t: Timestep
            context: Optional context
        
        Returns:
            Tuple of (predicted amplitude noise, predicted phase noise)
        """
        batch_size, seq_len, _ = amplitude.shape
        
        # Process amplitude with standard real-valued network
        amp_emb = nn.Dense(self.features)(amplitude)
        
        # Process phase with periodic-aware embedding
        phase_sin = jnp.sin(phase)
        phase_cos = jnp.cos(phase)
        phase_emb = jnp.concatenate([phase_sin, phase_cos], axis=-1)
        phase_emb = nn.Dense(self.features)(phase_emb)
        
        # Combine amplitude and phase information
        h = amp_emb + phase_emb
        
        # Time embedding
        t_emb = SinusoidalPositionEmbedding(self.features)(t)
        t_emb = nn.Dense(self.features)(t_emb)
        t_emb = nn.gelu(t_emb)
        t_emb = nn.Dense(self.features)(t_emb)
        
        # Add time embedding and global clock
        h = h + t_emb
        
        if abs_phase is not None:
            # Condition via global phase shift
            h_r, h_i = h * jnp.cos(abs_phase)[:, jnp.newaxis, jnp.newaxis], h * jnp.sin(abs_phase)[:, jnp.newaxis, jnp.newaxis]
            h = h + h_r + h_i # This is a simplified real-valued injection for SIFEUNet
        
        # Apply transformer blocks
        for _ in range(self.num_blocks):
            # Self-attention
            h_norm = nn.LayerNorm()(h)
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.features,
            )(h_norm, h_norm)
            h = h + attn_out
            
            # MLP
            h_norm = nn.LayerNorm()(h)
            mlp_out = nn.Dense(self.features * 4)(h_norm)
            mlp_out = nn.gelu(mlp_out)
            mlp_out = nn.Dense(self.features)(mlp_out)
            h = h + mlp_out
        
        # Separate output heads for amplitude and phase
        # Amplitude: predict noise (can be positive or negative)
        amp_noise = nn.Dense(amplitude.shape[-1])(h)
        
        # Phase: predict noise (unconstrained)
        phase_noise = nn.Dense(phase.shape[-1])(h)
        
        return amp_noise, phase_noise


# Additional utility functions

class ComplexResidualBlock2D(nn.Module):
    """
    Complex-valued residual block for 2D fields.
    """
    features: int
    kernel_size: Tuple[int, int] = (3, 3)
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x: Array, t_emb: Array, abs_phase: Optional[Array] = None, 
                 action_emb: Optional[Array] = None, deterministic: bool = False) -> Array:
        # First convolution
        h = ComplexConv(features=self.features, kernel_size=self.kernel_size)(x)
        h = ComplexLayerNorm()(h)
        h = ComplexModReLU()(h)
        
        # Time and action embedding projections
        t_proj = ComplexLinear(self.features)(t_emb)
        h = h + t_proj[:, jnp.newaxis, jnp.newaxis, :]
        
        if action_emb is not None:
            a_proj = ComplexLinear(self.features)(action_emb)
            h = h + a_proj[:, jnp.newaxis, jnp.newaxis, :]
            
        if abs_phase is not None:
            h = h * jnp.exp(1j * abs_phase)[:, jnp.newaxis, jnp.newaxis, jnp.newaxis]
            
        # Dropout
        h = ComplexDropout(self.dropout_rate)(h, deterministic)
        
        # Second convolution
        h = ComplexConv(features=self.features, kernel_size=self.kernel_size)(h)
        h = ComplexLayerNorm()(h)
        
        # Residual connection
        if x.shape[-1] != self.features:
            x = ComplexLinear(self.features)(x)
            
        return ComplexModReLU()(h + x)


class ComplexDownBlock2D(nn.Module):
    """
    Downsampling block for 2D complex fields.
    """
    features: int
    num_layers: int = 2
    kernel_size: Tuple[int, int] = (3, 3)
    attention: bool = False
    dropout_rate: float = 0.1
    num_experts: int = 0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        t_emb: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action_emb: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Tuple[Array, Array]:
        for i in range(self.num_layers):
            x = ComplexResidualBlock2D(
                features=self.features,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate
            )(x, t_emb, abs_phase, action_emb, deterministic)
            
            if self.attention:
                # Flatten (H, W) for attention
                B, H, W, C = x.shape
                h = x.reshape(B, H * W, C)
                h = ComplexTransformerBlock(
                    features=self.features,
                    context_dim=self.features if context is not None else self.features,
                    dropout_rate=self.dropout_rate,
                    num_experts=self.num_experts
                )(h, context, mask, deterministic)
                x = h.reshape(B, H, W, C)
        
        # Downsample
        skip = x
        x = ComplexConv(features=self.features, kernel_size=(3, 3), strides=(2, 2))(x)
        
        return x, skip


class ComplexUpBlock2D(nn.Module):
    """
    Upsampling block for 2D complex fields.
    """
    features: int
    num_layers: int = 2
    kernel_size: Tuple[int, int] = (3, 3)
    attention: bool = False
    dropout_rate: float = 0.1
    num_experts: int = 0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        skip: Array,
        t_emb: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action_emb: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        # Upsample
        h, w = skip.shape[1], skip.shape[2]
        # x shape: (B, h_small, w_small, C)
        # We need to resize to (h, w)
        # Using standard resize but preserving complex info
        x_r, x_i = jnp.real(x), jnp.imag(x)
        x_r = jax.image.resize(x_r, (x.shape[0], h, w, x.shape[3]), method='bilinear')
        x_i = jax.image.resize(x_i, (x.shape[0], h, w, x.shape[3]), method='bilinear')
        x = x_r + 1j * x_i
        
        # Concatenate with skip connection
        x = jnp.concatenate([x, skip], axis=-1)
        x = ComplexLinear(self.features)(x)
        
        for i in range(self.num_layers):
            x = ComplexResidualBlock2D(
                features=self.features,
                kernel_size=self.kernel_size,
                dropout_rate=self.dropout_rate
            )(x, t_emb, abs_phase, action_emb, deterministic)
            
            if self.attention:
                B, H, W, C = x.shape
                h_flattened = x.reshape(B, H * W, C)
                h_flattened = ComplexTransformerBlock(
                    features=self.features,
                    context_dim=self.features if context is not None else self.features,
                    dropout_rate=self.dropout_rate,
                    num_experts=self.num_experts
                )(h_flattened, context, mask, deterministic)
                x = h_flattened.reshape(B, H, W, C)
        
        return x


class ComplexUNet2D(nn.Module):
    """
    Complex-valued 2D U-Net for image/lattice modeling.
    """
    features: int = 128
    num_down_layers: int = 2
    num_up_layers: int = 2
    num_blocks: int = 3
    attention_levels: Sequence[bool] = (False, True, True)
    num_heads: int = 8
    dropout_rate: float = 0.1
    context_dim: Optional[int] = None
    num_experts: int = 0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        # Time and action embeddings
        t_emb = ComplexTimeEmbedding(self.features)(t)
        action_emb = None
        if action is not None:
            action_emb = ComplexTimeEmbedding(dim=self.features)(action)
            
        # Initial projection
        h = ComplexConv(features=self.features, kernel_size=(3, 3))(x)
        
        # Store skip connections
        skips = []
        
        # Encoder path
        for i in range(self.num_blocks):
            feat = self.features * (2 ** i)
            attn = self.attention_levels[i] if i < len(self.attention_levels) else True
            
            h, skip = ComplexDownBlock2D(
                features=feat,
                num_layers=self.num_down_layers,
                attention=attn,
                dropout_rate=self.dropout_rate,
                num_experts=self.num_experts
            )(h, t_emb, context, abs_phase, action_emb, mask, deterministic)
            skips.append(skip)
            
        # Bottleneck
        B, H, W, C = h.shape
        h_flat = h.reshape(B, H * W, C)
        h_flat = ComplexTransformerBlock(
            features=C,
            context_dim=self.context_dim or self.features,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
            num_experts=self.num_experts
        )(h_flat, context, mask, deterministic)
        h = h_flat.reshape(B, H, W, C)
        
        # Decoder path
        for i in reversed(range(self.num_blocks)):
            feat = self.features * (2 ** i)
            attn = self.attention_levels[i] if i < len(self.attention_levels) else True
            
            h = ComplexUpBlock2D(
                features=feat,
                num_layers=self.num_up_layers,
                attention=attn,
                dropout_rate=self.dropout_rate,
                num_experts=self.num_experts
            )(h, skips[i], t_emb, context, abs_phase, action_emb, mask, deterministic)
            
        # Output projection
        h = ComplexLayerNorm()(h)
        h = ComplexConv(features=x.shape[-1], kernel_size=(3, 3))(h)
        
        return h


class ComplexPatchEncoder(nn.Module):
    """
    Slices a 2D complex image (B, H, W, C) into non-overlapping patches
    and flattens them into a 1D sequence of complex tokens (B, seq_len, embed_dim).
    """
    patch_size: int = 4
    embed_dim: int = 128
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # x shape: (B, H, W, C)
        B, H, W, C = x.shape
        P = self.patch_size
        
        # Ensure divisible
        assert H % P == 0 and W % P == 0, f"Image size ({H}, {W}) must be divisible by patch size {P}"
        
        # Reshape into patches: (B, H/P, P, W/P, P, C)
        x = x.reshape(B, H // P, P, W // P, P, C)
        
        # Transpose to group patches: (B, H/P, W/P, P, P, C)
        x = jnp.transpose(x, (0, 1, 3, 2, 4, 5))
        
        # Flatten patches into sequence tokens: (B, (H/P)*(W/P), P*P*C)
        seq_len = (H // P) * (W // P)
        x = x.reshape(B, seq_len, P * P * C)
        
        # Project each patch to embed_dim using ComplexLinear
        x = ComplexLinear(self.embed_dim)(x)
        
        # 2D Rotary Positional Encoding (RoPE) injected into Phase
        # We split the embed_dim into two halves: row features and col features
        half_dim = self.embed_dim // 2
        
        row_pos = jnp.arange(H // P)
        col_pos = jnp.arange(W // P)
        
        # Frequencies (using math.log for scalar compilation)
        freqs = jnp.exp(-jnp.arange(0, half_dim, 2) * (math.log(10000.0) / half_dim))
        
        # Row Phase (H/P, half_dim)
        row_angles = row_pos[:, jnp.newaxis] * freqs[jnp.newaxis, :]
        row_angles = jnp.repeat(row_angles, 2, axis=-1)
        
        # Col Phase (W/P, half_dim)
        col_angles = col_pos[:, jnp.newaxis] * freqs[jnp.newaxis, :]
        col_angles = jnp.repeat(col_angles, 2, axis=-1)
        
        # Combine into (H/P, W/P, embed_dim)
        row_angles_mesh = jnp.broadcast_to(row_angles[:, jnp.newaxis, :], (H // P, W // P, half_dim))
        col_angles_mesh = jnp.broadcast_to(col_angles[jnp.newaxis, :, :], (H // P, W // P, half_dim))
        
        grid_angles = jnp.concatenate([row_angles_mesh, col_angles_mesh], axis=-1)
        grid_angles = grid_angles.reshape(seq_len, self.embed_dim)
        
        # Inject RoPE via complex rotation
        x = x * jnp.exp(1j * grid_angles)[jnp.newaxis, :, :]      
        
        return x


class PhasePool(nn.Module):
    """
    Phase-Coherence Token Merging (Pooling).
    Reduces sequence length from N to k by clustering tokens with similar SIFE Phases (Logos).
    Returns pooled tokens and the soft assignment matrix A used for unpooling.
    """
    k: int
    
    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        # x is (B, N, d)
        phase = jnp.angle(x)
        
        # Combine into phase features (circular mean)
        mean_cos = jnp.mean(jnp.cos(phase), axis=-1, keepdims=True)
        mean_sin = jnp.mean(jnp.sin(phase), axis=-1, keepdims=True)
        phase_features = jnp.concatenate([mean_cos, mean_sin], axis=-1)
        
        # Linear layer to predict assignment logits to k clusters
        logits = nn.Dense(self.k)(phase_features)
        A = jax.nn.softmax(logits, axis=-1) # (B, N, k)
        
        # Pool to k tokens: A^T @ x -> (B, k, d)
        x_pooled = jnp.einsum('bnk,bnd->bkd', A, x)
        
        # Normalize by cluster size
        cluster_sizes = jnp.sum(A, axis=1, keepdims=True) # (B, 1, k)
        cluster_sizes = jnp.clip(cluster_sizes, 1e-5, None)
        x_pooled = x_pooled / cluster_sizes.transpose(0, 2, 1)
        
        return x_pooled, A


class PhaseUnpool(nn.Module):
    """
    Phase-Coherence Token Unmerging.
    Upsamples from k tokens back to N using the soft assignment matrix A.
    """
    @nn.compact
    def __call__(self, x_pooled: Array, A: Array) -> Array:
        # x_pooled is (B, k, d), A is (B, N, k)
        x_unpooled = jnp.einsum('bnk,bkd->bnd', A, x_pooled)
        return x_unpooled


class UnifiedSIFETransformer(nn.Module):
    """
    A unified complex-valued Diffusion Transformer (DiT) architecture
    that treats all data (text, image patches, audio frames) as a 1D
    sequence of complex potentials.
    """
    features: int = 256
    depth: int = 12
    num_heads: int = 8
    dropout_rate: float = 0.1
    context_dim: Optional[int] = None
    mlp_ratio: int = 4
    num_experts: int = 0
    
    @nn.compact
    def __call__(
        self,
        x: Array,
        t: Array,
        context: Optional[Array] = None,
        abs_phase: Optional[Array] = None,
        action: Optional[Array] = None,
        mask: Optional[Array] = None,
        deterministic: bool = False
    ) -> Array:
        # x shape: (B, seq_len, in_features)
        
        # Time embedding
        t_emb = ComplexTimeEmbedding(self.features)(t)
        
        # Project input feature dimension if necessary
        h = ComplexLinear(self.features)(x)
        
        # We need to inject the time embedding (and action/abs_phase) into the sequence
        # We broadcast the time embedding across the sequence length
        t_proj = ComplexLinear(self.features)(t_emb)
        h = h + t_proj[:, jnp.newaxis, :]
        
        if action is not None:
            action_emb = ComplexTimeEmbedding(dim=self.features)(action)
            a_proj = ComplexLinear(self.features)(action_emb)
            h = h + a_proj[:, jnp.newaxis, :]
            
        if abs_phase is not None:
            h = h * jnp.exp(1j * abs_phase)[:, jnp.newaxis, jnp.newaxis]
            
        # Intermediate sequence length halving for dense efficiency (Phase-Coherent Token Merging)
        do_pooling = (self.depth >= 6)
        mid_depth = self.depth // 2
        A_matrices = []
        
        # Transformer Blocks
        for i in range(self.depth):
            
            # Pool halfway down
            if do_pooling and i == self.depth // 4:
                # Reduce sequence length by 50%
                k = max(1, h.shape[1] // 2)
                h, A = PhasePool(k=k)(h)
                A_matrices.append(A)
                
            h = ComplexTransformerBlock(
                features=self.features,
                context_dim=self.context_dim or self.features,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                mlp_ratio=self.mlp_ratio,
                num_experts=self.num_experts
            )(h, context, mask, deterministic)
            
            # Unpool halfway up
            if do_pooling and i == self.depth - (self.depth // 4):
                A = A_matrices.pop()
                h = PhaseUnpool()(h, A)
            
        # Final LayerNorm & projection to original features
        h = ComplexLayerNorm()(h)
        out = ComplexLinear(x.shape[-1])(h)
        
        return out
