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


class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer.
    
    Implements: y = (W_r + i*W_i) @ (x_r + i*x_i) + (b_r + i*b_i)
    
    This preserves phase relationships in the input and allows
    the network to learn rotation and scaling in the complex plane.
    """
    features: int
    use_bias: bool = True
    kernel_init: Any = initializers.lecun_normal()
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
    
    Normalizes both amplitude and phase independently while
    preserving the complex structure of the data.
    """
    epsilon: float = 1e-6
    
    @nn.compact
    def __call__(self, x: Array) -> Array:
        # Safe amplitude
        amp = jnp.sqrt(jnp.real(x)**2 + jnp.imag(x)**2 + self.epsilon)
        amp_mean = jnp.mean(amp, axis=-1, keepdims=True)
        amp_var = jnp.var(amp, axis=-1, keepdims=True)
        amp_norm = (amp - amp_mean) / jnp.sqrt(amp_var + self.epsilon)
        
        # Learnable scale and shift
        gamma = self.param('gamma', initializers.ones, (x.shape[-1],))
        beta = self.param('beta', initializers.zeros, (x.shape[-1],))
        
        # Reconstruct complex result stably: out = gamma * (x / (amp + eps)) * amp_norm + beta
        # This preserves phase coherence while avoiding NaNs at zero
        return gamma * amp_norm * (x / (amp + self.epsilon)) + beta


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




class ComplexGatingNetwork(nn.Module):
    """
    Computes routing weights for Mixture of Experts from complex input.
    """
    num_experts: int
    num_experts_per_token: int = 1
    
    @nn.compact
    def __call__(self, x: Array) -> Tuple[Array, Array]:
        # x is (batch, seq, features)
        # Use both amplitude and phase for routing
        amp = jnp.real(jnp.abs(x))
        phase = jnp.angle(x)
        
        # Combine into routing features
        routing_input = jnp.concatenate([amp, phase], axis=-1)
        
        # Standard real-valued linear layer for routing scores
        scores = nn.Dense(self.num_experts)(routing_input)
        
        # Softmax to get routing probabilities
        probs = jax.nn.softmax(scores, axis=-1)
        
        # Select top-k experts
        if self.num_experts_per_token == 1:
            expert_indices = jnp.argmax(probs, axis=-1)
            # Mask out non-selected experts (optional for sparse MoE)
            return probs, expert_indices
        else:
            # Multi-expert routing (Top-K)
            top_k_probs, top_k_indices = jax.lax.top_k(probs, self.num_experts_per_token)
            # Re-normalize top-k
            top_k_probs = top_k_probs / jnp.sum(top_k_probs, axis=-1, keepdims=True)
            return top_k_probs, top_k_indices


class ComplexMoELayer(nn.Module):
    """
    Mixture of Experts layer for the SIFE complex field.
    """
    num_experts: int
    features: int
    mlp_ratio: int = 4
    num_experts_per_token: int = 1
    
    @nn.compact
    def __call__(self, x: Array, deterministic: bool = False) -> Array:
        # 1. Gate
        probs, indices = ComplexGatingNetwork(
            num_experts=self.num_experts,
            num_experts_per_token=self.num_experts_per_token
        )(x)
        
        # 2. Sequential/Vmap Expert Execution (Simplified for now)
        # In a real MoE, we'd use sparse operations or vmap over experts
        # Here we'll implement a functional routing for JAX compatibility
        
        out = jnp.zeros_like(x)
        
        # For each expert, process the tokens assigned to it
        for i in range(self.num_experts):
            # Create the expert MLP
            expert_mlp = lambda inp: ComplexLinear(self.features)(
                ComplexModReLU()(
                    ComplexLinear(self.features * self.mlp_ratio)(inp)
                )
            )
            
            # Mask for tokens routed to this expert
            if self.num_experts_per_token == 1:
                mask = (indices == i)
                expert_weight = probs[..., i:i+1] # Probability for this expert
            else:
                # Multi-expert masking
                mask = jnp.any(indices == i, axis=-1)
                # Find the probability weight for this expert in the top-k set
                # This is a bit complex for a loop, so we'll simplify:
                expert_weight = jnp.sum(jnp.where(indices == i, probs, 0.0), axis=-1, keepdims=True)
            
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


# Utility functions

def complex_he_init(key: PRNGKey, shape: Tuple[int, ...]) -> Array:
    """
    He initialization adapted for complex weights.
    
    For complex weights, we initialize both real and imaginary parts
    with variance scaled by 1/sqrt(fan_in).
    """
    fan_in = math.prod(shape[:-1])
    std = math.sqrt(2.0 / fan_in)
    
    key1, key2 = jax.random.split(key)
    Wr = jax.random.normal(key1, shape) * std
    Wi = jax.random.normal(key2, shape) * std
    
    return Wr + 1j * Wi


def complex_xavier_init(key: PRNGKey, shape: Tuple[int, ...]) -> Array:
    """
    Xavier/Glorot initialization for complex weights.
    """
    fan_in = math.prod(shape[:-1])
    fan_out = shape[-1]
    std = math.sqrt(2.0 / (fan_in + fan_out))
    
    key1, key2 = jax.random.split(key)
    Wr = jax.random.normal(key1, shape) * std
    Wi = jax.random.normal(key2, shape) * std
    
    return Wr + 1j * Wi

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
