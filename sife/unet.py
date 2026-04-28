import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Any, Callable, Optional, Sequence, Tuple
import math

class ComplexLinear(nn.Module):
    features: int
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        in_features = inputs.shape[-1]
        k_real = self.param('kernel_real', nn.initializers.glorot_normal(), (in_features, self.features))
        k_imag = self.param('kernel_imag', nn.initializers.glorot_normal(), (in_features, self.features))
        kernel = k_real + 1j * k_imag
        out = jnp.dot(inputs, kernel)
        if self.use_bias:
            b_real = self.param('bias_real', nn.initializers.zeros_init(), (self.features,))
            b_imag = self.param('bias_imag', nn.initializers.zeros_init(), (self.features,))
            bias = b_real + 1j * b_imag
            out = out + bias
        return out

class ComplexConv1D(nn.Module):
    features: int
    kernel_size: Tuple[int, ...]
    strides: Tuple[int, ...] = (1,)
    padding: str = 'SAME'
    use_bias: bool = True

    @nn.compact
    def __call__(self, inputs):
        # We can implement complex conv using real and imaginary parts explicitly.
        # (A + iB) * (C + iD) = (AC - BD) + i(AD + BC)
        conv_real = nn.Conv(self.features, self.kernel_size, strides=self.strides, padding=self.padding, use_bias=False, name='conv_real')
        conv_imag = nn.Conv(self.features, self.kernel_size, strides=self.strides, padding=self.padding, use_bias=False, name='conv_imag')
        
        real_out = conv_real(inputs.real) - conv_imag(inputs.imag)
        imag_out = conv_real(inputs.imag) + conv_imag(inputs.real)
        
        out = real_out + 1j * imag_out
        
        if self.use_bias:
            b_real = self.param('bias_real', nn.initializers.zeros_init(), (self.features,))
            b_imag = self.param('bias_imag', nn.initializers.zeros_init(), (self.features,))
            out = out + (b_real + 1j * b_imag)
            
        return out

class ComplexModReLU(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        amp = jnp.abs(inputs)
        phase = jnp.angle(inputs)
        bias = self.param('bias', nn.initializers.zeros_init(), (inputs.shape[-1],))
        new_amp = nn.relu(amp + bias)
        return new_amp * jnp.exp(1j * phase)

class ComplexLayerNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, inputs):
        # Normalize by mean magnitude (RMS), scale-only
        mag_sq = jnp.abs(inputs) ** 2
        rms = jnp.sqrt(jnp.mean(mag_sq, axis=-1, keepdims=True) + self.epsilon)
        normed = inputs / rms
        
        scale_real = self.param('scale_real', nn.initializers.ones_init(), (inputs.shape[-1],))
        scale_imag = self.param('scale_imag', nn.initializers.zeros_init(), (inputs.shape[-1],))
        scale = scale_real + 1j * scale_imag
        
        return normed * scale

class ComplexDropout(nn.Module):
    rate: float
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        if deterministic or self.rate == 0.0:
            return inputs
        
        keep_prob = 1.0 - self.rate
        rng = self.make_rng('dropout')
        # dropout mask on complex numbers
        mask = jax.random.bernoulli(rng, p=keep_prob, shape=inputs.shape)
        return inputs * mask / keep_prob

class ComplexSelfAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, deterministic: Optional[bool] = None):
        B, L, C = inputs.shape
        assert C == self.num_heads * self.head_dim
        
        q = ComplexLinear(C, use_bias=False)(inputs).reshape(B, L, self.num_heads, self.head_dim)
        k = ComplexLinear(C, use_bias=False)(inputs).reshape(B, L, self.num_heads, self.head_dim)
        v = ComplexLinear(C, use_bias=False)(inputs).reshape(B, L, self.num_heads, self.head_dim)
        
        # Hermitian inner product: Re(Q K^H)
        # q: (B, L, H, D), k: (B, S, H, D)
        # scores: (B, H, L, S)
        q_real, q_imag = q.real, q.imag
        k_real, k_imag = k.real, k.imag
        
        scores = jnp.einsum('blhd,bshd->bhls', q_real, k_real) + jnp.einsum('blhd,bshd->bhls', q_imag, k_imag)
        scores = scores / math.sqrt(self.head_dim)
        
        attn_weights = nn.softmax(scores, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(attn_weights, deterministic=deterministic)
        
        # Multiply with v: (B, H, L, S) @ (B, S, H, D) -> (B, L, H, D)
        out = jnp.einsum('bhls,bshd->blhd', attn_weights, v)
        out = out.reshape(B, L, C)
        
        return ComplexLinear(C)(out)

class ComplexCrossAttention(nn.Module):
    num_heads: int
    head_dim: int
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, context, deterministic: Optional[bool] = None):
        B, L, C = inputs.shape
        _, S, _ = context.shape
        
        q = ComplexLinear(C, use_bias=False)(inputs).reshape(B, L, self.num_heads, self.head_dim)
        k = ComplexLinear(C, use_bias=False)(context).reshape(B, S, self.num_heads, self.head_dim)
        v = ComplexLinear(C, use_bias=False)(context).reshape(B, S, self.num_heads, self.head_dim)
        
        q_real, q_imag = q.real, q.imag
        k_real, k_imag = k.real, k.imag
        
        scores = jnp.einsum('blhd,bshd->bhls', q_real, k_real) + jnp.einsum('blhd,bshd->bhls', q_imag, k_imag)
        scores = scores / math.sqrt(self.head_dim)
        
        attn_weights = nn.softmax(scores, axis=-1)
        attn_weights = nn.Dropout(self.dropout_rate, deterministic=self.deterministic)(attn_weights, deterministic=deterministic)
        
        out = jnp.einsum('bhls,bshd->blhd', attn_weights, v)
        out = out.reshape(B, L, C)
        
        return ComplexLinear(C)(out)

class ComplexTimeEmbedding(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, t):
        # t can be (B,)
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        
        # Map to complex via Amp/Phase formulation
        dense_amp = nn.Dense(self.dim)
        dense_phase = nn.Dense(self.dim)
        
        amp = nn.softplus(dense_amp(emb))
        phase = 2 * jnp.pi * nn.tanh(dense_phase(emb))
        
        return amp * jnp.exp(1j * phase)

class ComplexTransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x, context=None, t_emb=None, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        
        # Add time embedding
        if t_emb is not None:
            x = x + t_emb[:, None, :]
            
        # Self Attention
        y = ComplexLayerNorm()(x)
        y = ComplexSelfAttention(self.num_heads, self.head_dim, self.dropout_rate)(y, deterministic=deterministic)
        x = x + ComplexDropout(self.dropout_rate)(y, deterministic=deterministic)
        
        # Cross Attention
        if context is not None:
            y = ComplexLayerNorm()(x)
            y = ComplexCrossAttention(self.num_heads, self.head_dim, self.dropout_rate)(y, context, deterministic=deterministic)
            x = x + ComplexDropout(self.dropout_rate)(y, deterministic=deterministic)
            
        # MLP
        y = ComplexLayerNorm()(x)
        y = ComplexLinear(self.mlp_dim)(y)
        y = ComplexModReLU()(y)
        y = ComplexDropout(self.dropout_rate)(y, deterministic=deterministic)
        y = ComplexLinear(x.shape[-1])(y)
        x = x + ComplexDropout(self.dropout_rate)(y, deterministic=deterministic)
        
        return x

def PositionalPhaseEncoding(seq_len: int, dim: int):
    positions = jnp.arange(seq_len)[:, None]
    frequencies = jnp.exp(jnp.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    angles = positions * frequencies[None, :]
    
    # Generate phases wrapped to [-pi, pi]
    phases = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
    return jnp.exp(1j * phases)

class UnifiedSIFETransformer(nn.Module):
    num_layers: int
    num_heads: int
    head_dim: int
    mlp_dim: int
    out_dim: int
    dropout_rate: float = 0.0
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, x, t, context=None, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        B, L, _ = x.shape
        C = self.num_heads * self.head_dim
        
        x = ComplexLinear(C)(x)
        
        # Positional Phase Encoding
        pos_enc = PositionalPhaseEncoding(L, C)
        x = x * pos_enc[None, ...]
        
        t_emb = ComplexTimeEmbedding(C)(t)
        
        for _ in range(self.num_layers):
            x = ComplexTransformerBlock(
                self.num_heads, self.head_dim, self.mlp_dim, self.dropout_rate
            )(x, context=context, t_emb=t_emb, deterministic=deterministic)
            
        x = ComplexLayerNorm()(x)
        x = ComplexLinear(self.out_dim)(x)
        
        return x

class ComplexPatchEncoder(nn.Module):
    patch_size: int
    out_dim: int

    @nn.compact
    def __call__(self, images):
        # images: (B, H, W, C) complex field
        B, H, W, C = images.shape
        P = self.patch_size
        
        # Extract patches
        x = images.reshape(B, H // P, P, W // P, P, C)
        x = x.transpose((0, 1, 3, 2, 4, 5))
        x = x.reshape(B, -1, P * P * C)
        
        # Map to complex dim
        x = ComplexLinear(self.out_dim)(x)
        return x
