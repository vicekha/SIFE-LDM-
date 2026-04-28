import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Any, Callable
from functools import partial

# Constants for schedule
def cosine_beta_schedule(timesteps: int, s: float = 0.008):
    steps = timesteps + 1
    x = jnp.linspace(0, timesteps, steps)
    alphas_cumprod = jnp.cos(((x / timesteps) + s) / (1 + s) * jnp.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return jnp.clip(betas, 0.0001, 0.9999)

class GaussianDiffusion:
    def __init__(self, timesteps: int = 1000):
        self.timesteps = timesteps
        self.betas = cosine_beta_schedule(timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        
    def q_sample(self, x0: jnp.ndarray, t: jnp.ndarray, noise: jnp.ndarray) -> jnp.ndarray:
        """
        Apply Polar Coordinate Diffusion Noise.
        Gaussian on Amplitude, Wrapped Gaussian on Phase.
        noise is expected to be a complex array where real is noise_A and imag is noise_theta.
        """
        A0 = jnp.abs(x0)
        theta0 = jnp.angle(x0)
        
        noise_A = jnp.real(noise)
        noise_theta = jnp.imag(noise)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Expand dims for broadcasting if necessary
        while len(sqrt_alpha.shape) < len(x0.shape):
            sqrt_alpha = jnp.expand_dims(sqrt_alpha, -1)
            sqrt_one_minus_alpha = jnp.expand_dims(sqrt_one_minus_alpha, -1)
            
        A_t = sqrt_alpha * A0 + sqrt_one_minus_alpha * noise_A
        A_t = jnp.abs(A_t) # Amplitude must be positive
        
        # Wrapped Gaussian for phase: sigma approaches pi as alpha approaches 0
        sigma_theta = sqrt_one_minus_alpha * jnp.pi
        theta_t = theta0 + sigma_theta * noise_theta
        theta_t = ((theta_t + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        
        return A_t * jnp.exp(1j * theta_t)

class DDIMSampler:
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
        
    def sample_step(self, x_t: jnp.ndarray, pred_noise: jnp.ndarray, t: int, t_prev: int, eta: float = 0.0) -> jnp.ndarray:
        alpha = self.diffusion.alphas_cumprod[t]
        alpha_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev >= 0 else 1.0
        
        A_t = jnp.abs(x_t)
        theta_t = jnp.angle(x_t)
        
        noise_A = jnp.real(pred_noise)
        noise_theta = jnp.imag(pred_noise)
        
        sigma_t = eta * jnp.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
        
        # Predict x0
        pred_A0 = (A_t - jnp.sqrt(1 - alpha) * noise_A) / jnp.sqrt(alpha)
        # For phase, the noise was scaled by pi * sqrt(1 - alpha)
        pred_theta0 = theta_t - jnp.sqrt(1 - alpha) * jnp.pi * noise_theta
        
        # Direction pointing to x_t
        dir_A_t = jnp.sqrt(1 - alpha_prev - sigma_t**2) * noise_A
        dir_theta_t = jnp.sqrt(1 - alpha_prev - sigma_t**2) * jnp.pi * noise_theta
        
        A_prev = jnp.sqrt(alpha_prev) * pred_A0 + dir_A_t
        theta_prev = pred_theta0 + dir_theta_t
        
        # Noise (if eta > 0)
        # To strictly implement DDIM, we can assume eta=0 for deterministic sampling
        
        theta_prev = ((theta_prev + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        return jnp.abs(A_prev) * jnp.exp(1j * theta_prev)

class MaskedDiffusion:
    def __init__(self, timesteps: int = 100):
        self.timesteps = timesteps
        
    def apply_mask(self, x0: jnp.ndarray, mask_ratio: float, key: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        x0: Complex embedding where unmasked amplitude = 1.0, phase = semantic rotation.
        mask_ratio: fraction of tokens to mask.
        Returns masked_x and mask (1 if masked, 0 if unmasked).
        """
        B, L, D = x0.shape
        # Masking per token
        rand = jax.random.uniform(key, (B, L, 1))
        mask = rand < mask_ratio
        
        # Masked tokens have Amplitude = 0.0, phase random or 0
        A0 = jnp.abs(x0)
        theta0 = jnp.angle(x0)
        
        A_masked = jnp.where(mask, 0.0, A0)
        
        masked_x = A_masked * jnp.exp(1j * theta0)
        return masked_x, mask
        
    def unmask_step(self, current_x: jnp.ndarray, pred_phase: jnp.ndarray, unmask_ratio: float) -> jnp.ndarray:
        """
        For inference. Unmask the top confident tokens.
        Confidence can be represented by the coherence or explicitly modeled.
        Here we assume pred_phase gives the target phases, and we just pick a fraction to unmask.
        """
        # A simple iteration strategy: turn amplitude from 0 to 1 for the selected tokens.
        # This will be detailed in inference logic, here we just provide the primitive.
        pass

class SIFEDiffusion:
    def __init__(self, diffusion: GaussianDiffusion, hamiltonian_fn: Callable, guidance_scale: float = 0.1):
        self.diffusion = diffusion
        self.hamiltonian_fn = hamiltonian_fn
        self.guidance_scale = guidance_scale
        
    def sample_step_with_guidance(self, x_t: jnp.ndarray, pred_noise: jnp.ndarray, t: int, t_prev: int, config: Any) -> jnp.ndarray:
        """
        Apply Hamiltonian guidance: x_t = x_t - scale * grad(H(x_t))
        """
        # Compute gradient of Hamiltonian w.r.t complex field x_t
        def energy_fn(x):
            from .field import SIFField
            # Dummy field state to compute energy
            # We assume x is 1D or 2D complex patch sequence
            field = SIFField(
                amplitude=jnp.abs(x),
                phase=jnp.angle(x),
                fluctuation=jnp.angle(x),
                velocity_amp=jnp.zeros_like(x.real),
                velocity_phi=jnp.zeros_like(x.real)
            )
            return self.hamiltonian_fn(field, config)
            
        grad_H = jax.grad(energy_fn)(x_t)
        
        # Apply guidance to the predicted noise or directly to x_t
        # Usually guidance adjusts the predicted noise: eps_hat = eps - scale * grad
        guided_noise = pred_noise + self.guidance_scale * grad_H
        
        sampler = DDIMSampler(self.diffusion)
        return sampler.sample_step(x_t, guided_noise, t, t_prev)
