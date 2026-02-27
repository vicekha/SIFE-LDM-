"""
SIFE-LDM: Diffusion Process Implementation
===========================================

Implements the latent diffusion process for SIFE-LDM, including:
- Forward diffusion process (adding noise)
- Reverse diffusion process (denoising)
- DDIM sampler for fast sampling
- Guidance mechanisms (Hamiltonian, Truth Potential)

The diffusion operates on the complex-valued SIFE field,
adding Gaussian noise to both amplitude and phase components.

Author: SIFE-LDM Research Team
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax
from typing import Tuple, Optional, Callable, Sequence, NamedTuple
from functools import partial
import math

# Type aliases
Array = jnp.ndarray
PRNGKey = jnp.ndarray


class DiffusionConfig(NamedTuple):
    """Configuration for the diffusion process."""
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule: str = 'linear'  # 'linear', 'cosine', 'sqrt'
    prediction_type: str = 'epsilon'  # 'epsilon', 'x0', 'v'
    clip_denoised: bool = True
    clip_range: Tuple[float, float] = (-1.0, 1.0)


class GaussianDiffusion:
    """
    Gaussian diffusion process for complex-valued fields.
    
    The forward process adds Gaussian noise to both the real and imaginary
    parts of the complex field:
    
    q(Ψ_t | Ψ_0) = N(Ψ_t; √(ᾱ_t) Ψ_0, (1 - ᾱ_t) I)
    
    where α_t are the noise schedule coefficients.
    """
    
    def __init__(self, config: DiffusionConfig):
        self.config = config
        self.num_timesteps = config.num_timesteps
        
        # Compute noise schedule
        self.betas = self._get_betas(config)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = jnp.cumprod(self.alphas)
        self.alphas_cumprod_prev = jnp.concatenate([jnp.array([1.0]), self.alphas_cumprod[:-1]])
        
        # Precompute useful quantities
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1.0)
        
        # Posterior coefficients
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = jnp.log(
            jnp.maximum(self.posterior_variance, 1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * jnp.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _get_betas(self, config: DiffusionConfig) -> Array:
        """Compute the noise schedule β_t."""
        if config.schedule == 'linear':
            betas = jnp.linspace(config.beta_start, config.beta_end, config.num_timesteps)
        elif config.schedule == 'cosine':
            # Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
            steps = config.num_timesteps + 1
            s = 0.008
            t = jnp.linspace(0, 1, steps)
            alphas_cumprod = jnp.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = jnp.clip(betas, 0.0001, 0.9999)
        elif config.schedule == 'sqrt':
            # Square root schedule
            steps = config.num_timesteps
            t = jnp.linspace(0, 1, steps)
            alphas_cumprod = 1 - jnp.sqrt(t + 0.01)
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = jnp.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {config.schedule}")
        
        return betas
    
    def q_sample(
        self,
        x_0: Array,
        t: Array,
        key: PRNGKey,
        noise: Optional[Array] = None
    ) -> Array:
        """
        Sample from the forward diffusion process q(Ψ_t | Ψ_0).
        
        Args:
            x_0: Initial clean sample (complex field)
            t: Timestep indices
            key: Random key
            noise: Optional pre-generated noise
        
        Returns:
            Noisy sample Ψ_t
        """
        if noise is None:
            noise = jax.random.normal(key, x_0.shape, dtype=jnp.float32)
            noise = noise + 1j * jax.random.normal(jax.random.split(key)[0], x_0.shape, dtype=jnp.float32)
        
        # Get coefficients for timestep t
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alpha = sqrt_alpha.reshape(-1, *([1] * (x_0.ndim - 1)))
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(-1, *([1] * (x_0.ndim - 1)))
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    def predict_x0_from_epsilon(
        self,
        x_t: Array,
        epsilon: Array,
        t: Array
    ) -> Array:
        """
        Predict x_0 from the noise prediction.
        
        Ψ_0 = (Ψ_t - √(1-ᾱ_t) ε) / √(ᾱ_t)
        """
        sqrt_recip = self.sqrt_recip_alphas_cumprod[t]
        sqrt_recipm1 = self.sqrt_recipm1_alphas_cumprod[t]
        
        sqrt_recip = sqrt_recip.reshape(-1, *([1] * (x_t.ndim - 1)))
        sqrt_recipm1 = sqrt_recipm1.reshape(-1, *([1] * (x_t.ndim - 1)))
        
        x_0 = sqrt_recip * x_t - sqrt_recipm1 * epsilon
        
        if self.config.clip_denoised:
            # For complex values, clip the amplitude while preserving phase
            amplitude = jnp.abs(x_0)
            phase = jnp.angle(x_0)
            amplitude = jnp.clip(amplitude, self.config.clip_range[0], self.config.clip_range[1])
            x_0 = amplitude * jnp.exp(1j * phase)
        
        return x_0
    
    def predict_epsilon_from_x0(
        self,
        x_t: Array,
        x_0: Array,
        t: Array
    ) -> Array:
        """
        Predict noise from x_0 prediction.
        
        ε = (Ψ_t - √(ᾱ_t) Ψ_0) / √(1-ᾱ_t)
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        sqrt_alpha = sqrt_alpha.reshape(-1, *([1] * (x_t.ndim - 1)))
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(-1, *([1] * (x_t.ndim - 1)))
        
        return (x_t - sqrt_alpha * x_0) / sqrt_one_minus_alpha
    
    def predict_v_from_x0_epsilon(
        self,
        x_0: Array,
        epsilon: Array,
        t: Array
    ) -> Array:
        """
        Predict v = √(ᾱ_t) ε - √(1-ᾱ_t) Ψ_0
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        sqrt_alpha = sqrt_alpha.reshape(-1, *([1] * (x_0.ndim - 1)))
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(-1, *([1] * (x_0.ndim - 1)))
        
        return sqrt_alpha * epsilon - sqrt_one_minus_alpha * x_0
    
    def predict_x0_from_v(
        self,
        x_t: Array,
        v: Array,
        t: Array
    ) -> Array:
        """
        Predict x_0 from v prediction.
        
        Ψ_0 = √(ᾱ_t) Ψ_t - √(1-ᾱ_t) v
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        
        sqrt_alpha = sqrt_alpha.reshape(-1, *([1] * (x_t.ndim - 1)))
        sqrt_one_minus_alpha = sqrt_one_minus_alpha.reshape(-1, *([1] * (x_t.ndim - 1)))
        
        return sqrt_alpha * x_t - sqrt_one_minus_alpha * v
    
    def q_posterior_mean_variance(
        self,
        x_0: Array,
        x_t: Array,
        t: Array
    ) -> Tuple[Array, Array, Array]:
        """
        Compute the posterior q(Ψ_{t-1} | Ψ_t, Ψ_0).
        
        Returns:
            Tuple of (mean, variance, log_variance)
        """
        mean = (
            self.posterior_mean_coef1[t].reshape(-1, *([1] * (x_0.ndim - 1))) * x_0 +
            self.posterior_mean_coef2[t].reshape(-1, *([1] * (x_t.ndim - 1))) * x_t
        )
        variance = self.posterior_variance[t]
        log_variance = self.posterior_log_variance_clipped[t]
        
        return mean, variance, log_variance
    
    def p_sample(
        self,
        model: Callable,
        x_t: Array,
        t: Array,
        t_index: int,
        key: PRNGKey,
        context: Optional[Array] = None,
        clip_denoised: bool = True
    ) -> Array:
        """
        Sample from p(Ψ_{t-1} | Ψ_t) using the model prediction.
        """
        # Model prediction
        model_output = model(x_t, t, context)
        
        # Convert to x_0 prediction based on prediction type
        if self.config.prediction_type == 'epsilon':
            x_0 = self.predict_x0_from_epsilon(x_t, model_output, t)
            epsilon = model_output
        elif self.config.prediction_type == 'x0':
            x_0 = model_output
            epsilon = self.predict_epsilon_from_x0(x_t, x_0, t)
        elif self.config.prediction_type == 'v':
            x_0 = self.predict_x0_from_v(x_t, model_output, t)
            epsilon = self.predict_epsilon_from_x0(x_t, x_0, t)
        
        if clip_denoised:
            # For complex values, clip the amplitude while preserving phase
            amplitude = jnp.abs(x_0)
            phase = jnp.angle(x_0)
            amplitude = jnp.clip(amplitude, self.config.clip_range[0], self.config.clip_range[1])
            x_0 = amplitude * jnp.exp(1j * phase)
        
        # Get posterior
        model_mean, variance, log_variance = self.q_posterior_mean_variance(x_0, x_t, t)
        
        # Add noise (except at t=0)
        noise = jax.random.normal(key, x_t.shape, dtype=jnp.float32)
        noise = noise + 1j * jax.random.normal(jax.random.split(key)[0], x_t.shape, dtype=jnp.float32)
        
        # No noise at t=0
        nonzero_mask = jnp.array(t_index > 0, dtype=jnp.float32)
        nonzero_mask = nonzero_mask.reshape(-1, *([1] * (x_t.ndim - 1)))
        
        return model_mean + nonzero_mask * jnp.exp(0.5 * log_variance).reshape(-1, *([1] * (x_t.ndim - 1))) * noise


class DDIMSampler:
    """
    DDIM (Denoising Diffusion Implicit Models) sampler.
    
    DDIM allows for faster sampling by skipping timesteps while
    maintaining sample quality. It uses a deterministic sampling
    process that can be made stochastic with the eta parameter.
    
    From "Denoising Diffusion Implicit Models" (Jiaming Song et al., 2020)
    """
    
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
        self.num_timesteps = diffusion.num_timesteps
    
    def ddim_step(
        self,
        model: Callable,
        x_t: Array,
        t: int,
        t_prev: int,
        key: PRNGKey,
        context: Optional[Array] = None,
        eta: float = 0.0,
        clip_denoised: bool = True
    ) -> Array:
        """
        Perform one DDIM sampling step.
        
        Args:
            model: Noise prediction model
            x_t: Current noisy sample
            t: Current timestep
            t_prev: Previous timestep (t-1 or earlier if skipping)
            key: Random key
            context: Optional conditioning context
            eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
            clip_denoised: Whether to clip predicted x_0
        
        Returns:
            Sample at timestep t_prev
        """
        # Get model prediction
        t_array = jnp.array([t])
        model_output = model(x_t, t_array, context)
        
        # Convert to x_0 and epsilon
        if self.diffusion.config.prediction_type == 'epsilon':
            epsilon = model_output
            x_0 = self.diffusion.predict_x0_from_epsilon(x_t, epsilon, t_array)
        elif self.diffusion.config.prediction_type == 'x0':
            x_0 = model_output
            epsilon = self.diffusion.predict_epsilon_from_x0(x_t, x_0, t_array)
        elif self.diffusion.config.prediction_type == 'v':
            x_0 = self.diffusion.predict_x0_from_v(x_t, model_output, t_array)
            epsilon = self.diffusion.predict_epsilon_from_x0(x_t, x_0, t_array)
        
        if clip_denoised:
            # For complex values, clip the amplitude while preserving phase
            amplitude = jnp.abs(x_0)
            phase = jnp.angle(x_0)
            amplitude = jnp.clip(amplitude, self.diffusion.config.clip_range[0], self.diffusion.config.clip_range[1])
            x_0 = amplitude * jnp.exp(1j * phase)
        
        # Get alpha values
        alpha_t = self.diffusion.alphas_cumprod[t]
        alpha_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev >= 0 else jnp.array(1.0)
        
        # Compute sigma
        sigma = eta * jnp.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        
        # Compute the direction to x_t
        dir_xt = jnp.sqrt(1 - alpha_prev - sigma ** 2) * epsilon
        
        # Compute noise
        noise = jax.random.normal(key, x_t.shape, dtype=jnp.float32)
        noise = noise + 1j * jax.random.normal(jax.random.split(key)[0], x_t.shape, dtype=jnp.float32)
        
        # Compute x_{t-1}
        x_prev = jnp.sqrt(alpha_prev) * x_0 + dir_xt + sigma * noise
        
        return x_prev
    
    def sample(
        self,
        model: Callable,
        shape: Tuple[int, ...],
        key: PRNGKey,
        context: Optional[Array] = None,
        num_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True,
        return_intermediates: bool = False
    ) -> Array:
        """
        Generate samples using DDIM.
        
        Args:
            model: Noise prediction model
            shape: Shape of samples to generate
            key: Random key
            context: Optional conditioning context
            num_steps: Number of sampling steps (fewer = faster)
            eta: Stochasticity parameter
            clip_denoised: Whether to clip predicted x_0
            return_intermediates: Whether to return all intermediate samples
        
        Returns:
            Generated samples (and intermediates if requested)
        """
        # Determine which timesteps to use
        c = self.num_timesteps // num_steps
        ddim_timesteps = jnp.arange(0, self.num_timesteps, c)[::-1]
        
        # Initialize from pure noise
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape, dtype=jnp.float32)
        x = x + 1j * jax.random.normal(jax.random.split(subkey)[0], shape, dtype=jnp.float32)
        
        intermediates = [x] if return_intermediates else None
        
        # Iterate through timesteps
        for i, t in enumerate(ddim_timesteps):
            t_prev = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else -1
            
            key, subkey = jax.random.split(key)
            x = self.ddim_step(
                model, x, int(t), int(t_prev), subkey, context, eta, clip_denoised
            )
            
            if return_intermediates:
                intermediates.append(x)
        
        if return_intermediates:
            return x, intermediates
        return x
    
    def sample_loop(
        self,
        model: Callable,
        x_t: Array,
        key: PRNGKey,
        context: Optional[Array] = None,
        num_steps: int = 50,
        eta: float = 0.0,
        clip_denoised: bool = True
    ) -> Array:
        """
        Sample loop that can be used with jax.lax.scan for efficiency.
        """
        c = self.num_timesteps // num_steps
        ddim_timesteps = jnp.arange(0, self.num_timesteps, c)[::-1]
        
        def step_fn(carry, t):
            x, key = carry
            t_idx = jnp.argmax(ddim_timesteps == t)
            t_prev = jnp.where(t_idx + 1 < len(ddim_timesteps), ddim_timesteps[t_idx + 1], -1)
            
            key, subkey = jax.random.split(key)
            x = self.ddim_step(
                model, x, int(t), int(t_prev), subkey, context, eta, clip_denoised
            )
            
            return (x, key), None
        
        keys = jax.random.split(key, len(ddim_timesteps))
        (x, _), _ = lax.scan(step_fn, (x_t, key), ddim_timesteps)
        
        return x


class EulerMaruyamaSampler:
    """
    Continuous-Time Stochastic Differential Equation (SDE) Diffusion sampler.
    Uses Euler-Maruyama to solve the reverse-time SDE for generating the complex field.
    
    Reverse SDE: dΨ = [f(Ψ, t) - g(t)^2 ∇_Ψ log p_t(Ψ)] dt + g(t) dW
    """
    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
    
    def step(
        self,
        model: Callable,
        x_t: Array,
        t: float,
        dt: float,
        key: PRNGKey,
        context: Optional[Array] = None
    ) -> Array:
        """
        Perform one Euler-Maruyama step dt backward in continuous time t.
        """
        # Convert continuous t to discrete indices or continuous embedding
        t_idx = jnp.clip(int(t * self.diffusion.num_timesteps), 0, self.diffusion.num_timesteps - 1)
        t_array = jnp.array([t_idx], dtype=jnp.int32)
        
        # Predict noise using standard parameterized network
        epsilon = model(x_t, t_array, context)
        
        # Compute Score: s_theta = -epsilon / sqrt(1 - alpha_bar)
        sqrt_1m_alpha = self.diffusion.sqrt_one_minus_alphas_cumprod[t_idx]
        score = -epsilon / jnp.maximum(sqrt_1m_alpha, 1e-5)
        
        # Drift and diffusion terms for variance preserving (VP) SDE
        # f(x, t) = -1/2 beta(t) x
        # g(t)^2 = beta(t)
        beta_t = self.diffusion.betas[t_idx]
        
        drift = -0.5 * beta_t * x_t - beta_t * score
        diffusion_term = jnp.sqrt(beta_t)
        
        # dt is negative for reverse time
        abs_dt = jnp.abs(dt)
        
        # Noise
        noise = jax.random.normal(key, x_t.shape, dtype=jnp.float32)
        noise = noise + 1j * jax.random.normal(jax.random.split(key)[0], x_t.shape, dtype=jnp.float32)
        
        # Euler-Maruyama update: x_{t-dt} = x_t - drift * dt + diffusion * sqrt(dt) * noise
        # For reverse SDE, we subtract the drift term.
        x_prev = x_t - drift * abs_dt + diffusion_term * jnp.sqrt(abs_dt) * noise
        
        return x_prev
    
    def sample(
        self,
        model: Callable,
        shape: Tuple[int, ...],
        key: PRNGKey,
        context: Optional[Array] = None,
        num_steps: int = 100,
        sife_config=None,
        stability_threshold: float = 0.5
    ) -> Tuple[Array, int]:
        """
        Generate samples using the reverse-time SDE.
        
        If sife_config is provided, uses Attractor-Based Adaptive Inference:
        the loop exits early when the decoded field reaches a stable low-energy
        Hamiltonian state (is_field_stable), saving compute on easy prompts.
        
        Returns:
            (sample, steps_taken) — steps_taken < num_steps means early stopped.
        """
        # Initialize from noise
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape, dtype=jnp.float32)
        x = x + 1j * jax.random.normal(jax.random.split(subkey)[0], shape, dtype=jnp.float32)
        
        dt = 1.0 / num_steps
        time_steps = jnp.linspace(1.0, dt, num_steps)
        
        steps_taken = num_steps
        for i, t in enumerate(time_steps):
            key, subkey = jax.random.split(key)
            x = self.step(model, x, float(t), dt, subkey, context)
            
            # Attractor-Based Adaptive Inference (System 2 Stopping)
            # Only check every 10 steps for efficiency
            if sife_config is not None and i % 10 == 0 and i > 0:
                from .field import SIFField, is_field_stable
                # Decode x into a lightweight SIFField for stability check
                amp = jnp.abs(x)
                phi = jnp.angle(x)
                field = SIFField(
                    amplitude=amp[0],          # Check first sample in batch
                    phase=phi[0],
                    fluctuation=phi[0],
                    velocity_amp=jnp.zeros_like(amp[0]),
                    velocity_phi=jnp.zeros_like(phi[0])
                )
                if is_field_stable(field, sife_config, threshold=stability_threshold):
                    steps_taken = i + 1
                    break
        
        return x, steps_taken
    
    def cfg_guided_sample(
        self,
        model: Callable,
        shape: Tuple[int, ...],
        key: PRNGKey,
        context: Optional[Array] = None,
        guidance_scale: float = 7.5,
        num_steps: int = 100,
        sife_config=None,
        stability_threshold: float = 0.5
    ) -> Tuple[Array, int]:
        """
        Phase-Coherent Classifier-Free Guidance (CFG) sampling.
        
        Runs the model twice per step:
          - Conditional:   ε_cond   = model(x, t, context)
          - Unconditional: ε_uncond = model(x, t, None)
        
        Blended guidance in phase space:
          ε_guided = ε_uncond + s * (ε_cond - ε_uncond)
        
        This steers generation toward the target class without any additional
        parameters — purely exploiting the existing context slot.
        """
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape, dtype=jnp.float32)
        x = x + 1j * jax.random.normal(jax.random.split(subkey)[0], shape, dtype=jnp.float32)
        
        dt = 1.0 / num_steps
        time_steps = jnp.linspace(1.0, dt, num_steps)
        
        steps_taken = num_steps
        for i, t in enumerate(time_steps):
            t_idx = jnp.clip(int(float(t) * self.diffusion.num_timesteps),
                             0, self.diffusion.num_timesteps - 1)
            t_array = jnp.array([t_idx], dtype=jnp.int32)
            
            # Conditional noise prediction
            eps_cond = model(x, t_array, context)
            # Unconditional noise prediction (null context)
            eps_uncond = model(x, t_array, None)
            
            # CFG blend: steer epsilon toward the class
            eps_guided = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            
            # Apply SDE step with the guided score
            sqrt_1m_alpha = self.diffusion.sqrt_one_minus_alphas_cumprod[t_idx]
            score = -eps_guided / jnp.maximum(sqrt_1m_alpha, 1e-5)
            beta_t = self.diffusion.betas[t_idx]
            drift = -0.5 * beta_t * x - beta_t * score
            diffusion_term = jnp.sqrt(beta_t)
            abs_dt = jnp.abs(dt)
            
            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, x.shape, dtype=jnp.float32)
            noise = noise + 1j * jax.random.normal(jax.random.split(subkey)[0], x.shape, dtype=jnp.float32)
            x = x - drift * abs_dt + diffusion_term * jnp.sqrt(abs_dt) * noise
            
            # Attractor-Based Early Stopping
            if sife_config is not None and i % 10 == 0 and i > 0:
                from .field import SIFField, is_field_stable
                amp = jnp.abs(x)
                phi = jnp.angle(x)
                field = SIFField(
                    amplitude=amp[0],
                    phase=phi[0],
                    fluctuation=phi[0],
                    velocity_amp=jnp.zeros_like(amp[0]),
                    velocity_phi=jnp.zeros_like(phi[0])
                )
                if is_field_stable(field, sife_config, threshold=stability_threshold):
                    steps_taken = i + 1
                    break
        
        return x, steps_taken




class SIFEDiffusion:
    """
    SIFE-specific diffusion process that incorporates field dynamics.
    
    This extends the standard diffusion with:
    1. Hamiltonian guidance: Steer generation toward low-energy states
    2. Truth potential guidance: Favor phase-coherent configurations
    3. Field evolution: Apply SIFE dynamics during sampling
    """
    
    def __init__(
        self,
        diffusion: GaussianDiffusion,
        sife_config,  # SIFEConfig from field.py
        guidance_scale_hamiltonian: float = 0.1,
        guidance_scale_truth: float = 0.1,
        use_field_evolution: bool = True,
        field_evolution_steps: int = 5
    ):
        self.diffusion = diffusion
        self.sife_config = sife_config
        self.guidance_scale_h = guidance_scale_hamiltonian
        self.guidance_scale_t = guidance_scale_truth
        self.use_field_evolution = use_field_evolution
        self.field_evolution_steps = field_evolution_steps
    
    def guided_noise_prediction(
        self,
        model: Callable,
        x_t: Array,
        t: Array,
        context: Optional[Array] = None,
        context_field: Optional[Array] = None
    ) -> Array:
        """
        Compute noise prediction with SIFE guidance and temporal conditioning.
        
        ε_guided = ε_θ(Ψ_t, t, context, absolute_phase) - η_H ∇_Ψ H(Ψ) - η_Φ ∇_Ψ Φ_T(Ψ, Ψ_ctx)
        
        The absolute phase is ω_0 * t, providing a global clock.
        """
        # Temporal conditioning: absolute phase
        # t is in [0, num_timesteps], map it to "latent time"
        # For simplicity, we can use t directly or omega_0 * t
        abs_phase = self.sife_config.omega_0 * t.astype(jnp.float32)
        
        # Base prediction with absolute phase conditioning
        # Note: Model needs to be updated to accept abs_phase
        epsilon = model(x_t, t, context, abs_phase=abs_phase)
        
        # Convert to amplitude and phase for guidance computation
        amplitude = jnp.abs(x_t)
        phase = jnp.angle(x_t)
        
        # Hamiltonian guidance
        if self.guidance_scale_h > 0:
            from .field import compute_hamiltonian, SIFField
            
            def hamiltonian_fn(x):
                amp = jnp.abs(x)
                total_phi = jnp.angle(x)
                # Recover fluctuation: phi = theta - omega_0 * t
                # abs_phase is already computed in the outer scope
                fluc = total_phi - abs_phase
                
                field = SIFField(
                    amplitude=amp,
                    phase=total_phi,
                    fluctuation=fluc,
                    velocity_amp=jnp.zeros_like(amp),
                    velocity_phi=jnp.zeros_like(amp)
                )
                # Sum over batch for grad
                return jnp.sum(vmap(compute_hamiltonian, in_axes=(0, None))(field, self.sife_config))
            
            # Gradient of Hamiltonian w.r.t. field
            h_grad = grad(hamiltonian_fn)(x_t)
            epsilon = epsilon - self.guidance_scale_h * h_grad
        
        # Truth potential guidance
        if self.guidance_scale_t > 0 and context_field is not None:
            from .field import truth_potential
            
            def truth_fn(x):
                amp = jnp.abs(x)
                ph = jnp.angle(x)
                ctx_amp = jnp.abs(context_field)
                ctx_ph = jnp.angle(context_field)
                
                # Cross truth potential between current and context
                cross_coherence = jnp.sum(
                    jnp.cos(ph - ctx_ph) * amp * ctx_amp
                )
                return cross_coherence
            
            t_grad = grad(truth_fn)(x_t)
            epsilon = epsilon + self.guidance_scale_t * t_grad
        
        return epsilon
    
    def guided_ddim_step(
        self,
        model: Callable,
        x_t: Array,
        t: int,
        t_prev: int,
        key: PRNGKey,
        context: Optional[Array] = None,
        context_field: Optional[Array] = None,
        eta: float = 0.0
    ) -> Array:
        """
        DDIM step with SIFE guidance.
        """
        t_array = jnp.array([t])
        
        # Get guided noise prediction
        epsilon = self.guided_noise_prediction(model, x_t, t_array, context, context_field)
        
        # Predict x_0
        x_0 = self.diffusion.predict_x0_from_epsilon(x_t, epsilon, t_array)
        
        # Clip - for complex values, clip the amplitude while preserving phase
        amplitude = jnp.abs(x_0)
        phase = jnp.angle(x_0)
        amplitude = jnp.clip(amplitude, self.diffusion.config.clip_range[0], self.diffusion.config.clip_range[1])
        x_0 = amplitude * jnp.exp(1j * phase)
        
        # Apply field evolution to x_0
        if self.use_field_evolution and self.field_evolution_steps > 0:
            from .field import SIFField, evolve_field
            
            # Convert to SIFE field
            # The noisy field x_0 is at total phase theta = abs_phase + phi
            abs_phase = self.sife_config.omega_0 * t
            
            field = SIFField(
                amplitude=jnp.abs(x_0),
                phase=jnp.angle(x_0),
                fluctuation=jnp.angle(x_0) - abs_phase,
                velocity_amp=jnp.zeros_like(jnp.abs(x_0)),
                velocity_phi=jnp.zeros_like(jnp.angle(x_0))
            )
            
            # Evolve (batched)
            from jax import vmap
            batched_evolve = vmap(evolve_field, in_axes=(0, None, None))
            field = batched_evolve(field, self.sife_config, self.field_evolution_steps)
            
            # Convert back
            x_0 = field.complex_field
        
        # Get alpha values
        alpha_t = self.diffusion.alphas_cumprod[t]
        alpha_prev = self.diffusion.alphas_cumprod[t_prev] if t_prev >= 0 else jnp.array(1.0)
        
        # Compute sigma
        sigma = eta * jnp.sqrt(
            (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
        )
        
        # Direction
        dir_xt = jnp.sqrt(1 - alpha_prev - sigma ** 2) * epsilon
        
        # Noise
        noise = jax.random.normal(key, x_t.shape, dtype=jnp.float32)
        noise = noise + 1j * jax.random.normal(jax.random.split(key)[0], x_t.shape, dtype=jnp.float32)
        
        return jnp.sqrt(alpha_prev) * x_0 + dir_xt + sigma * noise
    
    def sample(
        self,
        model: Callable,
        shape: Tuple[int, ...],
        key: PRNGKey,
        context: Optional[Array] = None,
        context_field: Optional[Array] = None,
        num_steps: int = 50,
        eta: float = 0.0
    ) -> Array:
        """
        Generate samples with SIFE guidance.
        """
        ddim = DDIMSampler(self.diffusion)
        num_timesteps = self.diffusion.num_timesteps
        
        c = num_timesteps // num_steps
        ddim_timesteps = jnp.arange(0, num_timesteps, c)[::-1]
        
        # Initialize from noise
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape, dtype=jnp.float32)
        x = x + 1j * jax.random.normal(jax.random.split(subkey)[0], shape, dtype=jnp.float32)
        
        # Sample
        for i, t in enumerate(ddim_timesteps):
            t_prev = ddim_timesteps[i + 1] if i + 1 < len(ddim_timesteps) else -1
            
            key, subkey = jax.random.split(key)
            x = self.guided_ddim_step(
                model, x, int(t), int(t_prev), subkey, context, context_field, eta
            )
        
        return x


def compute_loss(
    model: Callable,
    params: dict,
    batch: Array,
    t: Array,
    key: PRNGKey,
    diffusion: GaussianDiffusion,
    context: Optional[Array] = None
) -> Array:
    """
    Compute the diffusion training loss.
    
    L = E_{t, x_0, ε} [||ε - ε_θ(Ψ_t, t)||²]
    
    Args:
        model: Model apply function
        params: Model parameters
        batch: Clean samples
        t: Timesteps
        key: Random key
        diffusion: Diffusion process
        context: Optional conditioning context
    
    Returns:
        Loss value
    """
    # Sample noise
    noise = jax.random.normal(key, batch.shape, dtype=jnp.float32)
    noise = noise + 1j * jax.random.normal(jax.random.split(key)[0], batch.shape, dtype=jnp.float32)
    
    # Add noise to get x_t
    x_t = diffusion.q_sample(batch, t, key, noise)
    
    # Predict noise
    epsilon_pred = model(params, x_t, t, context)
    
    # Compute loss
    loss = jnp.mean(jnp.abs(epsilon_pred - noise) ** 2)
    
    return loss


def compute_v_prediction_loss(
    model: Callable,
    params: dict,
    batch: Array,
    t: Array,
    key: PRNGKey,
    diffusion: GaussianDiffusion,
    context: Optional[Array] = None
) -> Array:
    """
    Compute the v-prediction loss (progressive distillation).
    
    v = √(ᾱ_t) ε - √(1-ᾱ_t) x_0
    
    This parametrization is better for high noise levels.
    """
    # Sample noise
    noise = jax.random.normal(key, batch.shape, dtype=jnp.float32)
    noise = noise + 1j * jax.random.normal(jax.random.split(key)[0], batch.shape, dtype=jnp.float32)
    
    # Get x_t
    x_t = diffusion.q_sample(batch, t, key, noise)
    
    # Compute target v
    v_target = diffusion.predict_v_from_x0_epsilon(batch, noise, t)
    
    # Predict v
    v_pred = model(params, x_t, t, context)
    
    # Loss
    loss = jnp.mean(jnp.abs(v_pred - v_target) ** 2)
    
    return loss


# Learning rate schedules

def cosine_lr_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr_max: float,
    lr_min: float = 0.0
) -> float:
    """
    Cosine learning rate schedule with linear warmup.
    JAX-compatible version using jnp.where.
    """
    # Use jnp.where instead of Python if for JAX compatibility
    warmup_lr = lr_max * step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = jnp.minimum(progress, 1.0)
    
    cosine_lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + jnp.cos(jnp.pi * progress))
    
    return jnp.where(step < warmup_steps, warmup_lr, cosine_lr)


def constant_lr_schedule(
    step: int,
    warmup_steps: int,
    lr_max: float
) -> float:
    """
    Constant learning rate with linear warmup.
    JAX-compatible version.
    """
    warmup_lr = lr_max * step / warmup_steps
    return jnp.where(step < warmup_steps, warmup_lr, lr_max)


def polynomial_lr_schedule(
    step: int,
    total_steps: int,
    warmup_steps: int,
    lr_max: float,
    lr_min: float = 0.0,
    power: float = 1.0
) -> float:
    """
    Polynomial decay learning rate schedule.
    JAX-compatible version.
    """
    warmup_lr = lr_max * step / warmup_steps
    
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    progress = jnp.minimum(progress, 1.0)
    
    poly_lr = lr_min + (lr_max - lr_min) * (1 - progress) ** power
    
    return jnp.where(step < warmup_steps, warmup_lr, poly_lr)
