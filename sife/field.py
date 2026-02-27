"""
SIFE-LDM: Core Field Equations
==============================

Implements the Structured Intelligence Field Equation (SIFE) based on
Lagrangian-Hamiltonian mechanics with a teleological Truth Potential.

The field Ψ_n(t) ∈ ℂ is decomposed into:
- Amplitude A_n(t) (Pathos): represents salience/importance
- Phase θ_n(t) (Logos): represents logical relations

Author: SIFE-LDM Research Team
License: MIT
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from functools import partial
from typing import Tuple, Optional, NamedTuple

# Type aliases
Array = jnp.ndarray
PRNGKey = jnp.ndarray


class SIFEConfig(NamedTuple):
    """Configuration parameters for SIFE field dynamics."""
    m: float = 1.0          # Mass parameter (inertia)
    k: float = 1.0          # Coupling strength (spatial)
    alpha: float = 0.25     # Quartic potential coefficient
    beta: float = 1.0       # Quadratic potential coefficient
    gamma: float = 0.1      # Truth potential strength
    lambda_res: float = 0.5 # Resonance coupling strength
    h: float = 1.0          # Lattice spacing
    dt: float = 0.01        # Time step for integration
    omega_0: float = 2 * jnp.pi / 360 # Standardised background rotation (1 deg/s)
    # AGI Context Dynamics
    m_ctx: float = 10.0      # Context field mass (larger = slower)
    k_ctx: float = 0.5       # Context field stiffness (smaller = more global)
    eta_leak: float = 0.001  # Energy leakage (forgetting)
    lambda_ctx: float = 0.1  # Coupling strength to context
    eta_damping: float = 0.01 # Velocity damping for stable attractors


class SIFField(NamedTuple):
    """Represents a SIFE field state at a given time."""
    amplitude: Array      # A_n: Real amplitude at each lattice node
    phase: Array          # θ_n: Total phase at each lattice node
    fluctuation: Array    # φ_n: Deviation from background rotation
    velocity_amp: Array   # dA_n/dt: Rate of change of amplitude
    velocity_phi: Array   # dφ_n/dt: Rate of change of fluctuation
    
    @property
    def complex_field(self) -> Array:
        """Convert to complex representation Ψ = A * exp(i*θ)"""
        return self.amplitude * jnp.exp(1j * self.phase)
    
    @property
    def field_shape(self) -> Tuple[int, ...]:
        """Return the shape of the field."""
        return self.amplitude.shape

    def apply_perturbation(self, delta_amp: Array, delta_phi: Array, scale: float = 1.0) -> 'SIFField':
        """Apply an external perturbation (action) to the field."""
        return self._replace(
            amplitude=self.amplitude + scale * delta_amp,
            fluctuation=self.fluctuation + scale * delta_phi,
            # Update total phase accordingly
            phase=(self.phase + scale * delta_phi) % (2 * jnp.pi)
        )


def initialize_field(
    key: PRNGKey,
    shape: Tuple[int, ...],
    init_scale: float = 0.1,
    phase_init: str = 'random'
) -> SIFField:
    """
    Initialize a SIFE field with random amplitude and phase.
    
    Args:
        key: JAX random key
        shape: Shape of the lattice (e.g., (seq_len,) for 1D, (H, W) for 2D)
        init_scale: Scale for amplitude initialization
        phase_init: Phase initialization strategy ('random', 'uniform', 'zeros')
    
    Returns:
        SIFField with initialized state
    """
    key1, key2 = jax.random.split(key)
    
    # Initialize amplitude (positive values)
    amplitude = init_scale * (1.0 + 0.5 * jax.random.normal(key1, shape))
    amplitude = jnp.abs(amplitude)  # Ensure positive
    
    # Initialize phase
    if phase_init == 'random':
        phase = 2 * jnp.pi * jax.random.uniform(key2, shape)
    elif phase_init == 'uniform':
        phase = jnp.zeros(shape)
    elif phase_init == 'zeros':
        phase = jnp.zeros(shape)
    else:
        raise ValueError(f"Unknown phase_init: {phase_init}")
    
    return SIFField(
        amplitude=amplitude,
        phase=phase,
        fluctuation=phase,  # Initially, fluctuation is the total phase if t=0
        velocity_amp=jnp.zeros(shape),
        velocity_phi=jnp.zeros(shape)
    )


def discrete_gradient(field: Array, h: float = 1.0) -> Tuple[Array, Array]:
    """
    Compute discrete gradient using central differences.
    
    For 1D: ∇_h ψ_n = (ψ_{n+1} - ψ_{n-1}) / (2h)
    For 2D: Returns gradients along each axis
    
    Args:
        field: Input field (real or complex)
        h: Lattice spacing
    
    Returns:
        Tuple of gradients along each axis
    """
    ndim = field.ndim
    
    if ndim == 1:
        # 1D case: central difference with periodic boundary
        grad_field = (jnp.roll(field, -1) - jnp.roll(field, 1)) / (2 * h)
        return (grad_field,)
    elif ndim == 2:
        # 2D case: gradients along rows and columns
        grad_y = (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0)) / (2 * h)
        grad_x = (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / (2 * h)
        return (grad_y, grad_x)
    else:
        raise ValueError(f"Unsupported field dimension: {ndim}")


def discrete_laplacian(field: Array, h: float = 1.0) -> Array:
    """
    Compute discrete Laplacian using finite differences.
    
    ∇²ψ_n = Σ_{neighbors} (ψ_neighbor - ψ_n) / h²
    
    Args:
        field: Input field
        h: Lattice spacing
    
    Returns:
        Discrete Laplacian of the field
    """
    ndim = field.ndim
    
    if ndim == 1:
        # 1D: d²ψ/dx² ≈ (ψ_{n+1} + ψ_{n-1} - 2ψ_n) / h²
        laplacian = (jnp.roll(field, -1) + jnp.roll(field, 1) - 2 * field) / (h ** 2)
    elif ndim == 2:
        # 2D: Include all 4 neighbors
        laplacian = (
            jnp.roll(field, -1, axis=0) + jnp.roll(field, 1, axis=0) +
            jnp.roll(field, -1, axis=1) + jnp.roll(field, 1, axis=1) -
            4 * field
        ) / (h ** 2)
    else:
        raise ValueError(f"Unsupported field dimension: {ndim}")
    
    return laplacian


def discrete_phase_laplacian(field: Array, h: float = 1.0) -> Array:
    """
    Compute discrete Laplacian for a periodic phase field.
    
    Handles 2π wrapping by using the shortest angular distance.
    """
    ndim = field.ndim
    
    if ndim == 1:
        # Shortest angular distances to neighbors
        d_prev = ((jnp.roll(field, 1) - field + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        d_next = ((jnp.roll(field, -1) - field + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        laplacian = (d_prev + d_next) / (h ** 2)
    elif ndim == 2:
        laplacian = 0.0
        for axis in [0, 1]:
            for shift in [1, -1]:
                d_neighbor = ((jnp.roll(field, shift, axis=axis) - field + jnp.pi) % (2 * jnp.pi)) - jnp.pi
                laplacian = laplacian + d_neighbor
        laplacian = laplacian / (h ** 2)
    else:
        raise ValueError(f"Unsupported field dimension: {ndim}")
    
    return laplacian


def truth_potential(
    amplitude: Array,
    phase: Array,
    h: float = 1.0
) -> Array:
    """
    Compute the Truth Potential Φ_T.
    
    Φ_T = Σ_{⟨i,j⟩} (2|Ψ_i|²|Ψ_j|² / (|Ψ_i|² + |Ψ_j|²)) * cos(θ_i - θ_j)
    
    This rewards phase coherence between neighboring nodes.
    
    Args:
        amplitude: Amplitude field A_n
        phase: Phase field θ_n
        h: Lattice spacing
    
    Returns:
        Scalar truth potential value
    """
    # Compute amplitude squared
    amp_sq = amplitude ** 2
    
    ndim = amplitude.ndim
    
    if ndim == 1:
        # Pairs with right neighbors only (to avoid double counting)
        amp_sq_right = jnp.roll(amp_sq, -1)
        phase_right = jnp.roll(phase, -1)
        
        # Weight factor
        weight = (2 * amp_sq * amp_sq_right) / (amp_sq + amp_sq_right + 1e-8)
        
        # Phase coherence term
        coherence = jnp.cos(phase - phase_right)
        
        # Sum over all bonds
        phi_T = jnp.sum(weight * coherence)
        
    elif ndim == 2:
        # 2D: Sum over horizontal and vertical bonds
        phi_T = 0.0
        
        for axis in [0, 1]:
            amp_sq_neighbor = jnp.roll(amp_sq, -1, axis=axis)
            phase_neighbor = jnp.roll(phase, -1, axis=axis)
            
            weight = (2 * amp_sq * amp_sq_neighbor) / (amp_sq + amp_sq_neighbor + 1e-8)
            coherence = jnp.cos(phase - phase_neighbor)
            
            phi_T = phi_T + jnp.sum(weight * coherence)
    else:
        raise ValueError(f"Unsupported dimension: {ndim}")
    
    return phi_T


def truth_potential_gradient(
    amplitude: Array,
    phase: Array,
    h: float = 1.0
) -> Tuple[Array, Array]:
    """
    Compute gradients of the Truth Potential with respect to amplitude and phase.
    
    Args:
        amplitude: Amplitude field
        phase: Phase field
        h: Lattice spacing
    
    Returns:
        Tuple of (∂Φ_T/∂A, ∂Φ_T/∂θ)
    """
    # Use automatic differentiation
    def phi_T_amp(A):
        return truth_potential(A, phase, h)
    
    def phi_T_phase(theta):
        return truth_potential(amplitude, theta, h)
    
    grad_amp = grad(phi_T_amp)(amplitude)
    grad_phase = grad(phi_T_phase)(phase)
    
    return grad_amp, grad_phase


def double_well_potential(amplitude: Array, alpha: float, beta: float) -> Array:
    """
    Compute the quartic double-well potential.
    
    V(A) = α|A|⁴ - β|A|²
    
    Creates two stable attractors at A = ±√(β/2α) (for appropriate parameters).
    
    Args:
        amplitude: Amplitude field
        alpha: Quartic coefficient
        beta: Quadratic coefficient
    
    Returns:
        Potential energy
    """
    amp_sq = amplitude ** 2
    return jnp.sum(alpha * amp_sq ** 2 - beta * amp_sq)


def double_well_gradient(amplitude: Array, alpha: float, beta: float) -> Array:
    """
    Gradient of the double-well potential.
    
    ∂V/∂A = 4αA³ - 2βA
    
    Args:
        amplitude: Amplitude field
        alpha: Quartic coefficient
        beta: Quadratic coefficient
    
    Returns:
        Gradient with respect to amplitude
    """
    return 4 * alpha * amplitude ** 3 - 2 * beta * amplitude


def resonance_coupling(
    amplitude: Array,
    phase: Array,
    lambda_res: float
) -> Tuple[Array, Array]:
    """
    Compute the non-local resonance coupling term.
    
    λ Σ_j exp(i(θ_j - θ_n)) Ψ_j = λ Σ_j A_j exp(i(θ_j - θ_n + θ_j))
    
    This implements meaningful interaction between field nodes,
    reinforcing configurations where phases are aligned.
    
    Args:
        amplitude: Amplitude field
        phase: Phase field
        lambda_res: Resonance coupling strength
    
    Returns:
        Tuple of (real part, imaginary part) contributions to dynamics
    """
    # Complex field
    psi = amplitude * jnp.exp(1j * phase)
    
    # Sum over all nodes (global coupling)
    # In practice, this can be made local or semi-local
    total_coupling = jnp.sum(psi)
    
    # Phase difference term
    coupling_real = lambda_res * jnp.real(total_coupling * jnp.exp(-1j * phase))
    coupling_imag = lambda_res * jnp.imag(total_coupling * jnp.exp(-1j * phase))
    
    return coupling_real, coupling_imag


def sife_equations(
    field: SIFField,
    config: SIFEConfig,
    context_field: Optional[Array] = None
) -> Tuple[Array, Array]:
    """
    Compute the SIFE equations of motion for amplitude and phase.
    
    Includes AGI augmentations:
    - Velocity damping (eta_damping)
    - Global context coupling (lambda_ctx)
    - Energy leakage (eta_leak)
    """
    A = field.amplitude
    phi = field.fluctuation
    v_A = field.velocity_amp
    v_phi = field.velocity_phi
    
    # Extract parameters
    m, k, h = config.m, config.k, config.h
    alpha, beta, gamma = config.alpha, config.beta, config.gamma
    lambda_res = config.lambda_res
    
    # Discrete Laplacian
    lap_A = discrete_laplacian(A, h)
    lap_phi = discrete_phase_laplacian(phi, h)
    
    # Double-well potential gradient
    dw_grad = double_well_gradient(A, alpha, beta)
    
    # Truth potential gradients (depend only on phase differences)
    phi_T_grad_A, phi_T_grad_phi = truth_potential_gradient(A, phi, h)
    
    # Resonance coupling (simplified to local neighbors)
    resonance_real, resonance_imag = local_resonance(A, phi, lambda_res)
    
    # Base RHS
    rhs_A = (k * lap_A - dw_grad - gamma * phi_T_grad_A + resonance_real)
    
    # Weight phase Laplacian by A to avoid singularity at A=0
    # Phase moves only where there is amplitude
    rhs_phi = (k * A * lap_phi - gamma * phi_T_grad_phi + resonance_imag)
    
    # AGI: Context Coupling
    if context_field is not None:
        ctx_A = jnp.abs(context_field)
        ctx_theta = jnp.angle(context_field)
        
        # Interaction term: lambda * A_ctx * cos(theta_ctx - theta)
        # Weight by local A to preserve wave physics
        coupling_A = config.lambda_ctx * ctx_A * jnp.cos(ctx_theta - field.phase)
        coupling_phi = config.lambda_ctx * A * ctx_A * jnp.sin(ctx_theta - field.phase)
        
        rhs_A = rhs_A + coupling_A
        rhs_phi = rhs_phi + coupling_phi
        
    # AGI: Damping (Stability)
    rhs_A = rhs_A - config.eta_damping * v_A
    rhs_phi = rhs_phi - config.eta_damping * v_phi
    
    # AGI: Leakage (Forgetting)
    rhs_A = rhs_A - config.eta_leak * A
    
    # Acceleration calculation
    accel_A = rhs_A / m + A * (config.omega_0 + v_phi) ** 2
    
    A_safe = jnp.where(A > 1e-8, A, 1e-8)
    accel_phi = rhs_phi / (m * A_safe) - 2 * v_A * (config.omega_0 + v_phi) / A_safe
    
    return accel_A, accel_phi


def local_resonance(
    amplitude: Array,
    phase: Array,
    lambda_res: float,
    h: float = 1.0
) -> Tuple[Array, Array]:
    """
    Compute local resonance coupling (sum over neighbors).
    
    More efficient than global coupling for large fields.
    
    Args:
        amplitude: Amplitude field
        phase: Phase field
        lambda_res: Resonance coupling strength
        h: Lattice spacing
    
    Returns:
        Tuple of (real, imaginary) contributions
    """
    ndim = amplitude.ndim
    
    if ndim == 1:
        # Sum over left and right neighbors
        A_left, A_right = jnp.roll(amplitude, 1), jnp.roll(amplitude, -1)
        theta_left, theta_right = jnp.roll(phase, 1), jnp.roll(phase, -1)
        
        # cos(θ_j - θ_n) * A_j
        real_sum = (
            A_left * jnp.cos(theta_left - phase) +
            A_right * jnp.cos(theta_right - phase)
        )
        
        # sin(θ_j - θ_n) * A_j
        imag_sum = (
            A_left * jnp.sin(theta_left - phase) +
            A_right * jnp.sin(theta_right - phase)
        )
        
    elif ndim == 2:
        # Sum over all 4 neighbors
        real_sum = 0.0
        imag_sum = 0.0
        
        for axis in [0, 1]:
            for shift in [1, -1]:
                A_neighbor = jnp.roll(amplitude, shift, axis=axis)
                theta_neighbor = jnp.roll(phase, shift, axis=axis)
                
                real_sum = real_sum + A_neighbor * jnp.cos(theta_neighbor - phase)
                imag_sum = imag_sum + A_neighbor * jnp.sin(theta_neighbor - phase)
    else:
        raise ValueError(f"Unsupported dimension: {ndim}")
    
    return lambda_res * real_sum, lambda_res * imag_sum


@partial(jit, static_argnums=(2,))
def leapfrog_step(
    field: SIFField,
    config: SIFEConfig,
    dt: Optional[float] = None,
    context_field: Optional[Array] = None
) -> SIFField:
    """
    Perform one leapfrog integration step.
    
    The leapfrog method is symplectic and conserves energy well:
    1. v(t + dt/2) = v(t) + (dt/2) * a(t)
    2. x(t + dt) = x(t) + dt * v(t + dt/2)
    3. v(t + dt) = v(t + dt/2) + (dt/2) * a(t + dt)
    
    Args:
        field: Current field state
        config: SIFE configuration
        dt: Time step (uses config.dt if not provided)
    
    Returns:
        Updated field state
    """
    if dt is None:
        dt = config.dt
    
    # Get accelerations at current position
    accel_A, accel_phi = sife_equations(field, config, context_field=context_field)
    
    # Half-step velocity update
    v_A_half = field.velocity_amp + 0.5 * dt * accel_A
    v_phi_half = field.velocity_phi + 0.5 * dt * accel_phi
    
    # Full-step position update
    A_new = field.amplitude + dt * v_A_half
    phi_new = field.fluctuation + dt * v_phi_half
    
    # Total phase update: θ(t+dt) = ω_0(t+dt) + φ(t+dt)
    # We assume t is tracked by the caller or we can just increment relative to previous theta
    # More robustly: increment theta by ω_0*dt + Δφ
    theta_new = (field.phase + config.omega_0 * dt + dt * v_phi_half) % (2 * jnp.pi)
    
    # Wrap fluctuation to [-π, π] for stability
    phi_new = ((phi_new + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    
    # Ensure amplitude stays positive
    A_new = jnp.abs(A_new)
    
    # Create intermediate field for acceleration calculation
    field_half = SIFField(
        amplitude=A_new,
        phase=theta_new,
        fluctuation=phi_new,
        velocity_amp=v_A_half,
        velocity_phi=v_phi_half
    )
    
    # Get accelerations at new position
    accel_A_new, accel_phi_new = sife_equations(field_half, config, context_field=context_field)
    
    # Complete velocity update
    v_A_new = v_A_half + 0.5 * dt * accel_A_new
    v_phi_new = v_phi_half + 0.5 * dt * accel_phi_new
    
    return SIFField(
        amplitude=A_new,
        phase=theta_new,
        fluctuation=phi_new,
        velocity_amp=v_A_new,
        velocity_phi=v_phi_new
    )


def evolve_field(
    field: SIFField,
    config: SIFEConfig,
    num_steps: int,
    context_field: Optional[Array] = None
) -> SIFField:
    """
    Evolve the field for multiple time steps.
    
    Args:
        field: Initial field state
        config: SIFE configuration
        num_steps: Number of time steps
    
    Returns:
        Evolved field state
    """
    def step_fn(f, _):
        return leapfrog_step(f, config, context_field=context_field), None
    
    final_field, _ = jax.lax.scan(step_fn, field, None, length=num_steps)
    return final_field


def compute_hamiltonian(
    field: SIFField,
    config: SIFEConfig
) -> Array:
    """
    Compute the Hamiltonian (total energy) of the field.
    
    H = T + V = Σ_n [m/2|Ψ̇_n|² + k/2|∇Ψ_n|² + V(Ψ_n)]
    
    Args:
        field: Field state
        config: SIFE configuration
    
    Returns:
        Total energy (scalar)
    """
    A = field.amplitude
    phi = field.fluctuation
    v_A = field.velocity_amp
    v_phi = field.velocity_phi
    # The conserved quantity in a rotating frame is the Jacobi invariant:
    # E = 1/2 m [Ȧ² + A²φ̇² - A²ω_0²] + V
    
    # Kinetic energy in rotating frame
    T = 0.5 * config.m * jnp.sum(v_A ** 2 + A ** 2 * v_phi ** 2)
    
    # Centrifugal potential energy (correction to V)
    V_cent = -0.5 * config.m * jnp.sum(A ** 2 * config.omega_0 ** 2)
    
    # Gradient energy: k/2 * |∇Ψ|² where Ψ = A exp(iφ)
    # Background rotation ω_0 t cancels in phase differences
    psi = A * jnp.exp(1j * phi)
    if A.ndim == 1:
        grad_psi = discrete_gradient(psi, config.h)[0]
        V_grad = 0.5 * config.k * jnp.sum(jnp.abs(grad_psi) ** 2)
    else:
        grads = discrete_gradient(psi, config.h)
        V_grad = 0.5 * config.k * sum(jnp.sum(jnp.abs(g) ** 2) for g in grads)
    
    # Potential energy: double-well + truth potential
    V_dw = double_well_potential(A, config.alpha, config.beta)
    V_truth = config.gamma * truth_potential(A, phi, config.h)
    
    return T + V_grad + V_dw + V_truth + V_cent


# Vectorized operations for batch processing
batch_initialize_field = vmap(initialize_field, in_axes=(0, None, None, None))
batch_compute_hamiltonian = vmap(compute_hamiltonian, in_axes=(0, None))


def compute_landscape_curvature(
    A: Array,
    config: SIFEConfig,
    dynamic_v: Optional[Array] = None
) -> Array:
    """
    Compute the curvature (second derivative) of the local potential landscape.
    Used for stability regularization to ensure deep attractors.
    """
    # Second derivative of V_dw w.r.t A: -beta + 3*alpha*A^2
    beta = config.beta
    if dynamic_v is not None:
        # Tighter beta for higher dynamic propagation velocity v
        # Reshape dynamic_v from (B, 1) -> (B, 1, ..., 1) to broadcast against A's shape
        extra_dims = A.ndim - dynamic_v.ndim
        dv = dynamic_v.reshape(dynamic_v.shape + (1,) * extra_dims)
        beta = beta * dv
        
    curvature_A = -beta + 3 * config.alpha * A**2
    return curvature_A


def is_field_stable(field: SIFField, config: SIFEConfig, threshold: float = 0.5) -> bool:
    """
    Evaluate if the current field configuration has reached a stable, low-energy state.
    
    This acts as the "System 2" stopping criterion. A stable field implies
    that the generated complex representation is logically and grammatically coherent.
    
    Args:
        field: The current SIFE field state.
        config: The SIFE configuration.
        threshold: The energy threshold per node below which the field is considered stable.
        
    Returns:
        True if the field is stable (average energy < threshold), False otherwise.
    """
    total_energy = compute_hamiltonian(field, config)
    num_nodes = field.amplitude.size
    avg_energy = total_energy / num_nodes
    return avg_energy < threshold

