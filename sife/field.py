import jax
import jax.numpy as jnp
from typing import Tuple, NamedTuple, Optional
from jax import grad, jit, vmap

Array = jnp.ndarray

class SIFEConfig(NamedTuple):
    mass: float = 1.0
    stiffness: float = 1.0
    alpha: float = 0.25
    beta: float = 1.0
    gamma: float = 0.1
    lambda_res: float = 0.5
    omega_0: float = 2 * jnp.pi / 360.0
    damping: float = 0.01
    leakage: float = 0.001

class SIFField(NamedTuple):
    amplitude: Array
    phase: Array
    fluctuation: Array
    velocity_amp: Array
    velocity_phi: Array

    @property
    def complex_field(self) -> Array:
        return self.amplitude * jnp.exp(1j * self.phase)

def initialize_field(key: Array, shape: Tuple[int, ...], init_scale: float = 0.1) -> SIFField:
    key1, key2 = jax.random.split(key)
    amplitude = jnp.abs(init_scale * (1.0 + 0.5 * jax.random.normal(key1, shape)))
    phase = 2 * jnp.pi * jax.random.uniform(key2, shape)
    return SIFField(
        amplitude=amplitude,
        phase=phase,
        fluctuation=phase,
        velocity_amp=jnp.zeros(shape),
        velocity_phi=jnp.zeros(shape)
    )

def discrete_gradient(field: Array) -> Tuple[Array, ...]:
    ndim = field.ndim
    if ndim == 1:
        return ((jnp.roll(field, -1) - jnp.roll(field, 1)) / 2.0,)
    elif ndim == 2:
        return (
            (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0)) / 2.0,
            (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / 2.0
        )
    elif ndim == 3:
        return ((jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1)) / 2.0,)
    raise ValueError(f"Unsupported ndim: {ndim}")

def discrete_laplacian(field: Array) -> Array:
    ndim = field.ndim
    if ndim == 1:
        return jnp.roll(field, -1) + jnp.roll(field, 1) - 2 * field
    elif ndim == 2:
        return (jnp.roll(field, -1, axis=0) + jnp.roll(field, 1, axis=0) +
                jnp.roll(field, -1, axis=1) + jnp.roll(field, 1, axis=1) - 4 * field)
    elif ndim == 3:
        return jnp.roll(field, -1, axis=1) + jnp.roll(field, 1, axis=1) - 2 * field
    raise ValueError(f"Unsupported ndim: {ndim}")

def discrete_phase_laplacian(field: Array) -> Array:
    ndim = field.ndim
    if ndim == 1:
        d_prev = ((jnp.roll(field, 1) - field + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        d_next = ((jnp.roll(field, -1) - field + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        return d_prev + d_next
    elif ndim == 2:
        lap = 0.0
        for axis in [0, 1]:
            for shift in [1, -1]:
                d_neighbor = ((jnp.roll(field, shift, axis=axis) - field + jnp.pi) % (2 * jnp.pi)) - jnp.pi
                lap = lap + d_neighbor
        return lap
    elif ndim == 3:
        d_prev = ((jnp.roll(field, 1, axis=1) - field + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        d_next = ((jnp.roll(field, -1, axis=1) - field + jnp.pi) % (2 * jnp.pi)) - jnp.pi
        return d_prev + d_next
    raise ValueError(f"Unsupported ndim: {ndim}")

def truth_potential(amplitude: Array, phase: Array) -> Array:
    amp_sq = amplitude ** 2
    ndim = amplitude.ndim
    if ndim == 1:
        amp_sq_right = jnp.roll(amp_sq, -1)
        phase_right = jnp.roll(phase, -1)
        weight = (2 * amp_sq * amp_sq_right) / (amp_sq + amp_sq_right + 1e-8)
        coherence = jnp.cos(phase - phase_right)
        return jnp.mean(weight * coherence)
    elif ndim == 2:
        phi_T = 0.0
        for axis in [0, 1]:
            amp_sq_neighbor = jnp.roll(amp_sq, -1, axis=axis)
            phase_neighbor = jnp.roll(phase, -1, axis=axis)
            weight = (2 * amp_sq * amp_sq_neighbor) / (amp_sq + amp_sq_neighbor + 1e-8)
            coherence = jnp.cos(phase - phase_neighbor)
            phi_T = phi_T + jnp.mean(weight * coherence)
        return phi_T
    elif ndim == 3:
        amp_sq_right = jnp.roll(amp_sq, -1, axis=1)
        phase_right = jnp.roll(phase, -1, axis=1)
        weight = (2 * amp_sq * amp_sq_right) / (amp_sq + amp_sq_right + 1e-8)
        coherence = jnp.cos(phase - phase_right)
        return jnp.mean(weight * coherence)
    raise ValueError(f"Unsupported ndim: {ndim}")

def truth_potential_gradient(amplitude: Array, phase: Array) -> Tuple[Array, Array]:
    def phi_T_amp(A):
        return truth_potential(A, phase)
    def phi_T_phase(theta):
        return truth_potential(amplitude, theta)
    return grad(phi_T_amp)(amplitude), grad(phi_T_phase)(phase)

def sife_equations(field: SIFField, config: SIFEConfig) -> Tuple[Array, Array]:
    A = field.amplitude
    phi = field.fluctuation
    v_A = field.velocity_amp
    v_phi = field.velocity_phi
    
    lap_A = discrete_laplacian(A)
    lap_phi = discrete_phase_laplacian(phi)
    
    # Double-well potential gradient: V(A) = alpha*A^4 - beta*A^2 -> dV/dA = 4*alpha*A^3 - 2*beta*A
    dw_grad = 4 * config.alpha * A**3 - 2 * config.beta * A
    
    phi_T_grad_A, phi_T_grad_phi = truth_potential_gradient(A, phi)
    
    # Resonance coupling
    psi = A * jnp.exp(1j * phi)
    total_coupling = jnp.sum(psi)
    resonance_real = config.lambda_res * jnp.real(total_coupling * jnp.exp(-1j * phi))
    resonance_imag = config.lambda_res * jnp.imag(total_coupling * jnp.exp(-1j * phi))
    
    rhs_A = config.stiffness * lap_A - dw_grad - config.gamma * phi_T_grad_A + resonance_real
    rhs_phi = config.stiffness * A * lap_phi - config.gamma * phi_T_grad_phi + resonance_imag
    
    rhs_A = rhs_A - config.damping * v_A - config.leakage * A
    rhs_phi = rhs_phi - config.damping * v_phi
    
    accel_A = rhs_A / config.mass + A * (config.omega_0 + v_phi)**2
    A_safe = jnp.where(A > 1e-8, A, 1e-8)
    accel_phi = rhs_phi / (config.mass * A_safe) - 2 * v_A * (config.omega_0 + v_phi) / A_safe
    
    return accel_A, accel_phi

@jit
def leapfrog_step(field: SIFField, config: SIFEConfig, dt: float = 0.01) -> SIFField:
    accel_A, accel_phi = sife_equations(field, config)
    
    v_A_half = field.velocity_amp + 0.5 * dt * accel_A
    v_phi_half = field.velocity_phi + 0.5 * dt * accel_phi
    
    A_new = field.amplitude + dt * v_A_half
    phi_new = field.fluctuation + dt * v_phi_half
    theta_new = (field.phase + config.omega_0 * dt + dt * v_phi_half) % (2 * jnp.pi)
    phi_new = ((phi_new + jnp.pi) % (2 * jnp.pi)) - jnp.pi
    A_new = jnp.abs(A_new)
    
    field_half = SIFField(
        amplitude=A_new, phase=theta_new, fluctuation=phi_new,
        velocity_amp=v_A_half, velocity_phi=v_phi_half
    )
    
    accel_A_new, accel_phi_new = sife_equations(field_half, config)
    
    v_A_new = v_A_half + 0.5 * dt * accel_A_new
    v_phi_new = v_phi_half + 0.5 * dt * accel_phi_new
    
    return SIFField(
        amplitude=A_new, phase=theta_new, fluctuation=phi_new,
        velocity_amp=v_A_new, velocity_phi=v_phi_new
    )

def compute_hamiltonian(field: SIFField, config: SIFEConfig) -> Array:
    A = field.amplitude
    phi = field.fluctuation
    v_A = field.velocity_amp
    v_phi = field.velocity_phi
    
    # Use mean instead of sum to ensure scale-invariance and prevent loss explosion
    T = 0.5 * config.mass * jnp.mean(v_A**2 + A**2 * v_phi**2)
    V_cent = -0.5 * config.mass * jnp.mean(A**2 * config.omega_0**2)
    
    psi = A * jnp.exp(1j * phi)
    if A.ndim == 1:
        grad_psi = discrete_gradient(psi)[0]
        V_grad = 0.5 * config.stiffness * jnp.mean(jnp.abs(grad_psi)**2)
    elif A.ndim == 3:
        # For B, L, D tensors
        grads = discrete_gradient(psi)
        V_grad = 0.5 * config.stiffness * jnp.mean(jnp.stack([jnp.mean(jnp.abs(g)**2) for g in grads]))
    else:
        grads = discrete_gradient(psi)
        V_grad = 0.5 * config.stiffness * sum(jnp.mean(jnp.abs(g)**2) for g in grads)
        
    V_dw = jnp.mean(config.alpha * A**4 - config.beta * A**2)
    V_truth = config.gamma * truth_potential(A, phi)
    
    return T + V_grad + V_dw + V_truth + V_cent

batch_initialize_field = vmap(initialize_field, in_axes=(0, None, None))
batch_compute_hamiltonian = vmap(compute_hamiltonian, in_axes=(0, None))
