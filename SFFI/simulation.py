import numpy as np
from jax import numpy as jnp
import jax.numpy as jnp
from jax import jit, random, lax
from tqdm import tqdm
import jax

rng = np.random.default_rng()
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_default_matmul_precision", "F64_F64_F64")
  
# Define a generic sqrt_diffusion_func that accepts a flag for whether sqrt_diffusion is callable
def create_sqrt_diffusion_func(sqrt_diffusion, mult_noise):
    if not callable(sqrt_diffusion):
        sqrt_diffusion = jnp.asarray(sqrt_diffusion, dtype=jnp.float64)
        def sqrt_diffusion_func(x, noise):
            return noise * mult_noise * sqrt_diffusion
    else:
        def sqrt_diffusion_func(x, noise):
            return sqrt_diffusion(x) @ noise * mult_noise
    return sqrt_diffusion_func

# Define the body function outside of simulate if possible.
def create_body_fun(sqrt_diffusion_func, coefficient, base, dt):
    def body_fun(phi_prev, t, noise):
        phi_t = phi_prev + sqrt_diffusion_func(phi_prev, noise[t])
        for coef, func in zip(coefficient, base):
            phi_t += coef * func(phi_prev) * dt
        return phi_t, phi_t
    return body_fun

def simulate(base, coefficient, shape_image: tuple = (2, 50, 50), n=10000, dt=0.001,
             sqrt_diffusion=1, key=None, force_numba=None,
             first_image=None, over_sampling: int = 10, thermalised_time=0,
             diffusion_strength=1):
    num_steps = n
    shape_image = tuple(shape_image)
    thermalization_steps = int(thermalised_time / dt)
    dt = jnp.float64(dt)
    diffusion_strength = jnp.float64(diffusion_strength)
    dt_oversampling = jnp.float64(dt / over_sampling)

    # Prepare the initial condition
    if first_image is not None:
        phi0 = jnp.asarray(first_image, dtype=jnp.float64)
    else:
        phi0 = jnp.zeros(shape_image, dtype=jnp.float64)

    # Set up a key if none provided
    if key is None:
        print("Key is None, using random key")
        key = random.PRNGKey(np.random.randint(0, 2**32 - 1))
    
    mult_noise = jnp.sqrt(2 * diffusion_strength * dt_oversampling)
    sqrt_diffusion_func = create_sqrt_diffusion_func(sqrt_diffusion, mult_noise)
    body_fun = create_body_fun(sqrt_diffusion_func, coefficient, base, dt_oversampling)

    def run_chunk(start, end, phi_init, noise, over_sampling):
        def loop_body(phi, t):
            return body_fun(phi, t, noise)
        phi_init = jnp.asarray(phi_init, dtype=jnp.float64)
        _, phi_chunk = lax.scan(loop_body, phi_init, jnp.arange(start, end))
        phi_out = phi_chunk[over_sampling-1::over_sampling]
        phi_chunk = np.insert(phi_out, 0, phi_init, axis=0)
        return phi_chunk

    # noise_shape = (total_steps,) + shape_image
    # #noise = jnp.asarray(rng.normal(size=noise_shape), dtype=jnp.float64) #jnp.zeros((total_steps,) + shape_image, dtype=jnp.float64)
    # key, subkey = random.split(key)
    # noise = random.normal(subkey, shape=noise_shape, dtype=jnp.float64)
    # _, phi_all = run_chunk(0, total_steps, phi0, noise=noise)
    # phi_all = phi_all[::over_sampling]
    
    total_steps = int(num_steps + thermalization_steps)
    shape_phi = (total_steps,) + shape_image
    phi_all = np.empty(shape_phi)
    phi_current = phi0
    keys = random.split(key, over_sampling+1)
    max_chunk_size = 0
    for i in range(over_sampling):
        if diffusion_strength != 0:
            noise_shape = shape_phi
            noise = random.normal(keys[i+1], shape=noise_shape, dtype=jnp.float64)
            #noise = jnp.float64(rng.normal(size=noise_shape, ))
        else:
            noise = jnp.zeros(shape_phi)
        phi_chunk = run_chunk(0, total_steps, phi_current, noise, over_sampling)
        phi_current = phi_chunk[-1]
        phi_saved = phi_chunk[:-1]
        chunk_size = phi_saved.shape[0]
        phi_all[i * chunk_size:(i + 1) * chunk_size] = phi_saved
        max_chunk_size = max(max_chunk_size, (i + 1) *chunk_size)
    phi_all = phi_all[:max_chunk_size]
    out_phi = np.asarray(phi_all[thermalization_steps:])
    return out_phi, float(dt)


def simulate_batch(base, coefficient, first_image, size_batch = 1000,
             shape_output = (1000, 1, 50, 50), **parameters):
    if size_batch <= 1:
        parameters["shape_output"] = shape_output
        parameters["first_image"] = first_image
        return simulate(base, coefficient, **parameters)
    dt = parameters["dt"]
    phi_output = np.zeros(shape=shape_output)
    phi_output[0] = first_image
    parameters["shape_output"] = (size_batch, *shape_output[1:])
    for i in tqdm (range(phi_output.shape[0]-1), desc="Loading..."):
        parameters["first_image"] = phi_output[i]
        phi, dt = simulate(base, coefficient, **parameters)
        phi_output[i+1] = phi[-1]
    return phi_output, dt*(size_batch-1)