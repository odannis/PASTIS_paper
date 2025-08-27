import jax.numpy as jnp
from jax import jit

@jit
def laplacian(u, dx=1.0):
    """Compute the Laplacian of a 2D array `u` with periodic boundary conditions."""
    u_xx = (jnp.roll(u, -1, axis=0) - 2 * u + jnp.roll(u, 1, axis=0)) / dx**2
    u_yy = (jnp.roll(u, -1, axis=1) - 2 * u + jnp.roll(u, 1, axis=1)) / dx**2
    return u_xx + u_yy

# lap_u function
@jit
def lap_u(im, dx=1):
    """Compute Laplacian for the 'u' component."""
    re0 = laplacian(im[0], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

# lap_v function
@jit
def lap_v(im, dx=1.0):
    """Compute Laplacian for the 'v' component."""
    re1 = laplacian(im[1], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

# u_v2_for_u function
@jit
def u_v2_for_u(im, dx=1.0):
    """Compute u * v^2 for the 'u' component."""
    re0 = im[0] * im[1] ** 2
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

# u_v2_for_v function
@jit
def u_v2_for_v(im, dx=1.0):
    """Compute u * v^2 for the 'v' component."""
    re1 = im[0] * im[1] ** 2
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

# cste_for_u function
@jit
def cste_for_u(im, dx=1.0):
    """Set the 'u' component to ones."""
    re0 = jnp.ones_like(im[0])
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

# u_for_u function
@jit
def u_for_u(im, dx=1.0):
    """Pass through the 'u' component."""
    re = jnp.zeros_like(im)
    re = re.at[0].set(im[0])
    return re

# u_for_v function
@jit
def u_for_v(im, dx=1.0):
    """Set the 'v' component to 'u'."""
    re = jnp.zeros_like(im)
    re = re.at[1].set(im[0])
    return re

# u3_for_u function
@jit
def u3_for_u(im, dx=1.0):
    """Compute u^3 for the 'u' component."""
    re0 = im[0] ** 3
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

# v_for_v function
@jit
def v_for_v(im, dx=1.0):
    """Pass through the 'v' component."""
    re = jnp.zeros_like(im)
    re = re.at[1].set(im[1])
    return re

# v_for_u function
@jit
def v_for_u(im, dx=1.0):
    """Set the 'u' component to 'v'."""
    re = jnp.zeros_like(im)
    re = re.at[0].set(im[1])
    return re