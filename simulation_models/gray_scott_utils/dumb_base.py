import jax.numpy as jnp
from jax import jit

# Laplacian function with periodic boundary conditions
@jit
def laplacian(u, dx=1.0):
    """Compute the Laplacian of a 2D array u with periodic boundary conditions."""
    u_xx = (jnp.roll(u, -1, axis=0) - 2 * u + jnp.roll(u, 1, axis=0)) / dx**2
    u_yy = (jnp.roll(u, -1, axis=1) - 2 * u + jnp.roll(u, 1, axis=1)) / dx**2
    return u_xx + u_yy

# Laplacian functions for 'u' and 'v' components
@jit
def lap_u(im, dx=1.0):
    """Compute Laplacian for the 'u' component."""
    re0 = laplacian(im[0], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def lap_v(im, dx=1.0):
    """Compute Laplacian for the 'v' component."""
    re1 = laplacian(im[1], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def lap_u_v(im, dx=1.0):
    """Compute Laplacian for the 'u' component."""
    re0 = laplacian(im[0], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re0)
    return re

@jit
def lap_v_u(im, dx=1.0):
    """Compute Laplacian for the 'v' component."""
    re1 = laplacian(im[1], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re1)
    return re

# Polynomial functions up to order 4

# Degree 0
@jit
def cste_for_u(im, dx=1.0):
    """Set the 'u' component to ones."""
    re0 = jnp.ones_like(im[0])
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def cste_for_v(im, dx=1.0):
    """Set the 'v' component to ones."""
    re1 = jnp.ones_like(im[1])
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

# Degree 1
@jit
def u_for_u(im, dx=1.0):
    """Assign 'u' to the 'u' component."""
    re = jnp.zeros_like(im)
    re = re.at[0].set(im[0])
    return re

@jit
def v_for_v(im, dx=1.0):
    """Assign 'v' to the 'v' component."""
    re = jnp.zeros_like(im)
    re = re.at[1].set(im[1])
    return re

@jit
def u_for_v(im, dx=1.0):
    """Assign 'u' to the 'v' component."""
    re = jnp.zeros_like(im)
    re = re.at[1].set(im[0])
    return re

@jit
def v_for_u(im, dx=1.0):
    """Assign 'v' to the 'u' component."""
    re = jnp.zeros_like(im)
    re = re.at[0].set(im[1])
    return re

# Degree 2
@jit
def u_squared_for_u(im, dx=1.0):
    """Compute u^2 for the 'u' component."""
    re0 = im[0] ** 2
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def v_squared_for_v(im, dx=1.0):
    """Compute v^2 for the 'v' component."""
    re1 = im[1] ** 2
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_squared_for_v(im, dx=1.0):
    """Compute u^2 for the 'v' component."""
    re1 = im[0] ** 2
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def v_squared_for_u(im, dx=1.0):
    """Compute v^2 for the 'u' component."""
    re0 = im[1] ** 2
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def uv_for_u(im, dx=1.0):
    """Compute u * v for the 'u' component."""
    re0 = im[0] * im[1]
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def uv_for_v(im, dx=1.0):
    """Compute u * v for the 'v' component."""
    re1 = im[0] * im[1]
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

# Degree 3
@jit
def u_cubed_for_u(im, dx=1.0):
    """Compute u^3 for the 'u' component."""
    re0 = im[0] ** 3
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_cubed_for_v(im, dx=1.0):
    """Compute u^3 for the 'v' component."""
    re1 = im[0] ** 3
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_squared_v_for_u(im, dx=1.0):
    """Compute u^2 * v for the 'u' component."""
    re0 = (im[0] ** 2) * im[1]
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_squared_v_for_v(im, dx=1.0):
    """Compute u^2 * v for the 'v' component."""
    re1 = (im[0] ** 2) * im[1]
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_v_squared_for_u(im, dx=1.0):
    """Compute u * v^2 for the 'u' component."""
    re0 = im[0] * (im[1] ** 2)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_v_squared_for_v(im, dx=1.0):
    """Compute u * v^2 for the 'v' component."""
    re1 = im[0] * (im[1] ** 2)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def v_cubed_for_u(im, dx=1.0):
    """Compute v^3 for the 'u' component."""
    re0 = im[1] ** 3
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def v_cubed_for_v(im, dx=1.0):
    """Compute v^3 for the 'v' component."""
    re1 = im[1] ** 3
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

# Degree 4
@jit
def u_fourth_for_u(im, dx=1.0):
    """Compute u^4 for the 'u' component."""
    re0 = im[0] ** 4
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_fourth_for_v(im, dx=1.0):
    """Compute u^4 for the 'v' component."""
    re1 = im[0] ** 4
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_cubed_v_for_u(im, dx=1.0):
    """Compute u^3 * v for the 'u' component."""
    re0 = (im[0] ** 3) * im[1]
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_cubed_v_for_v(im, dx=1.0):
    """Compute u^3 * v for the 'v' component."""
    re1 = (im[0] ** 3) * im[1]
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_squared_v_squared_for_u(im, dx=1.0):
    """Compute u^2 * v^2 for the 'u' component."""
    re0 = (im[0] ** 2) * (im[1] ** 2)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_squared_v_squared_for_v(im, dx=1.0):
    """Compute u^2 * v^2 for the 'v' component."""
    re1 = (im[0] ** 2) * (im[1] ** 2)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_v_cubed_for_u(im, dx=1.0):
    """Compute u * v^3 for the 'u' component."""
    re0 = im[0] * (im[1] ** 3)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_v_cubed_for_v(im, dx=1.0):
    """Compute u * v^3 for the 'v' component."""
    re1 = im[0] * (im[1] ** 3)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def v_fourth_for_u(im, dx=1.0):
    """Compute v^4 for the 'u' component."""
    re0 = im[1] ** 4
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def v_fourth_for_v(im, dx=1.0):
    """Compute v^4 for the 'v' component."""
    re1 = im[1] ** 4
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

# Updated function list with all polynomial terms up to degree 4
function_list = [
    # Laplacian terms
    lap_u,
    lap_v,
    lap_u_v,
    lap_v_u,
    
    # Polynomial functions up to order 4
    cste_for_u,
    cste_for_v,
    u_for_u,
    v_for_v,
    u_for_v,
    v_for_u,
    u_squared_for_u,
    v_squared_for_v,
    u_squared_for_v,
    v_squared_for_u,
    uv_for_u,
    uv_for_v,
    u_cubed_for_u,
    u_cubed_for_v,
    u_squared_v_for_u,
    u_squared_v_for_v,
    u_v_squared_for_u,
    u_v_squared_for_v,
    v_cubed_for_u,
    v_cubed_for_v,
    
    u_fourth_for_u,
    u_fourth_for_v,
    u_cubed_v_for_u,
    u_cubed_v_for_v,
    u_squared_v_squared_for_u,
    u_squared_v_squared_for_v,
    u_v_cubed_for_u,
    u_v_cubed_for_v,
    v_fourth_for_u,
    v_fourth_for_v,
]

# Updated base_function_symbols as a list of tuples (expression, component)
base_function_symbols = [
    ('\\nabla^2 u', 'u'),        # lap_u
    ('\\nabla^2 v', 'v'),        # lap_v
   ('\\nabla^2 u', 'v'),        # lap_u_v
   ('\\nabla^2 v', 'u'),        # lap_v_u
    ('1', 'u'),                  # cste_for_u
    ('1', 'v'),                  # cste_for_v
    ('u', 'u'),                  # u_for_u
    ('v', 'v'),                  # v_for_v
    ('u', 'v'),                  # u_for_v
    ('v', 'u'),                  # v_for_u
    ('u^2', 'u'),                # u_squared_for_u
    ('v^2', 'v'),                # v_squared_for_v
    ('u^2', 'v'),                # u_squared_for_v
    ('v^2', 'u'),                # v_squared_for_u
    ('u v', 'u'),                # uv_for_u
    ('u v', 'v'),                # uv_for_v
    ('u^3', 'u'),                # u_cubed_for_u
    ('u^3', 'v'),                # u_cubed_for_v
    ('u^2 v', 'u'),              # u_squared_v_for_u
    ('u^2 v', 'v'),              # u_squared_v_for_v
    ('u v^2', 'u'),              # u_v_squared_for_u
    ('u v^2', 'v'),              # u_v_squared_for_v
    ('v^3', 'u'),                # v_cubed_for_u
    ('v^3', 'v'),                # v_cubed_for_v
    
    ('u^4', 'u'),                # u_fourth_for_u
    ('u^4', 'v'),                # u_fourth_for_v
    ('u^3 v', 'u'),              # u_cubed_v_for_u
    ('u^3 v', 'v'),              # u_cubed_v_for_v
    ('u^2 v^2', 'u'),            # u_squared_v_squared_for_u
    ('u^2 v^2', 'v'),            # u_squared_v_squared_for_v
    ('u v^3', 'u'),              # u_v_cubed_for_u
    ('u v^3', 'v'),              # u_v_cubed_for_v
    ('v^4', 'u'),                # v_fourth_for_u
    ('v^4', 'v'),                # v_fourth_for_v
]

@jit
def gradient(u, dx=1.0):
    """Compute the gradient of a 2D array u with periodic boundary conditions."""
    u_x = (jnp.roll(u, 1, axis=0) - u ) / (dx)
    u_y = (jnp.roll(u, 1, axis=1) - u) / (dx)
    return u_x, u_y

# Gradient terms
@jit
def du_dx_for_u(im, dx=1.0):
    """Compute du/dx for the 'u' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[0].set(du_dx)
    return re

@jit
def du_dy_for_u(im, dx=1.0):
    """Compute du/dy for the 'u' component."""
    _, du_dy = gradient(im[0], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[0].set(du_dy)
    return re

@jit
def dv_dx_for_u(im, dx=1.0):
    """Compute dv/dx for the 'u' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[0].set(dv_dx)
    return re

@jit
def dv_dy_for_u(im, dx=1.0):
    """Compute dv/dy for the 'u' component."""
    _, dv_dy = gradient(im[1], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[0].set(dv_dy)
    return re

@jit
def du_dx_for_v(im, dx=1.0):
    """Compute du/dx for the 'v' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[1].set(du_dx)
    return re

@jit
def du_dy_for_v(im, dx=1.0):
    """Compute du/dy for the 'v' component."""
    _, du_dy = gradient(im[0], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[1].set(du_dy)
    return re

@jit
def dv_dx_for_v(im, dx=1.0):
    """Compute dv/dx for the 'v' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[1].set(dv_dx)
    return re

@jit
def dv_dy_for_v(im, dx=1.0):
    """Compute dv/dy for the 'v' component."""
    _, dv_dy = gradient(im[1], dx=dx)
    re = jnp.zeros_like(im)
    re = re.at[1].set(dv_dy)
    return re

# Product terms with gradients
@jit
def u_du_dx_for_u(im, dx=1.0):
    """Compute u * du/dx for the 'u' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    re0 = im[0] * du_dx
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_du_dy_for_u(im, dx=1.0):
    """Compute u * du/dy for the 'u' component."""
    _, du_dy = gradient(im[0], dx=dx)
    re0 = im[0] * du_dy
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_dv_dx_for_u(im, dx=1.0):
    """Compute u * dv/dx for the 'u' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    re0 = im[0] * dv_dx
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_dv_dy_for_u(im, dx=1.0):
    """Compute u * dv/dy for the 'u' component."""
    _, dv_dy = gradient(im[1], dx=dx)
    re0 = im[0] * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def v_du_dx_for_u(im, dx=1.0):
    """Compute v * du/dx for the 'u' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    re0 = im[1] * du_dx
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def v_du_dy_for_u(im, dx=1.0):
    """Compute v * du/dy for the 'u' component."""
    _, du_dy = gradient(im[0], dx=dx)
    re0 = im[1] * du_dy
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def v_dv_dx_for_u(im, dx=1.0):
    """Compute v * dv/dx for the 'u' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    re0 = im[1] * dv_dx
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def v_dv_dy_for_u(im, dx=1.0):
    """Compute v * dv/dy for the 'u' component."""
    _, dv_dy = gradient(im[1], dx=dx)
    re0 = im[1] * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_du_dx_for_v(im, dx=1.0):
    """Compute u * du/dx for the 'v' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    re1 = im[0] * du_dx
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_du_dy_for_v(im, dx=1.0):
    """Compute u * du/dy for the 'v' component."""
    _, du_dy = gradient(im[0], dx=dx)
    re1 = im[0] * du_dy
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_dv_dx_for_v(im, dx=1.0):
    """Compute u * dv/dx for the 'v' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    re1 = im[0] * dv_dx
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_dv_dy_for_v(im, dx=1.0):
    """Compute u * dv/dy for the 'v' component."""
    _, dv_dy = gradient(im[1], dx=dx)
    re1 = im[0] * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def v_du_dx_for_v(im, dx=1.0):
    """Compute v * du/dx for the 'v' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    re1 = im[1] * du_dx
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def v_du_dy_for_v(im, dx=1.0):
    """Compute v * du/dy for the 'v' component."""
    _, du_dy = gradient(im[0], dx=dx)
    re1 = im[1] * du_dy
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def v_dv_dx_for_v(im, dx=1.0):
    """Compute v * dv/dx for the 'v' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    re1 = im[1] * dv_dx
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def v_dv_dy_for_v(im, dx=1.0):
    """Compute v * dv/dy for the 'v' component."""
    _, dv_dy = gradient(im[1], dx=dx)
    re1 = im[1] * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

# Product of gradients
@jit
def du_dx_squared_for_u(im, dx=1.0):
    """Compute (du/dx)^2 for the 'u' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    re0 = du_dx ** 2
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def du_dy_squared_for_u(im, dx=1.0):
    """Compute (du/dy)^2 for the 'u' component."""
    _, du_dy = gradient(im[0], dx=dx)
    re0 = du_dy ** 2
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def dv_dx_squared_for_u(im, dx=1.0):
    """Compute (dv/dx)^2 for the 'u' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    re0 = dv_dx ** 2
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def dv_dy_squared_for_u(im, dx=1.0):
    """Compute (dv/dy)^2 for the 'u' component."""
    _, dv_dy = gradient(im[1], dx=dx)
    re0 = dv_dy ** 2
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def du_dx_du_dy_for_u(im, dx=1.0):
    """Compute (du/dx) * (du/dy) for the 'u' component."""
    du_dx, du_dy = gradient(im[0], dx=dx)
    re0 = du_dx * du_dy
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def du_dx_dv_dx_for_u(im, dx=1.0):
    """Compute (du/dx) * (dv/dx) for the 'u' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    dv_dx, _ = gradient(im[1], dx=dx)
    re0 = du_dx * dv_dx
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def du_dx_dv_dy_for_u(im, dx=1.0):
    """Compute (du/dx) * (dv/dy) for the 'u' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    _, dv_dy = gradient(im[1], dx=dx)
    re0 = du_dx * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def du_dy_dv_dx_for_u(im, dx=1.0):
    """Compute (du/dy) * (dv/dx) for the 'u' component."""
    _, du_dy = gradient(im[0], dx=dx)
    dv_dx, _ = gradient(im[1], dx=dx)
    re0 = du_dy * dv_dx
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def du_dy_dv_dy_for_u(im, dx=1.0):
    """Compute (du/dy) * (dv/dy) for the 'u' component."""
    _, du_dy = gradient(im[0], dx=dx)
    _, dv_dy = gradient(im[1], dx=dx)
    re0 = du_dy * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def dv_dx_dv_dy_for_u(im, dx=1.0):
    """Compute (dv/dx) * (dv/dy) for the 'u' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    _, dv_dy = gradient(im[1], dx=dx)
    re0 = dv_dx * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def du_dx_squared_for_v(im, dx=1.0):
    """Compute (du/dx)^2 for the 'v' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    re1 = du_dx ** 2
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def du_dy_squared_for_v(im, dx=1.0):
    """Compute (du/dy)^2 for the 'v' component."""
    _, du_dy = gradient(im[0], dx=dx)
    re1 = du_dy ** 2
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def dv_dx_squared_for_v(im, dx=1.0):
    """Compute (dv/dx)^2 for the 'v' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    re1 = dv_dx ** 2
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def dv_dy_squared_for_v(im, dx=1.0):
    """Compute (dv/dy)^2 for the 'v' component."""
    _, dv_dy = gradient(im[1], dx=dx)
    re1 = dv_dy ** 2
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def du_dx_du_dy_for_v(im, dx=1.0):
    """Compute (du/dx) * (du/dy) for the 'v' component."""
    du_dx, du_dy = gradient(im[0], dx=dx)
    re1 = du_dx * du_dy
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def du_dx_dv_dx_for_v(im, dx=1.0):
    """Compute (du/dx) * (dv/dx) for the 'v' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    dv_dx, _ = gradient(im[1], dx=dx)
    re1 = du_dx * dv_dx
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def du_dx_dv_dy_for_v(im, dx=1.0):
    """Compute (du/dx) * (dv/dy) for the 'v' component."""
    du_dx, _ = gradient(im[0], dx=dx)
    _, dv_dy = gradient(im[1], dx=dx)
    re1 = du_dx * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def du_dy_dv_dx_for_v(im, dx=1.0):
    """Compute (du/dy) * (dv/dx) for the 'v' component."""
    _, du_dy = gradient(im[0], dx=dx)
    dv_dx, _ = gradient(im[1], dx=dx)
    re1 = du_dy * dv_dx
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def du_dy_dv_dy_for_v(im, dx=1.0):
    """Compute (du/dy) * (dv/dy) for the 'v' component."""
    _, du_dy = gradient(im[0], dx=dx)
    _, dv_dy = gradient(im[1], dx=dx)
    re1 = du_dy * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def dv_dx_dv_dy_for_v(im, dx=1.0):
    """Compute (dv/dx) * (dv/dy) for the 'v' component."""
    dv_dx, _ = gradient(im[1], dx=dx)
    _, dv_dy = gradient(im[1], dx=dx)
    re1 = dv_dx * dv_dy
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

function_list += (
    du_dx_for_u,
    du_dy_for_u,
    dv_dx_for_u,
    dv_dy_for_u,
    du_dx_for_v,
    du_dy_for_v,
    dv_dx_for_v,
    dv_dy_for_v,
    u_du_dx_for_u,
    u_du_dy_for_u,
    u_dv_dx_for_u,
    u_dv_dy_for_u,
    v_du_dx_for_u,
    v_du_dy_for_u,
    v_dv_dx_for_u,
    v_dv_dy_for_u,
    u_du_dx_for_v,
    u_du_dy_for_v,
    u_dv_dx_for_v,
    u_dv_dy_for_v,
    v_du_dx_for_v,
    v_du_dy_for_v,
    v_dv_dx_for_v,
    v_dv_dy_for_v,
    du_dx_squared_for_u,
    du_dy_squared_for_u,
    dv_dx_squared_for_u,
    dv_dy_squared_for_u,
    du_dx_du_dy_for_u,
    du_dx_dv_dx_for_u,
    du_dx_dv_dy_for_u,
    du_dy_dv_dx_for_u,
    du_dy_dv_dy_for_u,
    dv_dx_dv_dy_for_u,
    du_dx_squared_for_v,
    du_dy_squared_for_v,
    dv_dx_squared_for_v,
    dv_dy_squared_for_v,
    du_dx_du_dy_for_v,
    du_dx_dv_dx_for_v,
    du_dx_dv_dy_for_v,
    du_dy_dv_dx_for_v,
    du_dy_dv_dy_for_v,
    dv_dx_dv_dy_for_v,)

base_function_symbols += [
    ('\\partial_x u', 'u'), #du_dx_for_u,
    ('\\partial_y u', 'u'), #du_dy_for_u,
    ('\\partial_x v', 'u'), #dv_dx_for_u,
    ('\\partial_y v', 'u'), #dv_dy_for_u,
    ('\\partial_x u', 'v'), #du_dx_for_v,
    ('\\partial_y u', 'v'), #du_dy_for_v,
    ('\\partial_x v', 'v'), #dv_dx_for_v,
    ('\\partial_y v', 'v'), #dv_dy_for_v,
    ('u \\partial_x u', 'u'), #u_du_dx_for_u,
    ('u \\partial_y u', 'u'), #u_du_dy_for_u,
    ('u \\partial_x v', 'u'), #u_dv_dx_for_u,
    ('u \\partial_y v', 'u'), #u_dv_dy_for_u,
    ('v \\partial_x u', 'u'), #v_du_dx_for_u,
    ('v \\partial_y u', 'u'), #v_du_dy_for_u,
    ('v \\partial_x v', 'u'), #v_dv_dx_for_u,
    ('v \\partial_y v', 'u'), #v_dv_dy_for_u,
    ('u \\partial_x u', 'v'), #u_du_dx_for_v,
    ('u \\partial_y u', 'v'), #u_du_dy_for_v,
    ('u \\partial_x v', 'v'), #u_dv_dx_for_v,
    ('u \\partial_y v', 'v'), #u_dv_dy_for_v,
    ('v \\partial_x u', 'v'), #v_du_dx_for_v,
    ('v \\partial_y u', 'v'), #v_du_dy_for_v,
    ('v \\partial_x v', 'v'), #v_dv_dx_for_v,
    ('v \\partial_y v', 'v'), #v_dv_dy_for_v,
    ('(\\partial_x u)^2', 'u'), #du_dx_squared_for_u,
    ('(\\partial_y u)^2', 'u'), #du_dy_squared_for_u,
    ('(\\partial_x v)^2', 'u'), #dv_dx_squared_for_u,
    ('(\\partial_y v)^2', 'u'), #dv_dy_squared_for_u,
    ('\\partial_x u \\partial_y u', 'u'), #du_dx_du_dy_for_u,
    ('\\partial_x u \\partial_x v', 'u'), #du_dx_dv_dx_for_u,
    ('\\partial_x u \\partial_y v', 'u'), #du_dx_dv_dy_for_u,
    ('\\partial_y u \\partial_x v', 'u'), #du_dy_dv_dx_for_u,
    ('\\partial_y u \\partial_y v', 'u'), #du_dy_dv_dy_for_u,
    ('\\partial_x v \\partial_y v', 'u'), #dv_dx_dv_dy_for_u,
    ('(\\partial_x u)^2', 'v'), #du_dx_squared_for_v,
    ('(\\partial_y u)^2', 'v'), #du_dy_squared_for_v,
    ('(\\partial_x v)^2', 'v'), #dv_dx_squared_for_v,
    ('(\\partial_y v)^2', 'v'), #dv_dy_squared_for_v,
    ('\\partial_x u \\partial_y u', 'v'), #du_dx_du_dy_for_v,
    ('\\partial_x u \\partial_x v', 'v'), #du_dx_dv_dx_for_v,
    ('\\partial_x u \\partial_y v', 'v'), #du_dx_dv_dy_for_v,
    ('\\partial_y u \\partial_x v', 'v'), #du_dy_dv_dx_for_v,
    ('\\partial_y u \\partial_y v', 'v'), #du_dy_dv_dy_for_v,
    ('\\partial_x v \\partial_y v', 'v'), #dv_dx_dv_dy_for_v,
]
    
#### 5 and higher order #####
@jit
def u_fifth_for_u(im, dx=1.0):
    """Compute u^5 for the 'u' component."""
    re0 = im[0] ** 5
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_fifth_for_v(im, dx=1.0):
    """Compute u^5 for the 'v' component."""
    re1 = im[0] ** 5
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_fourth_v_for_u(im, dx=1.0):
    """Compute u^4 * v for the 'u' component."""
    re0 = (im[0] ** 4) * im[1]
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_fourth_v_for_v(im, dx=1.0):
    """Compute u^4 * v for the 'v' component."""
    re1 = (im[0] ** 4) * im[1]
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_cubed_v_squared_for_u(im, dx=1.0):
    """Compute u^3 * v^2 for the 'u' component."""
    re0 = (im[0] ** 3) * (im[1] ** 2)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_cubed_v_squared_for_v(im, dx=1.0):
    """Compute u^3 * v^2 for the 'v' component."""
    re1 = (im[0] ** 3) * (im[1] ** 2)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_squared_v_cubed_for_u(im, dx=1.0):
    """Compute u^2 * v^3 for the 'u' component."""
    re0 = (im[0] ** 2) * (im[1] ** 3)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_squared_v_cubed_for_v(im, dx=1.0):
    """Compute u^2 * v^3 for the 'v' component."""
    re1 = (im[0] ** 2) * (im[1] ** 3)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def u_v_fourth_for_u(im, dx=1.0):
    """Compute u * v^4 for the 'u' component."""
    re0 = im[0] * (im[1] ** 4)
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def u_v_fourth_for_v(im, dx=1.0):
    """Compute u * v^4 for the 'v' component."""
    re1 = im[0] * (im[1] ** 4)
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

@jit
def v_fifth_for_u(im, dx=1.0):
    """Compute v^5 for the 'u' component."""
    re0 = im[1] ** 5
    re = jnp.zeros_like(im)
    re = re.at[0].set(re0)
    return re

@jit
def v_fifth_for_v(im, dx=1.0):
    """Compute v^5 for the 'v' component."""
    re1 = im[1] ** 5
    re = jnp.zeros_like(im)
    re = re.at[1].set(re1)
    return re

# # Add power 5 terms to the function list
# function_list += (
#     u_fifth_for_u,
#     u_fifth_for_v,
#     u_fourth_v_for_u,
#     u_fourth_v_for_v,
#     u_cubed_v_squared_for_u,
#     u_cubed_v_squared_for_v,
#     u_squared_v_cubed_for_u,
#     u_squared_v_cubed_for_v,
#     u_v_fourth_for_u,
#     u_v_fourth_for_v,
#     v_fifth_for_u,
#     v_fifth_for_v,
# )

# # Add power 5 terms to the base function symbols
# base_function_symbols += [
#     ('u^5', 'u'),
#     ('u^5', 'v'),
#     ('u^4 v', 'u'),
#     ('u^4 v', 'v'),
#     ('u^3 v^2', 'u'),
#     ('u^3 v^2', 'v'),
#     ('u^2 v^3', 'u'),
#     ('u^2 v^3', 'v'),
#     ('u v^4', 'u'),
#     ('u v^4', 'v'),
#     ('v^5', 'u'),
#     ('v^5', 'v'),
# ]



# @jit
# def u_sixth_for_u(im, dx=1.0):
#     """Compute u^6 for the 'u' component."""
#     re0 = im[0] ** 6
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_sixth_for_v(im, dx=1.0):
#     """Compute u^6 for the 'v' component."""
#     re1 = im[0] ** 6
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_fifth_v_for_u(im, dx=1.0):
#     """Compute u^5 * v for the 'u' component."""
#     re0 = (im[0] ** 5) * im[1]
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_fifth_v_for_v(im, dx=1.0):
#     """Compute u^5 * v for the 'v' component."""
#     re1 = (im[0] ** 5) * im[1]
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_fourth_v_squared_for_u(im, dx=1.0):
#     """Compute u^4 * v^2 for the 'u' component."""
#     re0 = (im[0] ** 4) * (im[1] ** 2)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_fourth_v_squared_for_v(im, dx=1.0):
#     """Compute u^4 * v^2 for the 'v' component."""
#     re1 = (im[0] ** 4) * (im[1] ** 2)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_cubed_v_cubed_for_u(im, dx=1.0):
#     """Compute u^3 * v^3 for the 'u' component."""
#     re0 = (im[0] ** 3) * (im[1] ** 3)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_cubed_v_cubed_for_v(im, dx=1.0):
#     """Compute u^3 * v^3 for the 'v' component."""
#     re1 = (im[0] ** 3) * (im[1] ** 3)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_squared_v_fourth_for_u(im, dx=1.0):
#     """Compute u^2 * v^4 for the 'u' component."""
#     re0 = (im[0] ** 2) * (im[1] ** 4)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_squared_v_fourth_for_v(im, dx=1.0):
#     """Compute u^2 * v^4 for the 'v' component."""
#     re1 = (im[0] ** 2) * (im[1] ** 4)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_v_fifth_for_u(im, dx=1.0):
#     """Compute u * v^5 for the 'u' component."""
#     re0 = im[0] * (im[1] ** 5)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_v_fifth_for_v(im, dx=1.0):
#     """Compute u * v^5 for the 'v' component."""
#     re1 = im[0] * (im[1] ** 5)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def v_sixth_for_u(im, dx=1.0):
#     """Compute v^6 for the 'u' component."""
#     re0 = im[1] ** 6
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def v_sixth_for_v(im, dx=1.0):
#     """Compute v^6 for the 'v' component."""
#     re1 = im[1] ** 6
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# # Add power 6 terms to the function list
# function_list += (
#     u_sixth_for_u,
#     u_sixth_for_v,
#     u_fifth_v_for_u,
#     u_fifth_v_for_v,
#     u_fourth_v_squared_for_u,
#     u_fourth_v_squared_for_v,
#     u_cubed_v_cubed_for_u,
#     u_cubed_v_cubed_for_v,
#     u_squared_v_fourth_for_u,
#     u_squared_v_fourth_for_v,
#     u_v_fifth_for_u,
#     u_v_fifth_for_v,
#     v_sixth_for_u,
#     v_sixth_for_v,
# )

# # Add power 6 terms to the base function symbols
# base_function_symbols += [
#     ('u^6', 'u'),
#     ('u^6', 'v'),
#     ('u^5 v', 'u'),
#     ('u^5 v', 'v'),
#     ('u^4 v^2', 'u'),
#     ('u^4 v^2', 'v'),
#     ('u^3 v^3', 'u'),
#     ('u^3 v^3', 'v'),
#     ('u^2 v^4', 'u'),
#     ('u^2 v^4', 'v'),
#     ('u v^5', 'u'),
#     ('u v^5', 'v'),
#     ('v^6', 'u'),
#     ('v^6', 'v'),
# ]

# @jit
# def u_seventh_for_u(im, dx=1.0):
#     """Compute u^7 for the 'u' component."""
#     re0 = im[0] ** 7
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_seventh_for_v(im, dx=1.0):
#     """Compute u^7 for the 'v' component."""
#     re1 = im[0] ** 7
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_sixth_v_for_u(im, dx=1.0):
#     """Compute u^6 * v for the 'u' component."""
#     re0 = (im[0] ** 6) * im[1]
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_sixth_v_for_v(im, dx=1.0):
#     """Compute u^6 * v for the 'v' component."""
#     re1 = (im[0] ** 6) * im[1]
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_fifth_v_squared_for_u(im, dx=1.0):
#     """Compute u^5 * v^2 for the 'u' component."""
#     re0 = (im[0] ** 5) * (im[1] ** 2)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_fifth_v_squared_for_v(im, dx=1.0):
#     """Compute u^5 * v^2 for the 'v' component."""
#     re1 = (im[0] ** 5) * (im[1] ** 2)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_fourth_v_cubed_for_u(im, dx=1.0):
#     """Compute u^4 * v^3 for the 'u' component."""
#     re0 = (im[0] ** 4) * (im[1] ** 3)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_fourth_v_cubed_for_v(im, dx=1.0):
#     """Compute u^4 * v^3 for the 'v' component."""
#     re1 = (im[0] ** 4) * (im[1] ** 3)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_cubed_v_fourth_for_u(im, dx=1.0):
#     """Compute u^3 * v^4 for the 'u' component."""
#     re0 = (im[0] ** 3) * (im[1] ** 4)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_cubed_v_fourth_for_v(im, dx=1.0):
#     """Compute u^3 * v^4 for the 'v' component."""
#     re1 = (im[0] ** 3) * (im[1] ** 4)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_squared_v_fifth_for_u(im, dx=1.0):
#     """Compute u^2 * v^5 for the 'u' component."""
#     re0 = (im[0] ** 2) * (im[1] ** 5)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_squared_v_fifth_for_v(im, dx=1.0):
#     """Compute u^2 * v^5 for the 'v' component."""
#     re1 = (im[0] ** 2) * (im[1] ** 5)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def u_v_sixth_for_u(im, dx=1.0):
#     """Compute u * v^6 for the 'u' component."""
#     re0 = im[0] * (im[1] ** 6)
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def u_v_sixth_for_v(im, dx=1.0):
#     """Compute u * v^6 for the 'v' component."""
#     re1 = im[0] * (im[1] ** 6)
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# @jit
# def v_seventh_for_u(im, dx=1.0):
#     """Compute v^7 for the 'u' component."""
#     re0 = im[1] ** 7
#     re = jnp.zeros_like(im)
#     re = re.at[0].set(re0)
#     return re

# @jit
# def v_seventh_for_v(im, dx=1.0):
#     """Compute v^7 for the 'v' component."""
#     re1 = im[1] ** 7
#     re = jnp.zeros_like(im)
#     re = re.at[1].set(re1)
#     return re

# # Add power 7 terms to the function list
# function_list += (
#     u_seventh_for_u,
#     u_seventh_for_v,
#     u_sixth_v_for_u,
#     u_sixth_v_for_v,
#     u_fifth_v_squared_for_u,
#     u_fifth_v_squared_for_v,
#     u_fourth_v_cubed_for_u,
#     u_fourth_v_cubed_for_v,
#     u_cubed_v_fourth_for_u,
#     u_cubed_v_fourth_for_v,
#     u_squared_v_fifth_for_u,
#     u_squared_v_fifth_for_v,
#     u_v_sixth_for_u,
#     u_v_sixth_for_v,
#     v_seventh_for_u,
#     v_seventh_for_v,
# )

# # Add power 7 terms to the base function symbols
# base_function_symbols += [
#     ('u^7', 'u'),
#     ('u^7', 'v'),
#     ('u^6 v', 'u'),
#     ('u^6 v', 'v'),
#     ('u^5 v^2', 'u'),
#     ('u^5 v^2', 'v'),
#     ('u^4 v^3', 'u'),
#     ('u^4 v^3', 'v'),
#     ('u^3 v^4', 'u'),
#     ('u^3 v^4', 'v'),
#     ('u^2 v^5', 'u'),
#     ('u^2 v^5', 'v'),
#     ('u v^6', 'u'),
#     ('u v^6', 'v'),
#     ('v^7', 'u'),
#     ('v^7', 'v'),
# ]