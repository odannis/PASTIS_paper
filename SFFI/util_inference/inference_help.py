import numpy as np
from jax import jit
from jax import numpy as jnp
import numpy.typing as npt
from typing import TYPE_CHECKING
from .polynome import PolynomeInfo
from .fourier import Fourier
import jax
from functools import partial
if TYPE_CHECKING:
    from ..sffi import SFFI
jax.config.update("jax_enable_x64", True)
#jax.config.update("jax_default_matmul_precision", "F64_F64_F64")
    
def evaluate_all_base(l_base, phi, use_jax: bool) -> tuple[npt.NDArray, npt.NDArray]:
    if type(l_base) is PolynomeInfo or type(l_base) is Fourier:
        r, l_name = l_base.get_basis_values(phi)
    else:
        l_name = [str(func) for func in l_base]
        r = np.empty((len(l_base), *phi.shape))
        if use_jax:
            r = evaluate_all_base_jax(tuple(l_base), phi)
        else:
            for i, func in enumerate(l_base):
                evaluate_base_python(func, phi, r[i])
    return r, np.array(l_name)

def evaluate_deriv_all_base_new(l_base, phi: npt.NDArray, eps : float = 0.00001, order_deriv : int = 1) -> npt.NDArray:
    if type(l_base) is PolynomeInfo:
        if l_base.base_function is None:
            l_base.base_function = l_base.polynomial_basis_function()
        l_base = l_base.base_function
    r = []
    for i, func in enumerate(l_base):
        if len(phi.shape) == 1:
            r.append(evaluate_base_deriv(func, phi, eps, order_deriv))
        else:
            r.append(evaluate_base_deriv_multidimension(func, phi, eps, order_deriv))
    return np.asarray(r)

def evaluate_deriv_all_base(l_base, phi, eps : float = 0.0000001, order_deriv : int = 1, use_jax = True) -> npt.NDArray:
    ###
    # return [matrix]_{k,t,i,j} = db_{k}_{j}/dx_{i}(x_t)
    if type(l_base) is PolynomeInfo:
        return l_base.polynomial_basis_derivate(phi)
    else:
        if order_deriv == 1:
            base_not_pertub, _ = evaluate_all_base(l_base, phi, use_jax=use_jax)
            base_not_pertub /= eps
            if len(phi.shape) == 1:
                r = np.zeros(base_not_pertub.shape)
                phi_pertub = phi.copy()
                phi_pertub += eps
                base_pertub, _ = evaluate_all_base(l_base, phi_pertub, use_jax=use_jax)
                base_pertub /= eps
                np.subtract(base_pertub, base_not_pertub, out=r)
            else:
                r = np.zeros((*base_not_pertub.shape, phi.shape[-1]))
                for dim in range(phi.shape[-1]):
                    phi_pertub = phi.copy()
                    phi_pertub[:, dim] += eps
                    base_pertub, _ = evaluate_all_base(l_base, phi_pertub, use_jax=use_jax)
                    base_pertub /= eps
                    np.subtract(base_pertub, base_not_pertub, out=r[:, :, dim, :])
        else:
            raise NotImplementedError
        return np.array(r)

@jit
def evaluate_base_deriv(func, phi: npt.NDArray, eps : float, order_deriv : int):
    base_m = np.empty_like(phi)
    if order_deriv == 1:
        for t in range(phi.shape[0]):
            base_m[t] = (func(phi[t] + eps) - func(phi[t])) / eps
    elif order_deriv == 2:
        for t in range(phi.shape[0]):
            base_m[t] = (func(phi[t] + eps) - 2*func(phi[t]) + func(phi[t] - eps)) / eps**2
    elif order_deriv == 3:
        for t in range(phi.shape[0]):
            base_m[t] = (func(phi[t] + 2*eps) - 3*func(phi[t] + eps) + 3*func(phi[t]) - func(phi[t] - eps)) / eps**3
    return base_m
         
@jit
def evaluate_base_deriv_multidimension(func, phi: npt.NDArray, eps : float, order_deriv : int):
    base_m = np.zeros((*phi.shape, phi.shape[-1]))
    if order_deriv == 1:
        for dim in range(base_m.shape[-1]):
            phi_pertub = phi.copy()
            phi_pertub[:, dim] += eps
            for t in range(base_m.shape[0]):
                base_m[t, dim] = (func(phi_pertub[t]) - func(phi[t])) / eps # matrix]_{i,j} = db_{1,j}/dx_{i}
    else:
        raise NotImplementedError
    return base_m

@partial(jax.jit, static_argnums=(0,))
def evaluate_all_base_jax(l_base, phi_chunk):
    results = []
    for func in l_base:
        func_vmap = jax.vmap(func, in_axes=(0))
        r_func = func_vmap(phi_chunk)
        results.append(r_func)
    r = jnp.stack(results, axis=0)
    return r

def evaluate_base_python(func, phi: npt.NDArray, base_m: npt.NDArray):
    for t in range(phi.shape[0]):
        base_m[t] = func(phi[t])

@jit
def project_delta_phi(delta_phi, base_m, A_normalisation : npt.NDArray | None, delta_t : float):
    base_m_flat = base_m.reshape(base_m.shape[0], base_m.shape[1], -1)
    delta_phi_flat = delta_phi.reshape(delta_phi.shape[0], -1)
    delta_phi_flat = normalise_data(delta_phi_flat, A_normalisation)
    t_max = min(delta_phi_flat.shape[0], base_m_flat.shape[1])
    if t_max != delta_phi_flat.shape[0] or t_max != base_m_flat.shape[1]:
        delta_phi_flat = delta_phi_flat[:t_max,:]
        base_m_flat =  base_m_flat[:,:t_max]
    F_projected = jnp.einsum('ifk,fk->i', base_m_flat, delta_phi_flat)*delta_t
    return F_projected

@jit
def compute_projection_matrix(base_m, real_base, A_normalisation : npt.NDArray | None, delta_t : float):
    num_funcs = base_m.shape[0]
    A_flat = real_base.reshape(num_funcs, real_base.shape[1], -1)
    B_flat = base_m.reshape(num_funcs, base_m.shape[1], -1)
    A_flat = normalise_data(A_flat, A_normalisation)
    m = jnp.einsum('ifk,jfk->ij', B_flat, A_flat)* delta_t
    return m

@jit
def normalise_data(m, A_normalisation):
    if A_normalisation is None:
        return m
    if len(A_normalisation.shape) == 0:
        m *= A_normalisation
    elif len(A_normalisation.shape) == 2:
        m = jnp.einsum('jk, ...k->...j', A_normalisation, m)
    else:
        raise Exception("Ill defined A_normalisation %s" % str(len(A_normalisation.shape)))
    return m

def invert_matrix(B):
    try:
        B_inv = np.linalg.pinv(B)
    except Exception as e:
        print(str(e))
        print("B is not p-invertible ! choose a better base ! --> Try regular inv")
        try:
            B_inv = np.linalg.inv(B)
        except np.linalg.LinAlgError:
            print("B is not invertible ! choose a better base ! --> Choose identity matrix")
            B_inv = np.eye(B.shape[0])
    return B_inv