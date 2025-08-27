import numpy as np
from jax import jit
from jax import numpy as jnp
import jax 
from functools import partial

def polynomial_basis(dim, order) -> list[np.ndarray]:
    # A simple polynomial basis, X -> X_mu X_nu ... up to polynomial
    # degree 'order'.
    
    # We first generate the coefficients, ie the indices mu,nu.. of
    # the polynomials, in a non-redundant way. We start with the
    # constant polynomial (empty list of coefficients) and iteratively
    # add new indices.
    coeffs = [ np.array([[]], dtype=int) ]
    for n in range(order):
            # Generate the next coefficients:
            new_coeffs = []
            for c in coeffs[-1]: 
                # We generate loosely ordered lists of coefficients
                # (c1 >= c2 >= c3 ...)  (avoids redundancies):
                for i in range( (c[-1]+1) if c.shape[0]>0 else dim):
                    new_coeffs.append(list(c)+[i])
            coeffs.append(np.array(new_coeffs, dtype=int)) 
    # Group all coefficients together
    coeffs = [ c for degree in coeffs for c in degree ]
    return coeffs


class PolynomeInfo:
    def __init__(self, dim : int = 3, order : int = 2, simple_base : bool = False) -> None:
        """
        dim : dimension of the space
        order : order of the polynomials
        simple_base : if True, the basis is on the i dimension -> X_i X_mu X_nu ... up to order 'order'-1 (no constant term)
        """
        self.dim = dim
        self.simple_base = simple_base
        self.order = order
        self.base_function = self.polynomial_basis_function()
        self.non_zero_dim = []
        if self.simple_base:
            self.coeffs = polynomial_basis(self.dim, self.order -1)
        else:
            self.coeffs = polynomial_basis(self.dim, self.order)
        self.n_base = len(self.coeffs)*self.dim
        
    def __len__(self):
        return self.n_base

    # def get_basis_values(self, X : np.ndarray):
    #     """
    #     Evaluate the polynomial basis with coefficients of order defined in the class at the
    #     points 'X'.
    #     """
    #     coeffs = self.coeffs
    #     X = np.asarray(X)
    #     # Evaluate the polynomial basis with coefficients 'coeffs' at the
    #     # points 'X'.
    #     # We first generate the powers of X (X_mu X_nu ...):
    #     Xpowers = np.empty(((len(coeffs))*self.dim, *X.shape))
    #     l_name = []
    #     non_zero_dim = []
    #     name = np.array(["X_%s"%(i) for i in range(0, self.dim)])
    #     for i in range(len(coeffs)):
    #         out = jnp.prod(X[:,coeffs[i]], axis=1)
    #         for j in range(self.dim):
    #             if self.simple_base:
    #                 out = out.copy()
    #             Xpowers[i*self.dim + j, :, j] = out
    #             coeff_out = coeffs[i].copy()
    #             if self.simple_base:
    #                 Xpowers[i*self.dim + j, :, j] *=  (X[:, j])
    #                 coeff_out = np.append(coeffs[i], j)
    #             if len(coeff_out) == 0:
    #                 l_name.append("On %s : 1"%(name[j]))
    #             else:
    #                 l_name.append("On %s : %s"%(name[j], name[coeff_out]))
    #             non_zero_dim.append(j)
    #     if len(self.non_zero_dim) == 0:
    #         self.non_zero_dim = non_zero_dim
    #     return Xpowers, l_name
    
    def get_basis_values(self, X : np.ndarray):
        out = evaluate_all_base_jax(self.base_function, X)
        l_name = self.get_basis_name()
        return out, l_name
    
    def get_basis_name(self):
        """
        Evaluate the polynomial basis with coefficients of order defined in the class at the
        points 'X'.
        """
        coeffs = self.coeffs
        l_name = []
        non_zero_dim = []
        name = np.array(["X_%s"%(i) for i in range(0, self.dim)])
        for i in range(len(coeffs)):
            for j in range(self.dim):
                coeff_out = coeffs[i].copy()
                if self.simple_base:
                    coeff_out = np.append(coeffs[i], j)
                if len(coeff_out) == 0:
                    l_name.append("On %s : 1"%(name[j]))
                else:
                    l_name.append("On %s : %s"%(name[j], name[coeff_out]))
                non_zero_dim.append(j)
        if len(self.non_zero_dim) == 0:
            self.non_zero_dim = non_zero_dim
        return l_name
    
    def polynomial_basis_derivate(self, X : np.ndarray):
        """
        Evaluate the polynomial basis with coefficients of order defined in the class at the
        points 'X'.
        """
        coeffs = self.coeffs

        Xpowers = np.zeros(((len(coeffs))*self.dim, *X.shape, X.shape[-1]))
        for i in range(len(coeffs)):
            coeff = coeffs[i]
            if not self.simple_base:
                for j_deriv in range(self.dim):
                    power_mult, coeff_deriv = self.derivate_coeff(coeff, j_deriv)
                    if power_mult == 0:
                        out = np.zeros(X.shape[0])
                    else:
                        out = power_mult * np.prod(X[:,coeff_deriv], axis=1)
                    for j in range(self.dim):
                        Xpowers[i*self.dim + j, :, j_deriv, j] = out.copy()
            else:
                for j_deriv in range(self.dim):
                    for j in range(self.dim):
                        coeff_use = np.append(coeff, j)
                        power_mult, coeff_deriv = self.derivate_coeff(coeff_use, j)
                        if power_mult == 0:
                            out = out = np.zeros(X.shape[0])
                        else:
                            out = power_mult * np.prod(X[:,coeff_deriv], axis=1)
                        Xpowers[i*self.dim + j, :, j_deriv, j] = out.copy()
        return Xpowers
    
    def derivate_coeff(self, coeff : np.ndarray, dim):
        if len(coeff) == 0:
            return 0, []
        else:
            power_dim = np.sum(coeff == dim)
            if power_dim == 0:
                return 0, []
            else:
                new_coeff = coeff.copy()
                new_coeff = list(new_coeff)
                new_coeff.remove(dim)
                return power_dim, new_coeff
    
    def polynomial_basis_values_flat(self, X : np.ndarray):
        """
        Evaluate the polynomial basis with coefficients of order defined in the class at the
        points 'X'.
        """

        coeffs = polynomial_basis(self.dim, self.order)
        # Evaluate the polynomial basis with coefficients 'coeffs' at the
        # points 'X'.
        # We first generate the powers of X (X_mu X_nu ...):
        Xpowers = np.zeros((len(coeffs), X.shape[0]))
        l_name = []
        name = np.array(["X_%s"%(i) for i in range(0, self.dim)])
        for i in range(len(coeffs)):
            out = np.prod(X[:,coeffs[i]], axis=1)
            Xpowers[i, :] = out.copy()
            coeff_out = coeffs[i].copy()
            if len(coeff_out) == 0:
                l_name.append(" 1")
            else:
                l_name.append("%s"%(name[coeff_out]))
        return Xpowers, l_name
    
    def polynomial_basis_function(self):
        """ Return lambda functions for the polynomial basis.

        Returns:
            List[function]: List of basis functions
        """
        if self.simple_base:
            coeffs = polynomial_basis(self.dim, self.order -1)
        else:
            coeffs = polynomial_basis(self.dim, self.order)
            
        l_func = []
        for i in range(len(coeffs)):
            for j in range(self.dim):
                coeff_to_use = np.array(coeffs[i]).copy()
                if self.simple_base:
                    coeff_to_use = np.append(coeff_to_use, j)
                func = def_func_poly(j, coeff_to_use)
                func(np.array([0.0]*self.dim)) #precompile MANDATORY of function overwrite by new function
                func(np.array([0.1]*self.dim)) #precompile MANDATORY of function overwrite by new function
                l_func.append(func)
        return tuple(l_func)

def def_func_poly(dim, coeff_to_use):
    @jit
    def func(X):
        Xpowers = jnp.zeros_like(X)
        Xpowers = Xpowers.at[dim].set(jnp.prod(X[coeff_to_use]))
        return Xpowers
    return func
    
@partial(jax.jit, static_argnums=(0,))
def evaluate_all_base_jax(l_base, phi_chunk):
    results = []
    for func in l_base:
        func_vmap = jax.vmap(func, in_axes=(0))
        r_func = func_vmap(phi_chunk)
        results.append(r_func)
    r = jnp.stack(results, axis=0)
    return r
    
if __name__=='__main__':
    # import unittest
    # print("Test PolynomeInfo")
    # class TestPrime(unittest.TestCase):
    #     x = np.random.rand(100, 3)
    #     for simple_base in [True, False]:
    #         polynome_info = PolynomeInfo(dim=3, order=2, simple_base=simple_base)
    #         polynome_info.base_function = polynome_info.polynomial_basis_function()
            
    #         def test(self):
    #             base_evaluated, _ = self.polynome_info.get_basis_values(self.x)
    #             for index in range(self.x.shape[0]):
    #                 for i in range(len(base_evaluated)):
    #                     self.assertTrue(np.allclose(base_evaluated[i, index], self.polynome_info.base_function[i](self.x[index]))) #type: ignore
                
    # unittest.main()
    polynome_info = PolynomeInfo(dim=3, order=2, simple_base=False)
    x = np.random.rand(10_000_000, 3)
    polynome_info.get_basis_values(x)