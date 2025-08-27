from ._common_class import Model
import numpy as np
import SFFI.simulation as simu
from jax import jit
from jax import numpy as jnp
from SFFI.sffi import PolynomeInfo
import pysindy as ps

class Lorenz(Model):
    def __init__(self, r : float = 28, b : float = 7/3, s : float = 10, diffusion_constant : bool = True,
                 total_base = None,
                 **kwargs):
        self.init_params = locals().copy()
        del self.init_params['self']  # Remove 'self' from the dictionary
        del self.init_params['__class__']
        del self.init_params['total_base']  
        super().__init__(**kwargs)
        self.diffusion_constant = diffusion_constant
        self.r = r
        self.b = b
        self.s = s
        self.first_image = np.array([np.sqrt(b*(r-1))*1.1, (r-1)*1.1, (np.sqrt(b*(r-1)))*1.1], dtype=np.float32)
        self.shape_image = [3]
        self.n_differential_equation = 3
        self.feature_library = ps.PolynomialLibrary(degree=2, include_bias=True, include_interaction=True)
        self.index_real_base_sindy =           [1, 3, 12, 16, 21, 23, 25]
        self.index_real_base_sindy_weak_form = [2, 4, 14, 18, 24, 26, 28] ## Work with last version sindy
        self.total_base = PolynomeInfo(dim=3, order=2)
        self.index_real_base = [3, 5, 7, 9, 11, 17, 22]
        if total_base is not None:
            self.total_base = total_base
            self.index_real_base = [0]
                
        self.drift_term = self.drift_term_()
        self.sqrt_diffusion = self.sqrt_diffusion_()
        self.coefficient = [1]
        self.base = [self.drift_term]
        self.force_numba = self.drift_term
        
    def drift_term_(self):
        s = jnp.asarray(self.s, dtype=jnp.float64)
        r = jnp.asarray(self.r, dtype=jnp.float64)
        b = jnp.asarray(self.b, dtype=jnp.float64)

        @jit
        def F_t(x):
            x = jnp.asarray(x, dtype=jnp.float64)
            return jnp.array([
                s * (x[2] - x[0]),
                x[0] * x[2] - b * x[1],
                r * x[0] - x[2] - x[1] * x[0]
            ], dtype=jnp.float64)

        return F_t
    
    def sqrt_diffusion_(self):
        if self.diffusion_constant:
            return 1
        @jit
        def sqrt_D(x):
            out = jnp.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                out[i, i] = jnp.sqrt(jnp.abs(x[i]))
            return out
        return sqrt_D
