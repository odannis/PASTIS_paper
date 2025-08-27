from ._common_class import Model
import numpy as np
import SFFI.simulation as simu
from jax import jit
from SFFI.sffi import PolynomeInfo
import pysindy as ps

class TwoStates(Model):
    def __init__(self, a=1.0, **kwargs):
        self.init_params = locals().copy()
        print(self.init_params)
        del self.init_params['self']  # Remove 'self' from the dictionary
        del self.init_params['__class__']
        super().__init__(**kwargs)
        self.a = a
        dim = 1
        self.shape_image = [dim]
        self.base = [self.drift_term]
        self.coefficient = [1]
        self.force_numba = self.drift_term()
        self.n_differential_equation = dim
        self.feature_library = ps.PolynomialLibrary(degree=5, include_bias=True, include_interaction=True)
        self.index_real_base_sindy =           [1, 3]
        self.index_real_base_sindy_weak_form = [1, 3] ## Work with last version sindy
        self.total_base = PolynomeInfo(dim=dim, order=5)
        self.index_real_base = [1, 3]
        self.first_image = np.array([.7])
        
    def drift_term(self):
        """
        Potential V(x) = (x-a)(x+a)x^2 y^2
        """
        a = self.a
        # @njit(inline='always')
        # def F_t(x_array):
        #     x = x_array[0]
        #     y = x_array[1]
        #     return -1*np.array([(4*x**3 - 2*x*a**2)*y**2, 2*y*(x-a)*(x+a)*x**2])
        # return F_t
        @jit
        def F_t(x):
            return -(4*x**3 - 2*x*a**2)
        return F_t
    
    def sqrt_diffusion(self):
        return 1