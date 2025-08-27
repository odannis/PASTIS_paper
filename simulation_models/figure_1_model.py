from ._common_class import Model
import numpy as np
import SFFI.simulation as simu
from jax import jit
from SFFI.sffi import PolynomeInfo
import pysindy as ps


class Figure1Model(Model):
    def __init__(self, **kwargs):
        self.init_params = locals().copy()
        del self.init_params['self']  # Remove 'self' from the dictionary
        del self.init_params['__class__']
        super().__init__(**kwargs)
        self.first_image = np.array([0.0])
        self.shape_image = [1]
        self.n_differential_equation = 1
        
        degree_poly = 20
        self.feature_library = ps.PolynomialLibrary(degree=degree_poly, include_bias=True, include_interaction=True)
        self.index_real_base_sindy =           [0]
        self.index_real_base_sindy_weak_form = [0] ## Work with last version sindy
        
        self.total_base = PolynomeInfo(dim=self.n_differential_equation, order=degree_poly)
        self.index_real_base = [0]
        self.drift_term = self.drift_term_()
        self.sqrt_diffusion = self.sqrt_diffusion_()
        self.coefficient = [1]
        self.base = [self.drift_term]
        self.force_numba = self.drift_term
        
    def drift_term_(self):
        @jit
        def F_t(x):
            return -x/(1-x**2)**2
        return F_t
     
    def sqrt_diffusion_(self):
        return 1
