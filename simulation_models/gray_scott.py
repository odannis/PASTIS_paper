from ._common_class import Model
import numpy as np
import pysindy as ps
from .gray_scott_utils import base_function_gray_scott as bf
from .gray_scott_utils import dumb_base as db
from jax import numpy as jnp
from jax import jit

def gs_initialiser(shape, size=10):
        a = np.ones(shape)
        b = np.zeros(shape)
        centre = int(shape[0] / 2)
        # a[centre-size:centre+size,centre-size:centre+size] = 0.5
        # b[centre-size:centre+size,centre-size:centre+size] = 0.25
        a += np.random.normal(scale=0.1, size=shape)
        b += np.random.normal(scale=0.1, size=shape)
        y, x = np.ogrid[:shape[0], :shape[1]]
        mask = (x - centre)**2 + (y - centre)**2 <= size**2
        a[mask] = 0.5
        b[mask] = 0.25
        return np.array([a,b])
    
class GrayScott(Model):
    def __init__(self, Da=0.2097, Db=0.105, f=0.029, k=0.057, shape_image=(2,100, 100), **kwargs):
        self.init_params = locals().copy()
        del self.init_params['self']  # Remove 'self' from the dictionary
        del self.init_params['__class__'] 
        super().__init__(**kwargs)
        self.shape_image = shape_image
        self.first_image = gs_initialiser(self.shape_image[1:])
        self.base = [bf.lap_u, bf.u_v2_for_u, bf.cste_for_u, bf.u_for_u,
                        bf.lap_v, bf.u_v2_for_v, bf.v_for_v]
        self.coefficients = np.array([Da, -1, f, -f, Db, 1, -f - k])
        @jit
        def drift(x):
            x_out = jnp.zeros_like(x)
            coeffs = jnp.array(self.coefficients)
            for coef, func in zip(coeffs, self.base):
                x_out += coef * func(x)
            return x_out
        self.drift_term = drift
        self.sqrt_diffusion = 1
        self.total_base = db.function_list
        self.index_real_base = [0, 1, 4, 6, 7, 20, 21]
        # if total_base is not None:
        #     self.total_base = total_base
        self.feature_library = ps.PolynomialLibrary(degree=2, include_bias=True, include_interaction=True)
        self.index_real_base_sindy =           [1, 3, 12, 16, 21, 23, 25]
        self.index_real_base_sindy_weak_form = [2, 4, 14, 18, 24, 26, 28] ## Work with last version sindy

    def __str__(self):
        return f"GrayScott(Da={self.init_params['Da']}, Db={self.init_params['Db']}, f={self.init_params['f']}, k={self.init_params['k']})"

# import SFFI.simulation as simu
# parameter_simulation = {
#      "shape_output" : (700, 2, 100, 100), "dt" : 1, "diffusion" : 1e-4, "oversampling" : 1, "dx" : 1,
#      "thermalization_steps" : 100,
#     }

# Da, Db, f, k = [0.16, 0.08, 0.05, 0.065]
# Da, Db, f, k = [0.2097, 0.105, 0.03, 0.062] # https://github.com/pmneila/jsexp/blob/48ea6d6dbea46f38080701f01a98341c15fee946/grayscott/index.html#L29
# Da, Db, f, k = [0.2097, 0.105, 0.029, 0.057] # https://github.com/pmneila/jsexp/blob/48ea6d6dbea46f38080701f01a98341c15fee946/grayscott/index.html#L29

# Da *= parameter_simulation["dx"]**2
# Db *= parameter_simulation["dx"]**2

# base = (bf.lap_u, bf.u_v2_for_u, bf.cste_for_u, bf.u_for_u,
#         bf.lap_v, bf.u_v2_for_v, bf.v_for_v)

# base_function_symbols_U = [' \\nabla^2 u', ' u v^2', ' ', ' u']
# base_function_symbols_V = [' \\nabla^2 v', ' u v^2', ' v']

# coefficient_simu = np.array([Da, -1, f, -f,
#                Db, 1, -f - k])
               
    
# first_image = gs_initialiser(parameter_simulation["shape_output"][2:])
# parameter_simulation["first_image"] = first_image
# phi, delta_t = simu.simulate(base, coefficient_simu, **parameter_simulation)
# dx = parameter_simulation["dx"]