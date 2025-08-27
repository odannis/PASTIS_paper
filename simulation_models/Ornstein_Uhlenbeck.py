from ._common_class import Model
import numpy as np
from SFFI.sffi import PolynomeInfo
import pysindy as ps
from SFFI.simulation import simulate
from jax import jit
from jax import numpy as jnp

class OrnsteinUhlenbeck(Model):
    def __init__(self, omega=np.array([[1.   , 0.   , 0.   , 1, 0.   ],
                                        [0.   , 1.   , 0.   , 0, 0.   ],
                                        [0   , 0.   , 1.   , 0.   , 1   ],
                                        [0, -1, 0.   , 1.   , 0.   ],
                                        [0.   , 1   , 0.   , 0.   , 1.   ]]),
                diffusion_constant = True,
                use_repulsive_gaussian = False,
                degree_polynome : int = 1,
                diffusion_strength = 100.0,
                dt=0.01,
                n=10**6,
                **kwargs):
        
        self.init_params = locals().copy()
        if type(omega) != list:
            self.init_params["omega"] = omega.tolist()
        del self.init_params['self']  # Remove 'self' from the dictionary
        del self.init_params['__class__']
        super().__init__(diffusion_strength=diffusion_strength, dt=dt, n=n, **kwargs)
        self.diffusion_constant = diffusion_constant
        self.use_repulsive_gaussian = use_repulsive_gaussian
        self.omega = np.array(omega)
        dim = self.omega.shape[0]
        self.degree_polynome = degree_polynome
        self.shape_image = [dim]
        self.drift_term = self.drift_term_() 
        self.base = [self.drift_term]
        self.sqrt_diffusion = self.sqrt_diffusion_()
        self.coefficient = [1]
        self.force_numba = self.drift_term
        self.n_differential_equation = dim
        self.feature_library = ps.PolynomialLibrary(degree=degree_polynome, include_bias=True, include_interaction=True)
        self.total_base = PolynomeInfo(dim=dim, order=degree_polynome)
        self.first_image = np.zeros(dim)
        self.index_real_base = self.find_index_real_model()
        l_real_func = []
        for i,j in zip(range(dim), range(dim)):
            if omega[i][j] != 0:
                l_real_func.append("On x%s : x%s"%(i, j))
        self.index_real_base_sindy = self.find_index_real_model_sindy(l_real_func)
        self.index_real_base_sindy_weak_form = self.find_index_real_model_sindy(l_real_func, weak_form=True)
        
    def drift_term_(self):
        omega = self.omega
        if not self.use_repulsive_gaussian:
            @jit
            def F_t(x):
                return -omega@x
        else:
            @jit
            def F_t(x):
                return -omega@x + 10 * x * jnp.exp(-jnp.sum((x)**2))
        return F_t
    
    def sqrt_diffusion_(self):
        if self.diffusion_constant:
            return 1  
        @jit
        def sqrt_D(x):
            out = np.zeros((x.shape[0], x.shape[0]))
            for i in range(x.shape[0]):
                out[i, i] = np.abs(x[i])
            return out
        return sqrt_D
    
    def find_index_real_model_sindy(self, l_real_func : list, weak_form = False):
        para = self.get_parameter_simulation()
        para["n"] = 100
        phi, dt = simulate(**para)
        feature_names = ['x', 'y', 'z']
        optimizer = ps.STLSQ(threshold=0)
        if not weak_form:
            model = ps.SINDy(optimizer=optimizer, feature_names=feature_names)
            model.fit(phi, t=1)
        else:
            t_train =  np.array(range(phi.shape[0])) * dt
            #library_functions = [lambda x: x, lambda x, y: x * y, lambda x: x ** 2]
            #library_function_names = [lambda x: x, lambda x, y: x + " "+ y, lambda x: x +" " +  x]
            ode_lib = ps.WeakPDELibrary(
                #library_functions=library_functions,
                #function_names=library_function_names,
                function_library=self.feature_library,
                spatiotemporal_grid=t_train,
                include_bias=True,
                is_uniform=True,
            )
            model = ps.SINDy(feature_names=feature_names, optimizer=optimizer, feature_library=ode_lib)
            model.fit(phi, t=float(dt))

        name_ = np.array(model.feature_library.get_feature_names())

        l_name_base = []
        for i in range(model.coefficients().shape[0]):
            l_dim = []
            x_dim = "On x%s :" % (i)
            for str_x in name_:
                l_dim.append("%s %s"%(x_dim, str_x))
            l_name_base.append(l_dim)
        l_name_base = np.array(l_name_base).flatten().tolist()

        index_real_base = []
        for func_name in l_real_func:
            index_real_base.append(l_name_base.index(func_name))
        index_real_base = sorted(index_real_base)
        return index_real_base
    
    def find_index_real_model(self):
        para = self.get_parameter_simulation()
        para["n"] = 100
        phi, dt = simulate(**para)
        if type(self.total_base) == PolynomeInfo:
            _, l_name_base = self.total_base.get_basis_values(phi)
            l_name_base = np.array(l_name_base)
            index_real_base = []
            for i in range(self.shape_image[0]):
                for j in range(self.shape_image[0]):
                    if self.omega[i, j] != 0:
                        func_name = "On X_%s : ['X_%s']" % (i, j)
                        index_real_base.append(list(l_name_base).index(func_name))
            return sorted(index_real_base)
        else:
            raise NotImplementedError
        

        
