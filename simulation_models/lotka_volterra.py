from math import sqrt
from ._common_class import Model
import numpy as np
import SFFI.simulation as simu
from SFFI.sffi import PolynomeInfo
from SFFI.simulation import simulate
import pysindy as ps
from jax import numpy as jnp
from jax import jit

class LotkaVolterra(Model):
    def __init__(self, A = np.array([[ 1.,  1., -1.],[-1.,  1.,  1.],[ 1., -1.,  1.]]), r = np.array([1, 1, 1]), simple_base=True,
                 diffusion_constant=False, r_in_prefactor=True, **kwargs):
        self.diffusion_constant = diffusion_constant
        self.init_params = locals().copy()
        A = np.array(A)
        r = np.array(r)
        self.init_params["A"] = A.tolist()
        self.init_params["r"] = r.tolist()
        del self.init_params['self']  # Remove 'self' from the dictionary
        del self.init_params['__class__']
        super().__init__(**kwargs)
        assert A.shape[0] == r.shape[0]
        self.r_in_prefactor = r_in_prefactor
        self.A = np.array(A, dtype=np.float32)
        self.r = np.array(r, dtype=np.float32)
        self.first_image = np.linalg.pinv(A)@np.ones(A.shape[0]) 
        self.shape_image = [A.shape[0]]
        self.drift_term = self.drift_term_()
        self.sqrt_diffusion = self.sqrt_diffusion_()
        self.base = [self.drift_term] 
        self.coefficient = [1]
        self.force_numba = self.drift_term
        self.n_differential_equation = A.shape[0]
        self.total_base = PolynomeInfo(dim = A.shape[0], order = 2, simple_base=simple_base)
        self.index_real_base = self.find_index_real_base(total_base=self.total_base)
        self.feature_library = ps.PolynomialLibrary(degree=2, include_bias=True, include_interaction=True)
        self.constraint_rhs, self.constraint_lhs = self.constraints()
        self.index_real_base_sindy = self.find_index_real_model_sindy()
        self.index_real_base_sindy_weak_form = self.find_index_real_model_sindy(weak_form=True)
        self.simple_base = simple_base

    def drift_term_(self):
        A = self.A
        r = self.r
        if self.r_in_prefactor:
            @jit
            def F_t(x):
                return (r*x)*(1 - A@x)
            return F_t
        else:
            @jit
            def F_t(x):
                return x*(r - A@x)
            return F_t
        
    def sqrt_diffusion_(self):
        if self.diffusion_constant:
            return 1
        r = self.r    
        # @jit
        # def sqrt_D(x):
        #     out = jnp.zeros((x.shape[0], x.shape[0]))
        #     for i in range(x.shape[0]):
        #         out[i, i] = x[i]
        #     return out
        def sqrt_D(x):
            return jnp.diag(x)
        return sqrt_D
    
    def find_index_real_base(self, total_base : PolynomeInfo):
        para = self.get_parameter_simulation()
        para["n"] = 100
        phi = simulate(**para)[0]
        _, l_name_base = total_base.get_basis_values(phi)
        l_name_base = np.array(l_name_base)
        index_real_base = []

        for i in range(self.r.shape[0]):
            if self.r[i] != 0:
                func_name = "On X_%s : ['X_%s']" % (i, i)
                index_real_base.append(list(l_name_base).index(func_name))
                
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if self.A[i, j] != 0:
                    try:
                        func_name = "On X_%s : ['X_%s' 'X_%s']" % (i, max(i, j), min(j,i))
                        index_real_base.append(list(l_name_base).index(func_name))
                    except:
                        func_name = "On X_%s : ['X_%s' 'X_%s']" % (i, min(i, j), max(j,i))
                        index_real_base.append(list(l_name_base).index(func_name))
        index_real_base = sorted(index_real_base)
        return index_real_base
    
    def find_index_real_model_sindy(self, weak_form = False):
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
            model.fit(phi, t=dt)

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
        for i in range(self.r.shape[0]):
            if self.r[i] != 0:
                func_name = "On x%s : x%s" % (i, i)
                index_real_base.append(list(l_name_base).index(func_name))
                
        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]):
                if self.A[i, j] != 0:
                    try:
                        if i == j:
                            func_name = "On x%s : x%s^2" % (i, i)
                        else:
                            func_name = "On x%s : x%s x%s" % (i, max(i, j), min(j,i))
                        index_real_base.append(list(l_name_base).index(func_name))
                    except:
                        func_name = "On x%s : x%s x%s" % (i, min(i, j), max(j,i))
                        index_real_base.append(list(l_name_base).index(func_name))
        
        index_real_base = sorted(index_real_base)
        return index_real_base
    
    def constraints(self):
        para = self.get_parameter_simulation()
        para["n"] = 100
        x_simu, _ = simulate(**para)
        library = self.feature_library
        
        library.fit([ps.AxesArray(x_simu, {"ax_sample": 0, "ax_coord": 1})])
        n_features = library.n_output_features_
        #print(f"Features ({n_features}):", library.get_feature_names())

        # Set constraints
        total_contraints = 0
        for dim in range(self.A.shape[0]):
            dimension = "x" + str(dim)
            for i,f in enumerate(library.get_feature_names()):
                if dimension not in f:
                    if len(f) > 2:
                        total_contraints += 1
        n_targets = x_simu.shape[1]
        constraint_rhs = np.array([0]*total_contraints)
        constraint_lhs = np.zeros((constraint_rhs.shape[0], n_targets * n_features))
        c = 0
        for dim in range(self.A.shape[0]):
            dimension = "x" + str(dim)
            for i,f in enumerate(library.get_feature_names()):
                if dimension not in f:
                    if len(f) > 2:
                        constraint_lhs[c, i + dim*n_features] = 1
                        c += 1
        return constraint_rhs, constraint_lhs