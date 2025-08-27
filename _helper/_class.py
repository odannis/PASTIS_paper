try:
    import SFFI.simulation as simu
    import SFFI.sffi as inf
    import SFFI.SBR_progression as S
    import SFFI.Cross_validation as CV
except ImportError:
    import sys
    import os
    dir2 = os.path.abspath('')
    dir1 = os.path.dirname(dir2)
    if not dir1 in sys.path: sys.path.append(dir1)
    if not dir2 in sys.path: sys.path.append(dir2)
    import SFFI.simulation as simu
    import SFFI.sffi as inf
    import SFFI.SBR_progression as S
    import SFFI.Cross_validation as CV

from dataclasses import dataclass
import numpy as np
from simulation_models._common_class import Model
from copy import deepcopy
from jax import vmap
from SFFI.util_inference.inference_help import evaluate_all_base

def timeit(method):
    """Decorator to time a function"""
    import time
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"{method.__name__} took: {te-ts} sec")
        return result
    return timed

class InferenceParameter:
    """data information for the inference algorithm"""
    dt : float
    threshold_sindy : float 
    A_normalisation : bool|None  # If True use a matrix estimated for rescaled MSE, if None minimize MSE
    use_sindy: bool
    use_ensemble_sindy: bool
    use_PASTIS: bool
    use_AIC : bool
    use_BIC : bool
    use_weak_sindy : bool
    convention : str
    D_strength : float = 1
    F_real : np.ndarray|None = None
    type_D : str = "constant"
    x_no_experimental_noise : np.ndarray|None = None
    n_future : int = 0
    experimental_noise : float = 0
    init_params : dict|None = None
    diffusion : str = "Constant" # "Constant" or "Multiplicative"
    auto_selectivity : bool = False
    fast_exploration : bool = False
    loop_on_p_PASTIS : bool = False
    use_loop_on_p_PASTIS : bool = False
    use_lasso: bool = False
    l_threshold_lasso: list = []
    l_threshold_sindy: list = []
    threshold_lasso: float = 0.5
    benchmark_CV_k_validation : bool = False

    def __init__(self, model : Model, threshold_sindy = 0.5, threshold_lasso = 0.5, A_normalisation=True,
                 use_sindy=True, use_PASTIS=True, use_AIC=False, use_ensemble_sindy=False, use_weak_sindy=False, use_ConstrainedSR3=False,
                 convention="Ito", use_CrossValidation=False, use_BIC = False, use_lasso=False, l_threshold_lasso = [], l_threshold_sindy = []):
        self.model = model
        self.init_params = model.init_params
        self.total_base = model.total_base
        self.index_real_base = model.index_real_base
        self.index_real_base_sindy = model.index_real_base_sindy
        self.feature_library = model.feature_library
        
        if use_ConstrainedSR3:
            self.constraint_rhs = model.constraint_rhs # type: ignore
            self.constraint_lhs = model.constraint_lhs # type: ignore
        
        self.threshold_sindy = threshold_sindy
        self.use_sindy = use_sindy
        self.use_PASTIS = use_PASTIS
        self.use_AIC = use_AIC
        self.use_BIC = use_BIC
        self.use_ensemble_sindy = use_ensemble_sindy
        self.use_weak_sindy = use_weak_sindy
        self.use_ConstrainedSR3 = use_ConstrainedSR3
        self.A_normalisation = A_normalisation
        self.convention = convention
        self.use_CrossValidation = use_CrossValidation
        self.use_lasso = use_lasso
        self.threshold_lasso = threshold_lasso
        self.l_threshold_sindy = l_threshold_sindy
        self.l_threshold_lasso = l_threshold_lasso

    def _populate(self, x_train : np.ndarray, x_predictive_error,  para_simulation):
        F_t = self.model.drift_term
        sqrt_diffusion = para_simulation["sqrt_diffusion"]
        self.dt = para_simulation["dt"]
        self.time = x_train.shape[0]*self.dt
        self.D_strength = para_simulation["diffusion_strength"]
        if callable(sqrt_diffusion) :
            self.type_D = "function"
        else:
            self.type_D = "constant"
        self.F_real = np.asarray(vmap(F_t)(x_train)) # np.array([F_t(x) for x in x_train])
        self.F_real_predictive_error = np.asarray(vmap(F_t)(x_predictive_error))
        self.x_train = x_train
        self.x_predictive_error = x_predictive_error
        
    def update_error_real_model(self, model_l0 : S.Exploration, real_nodes, inf_1: inf.SFFI):
        # Compute the base for self.x_train and self.x_predictive_error and store it
        #self.base_evaluated_x_train, _ = evaluate_all_base(inf_1.base, self.x_train, inf_1.use_jax)
        #self.base_evaluated_x_predictive_error, _ = evaluate_all_base(inf_1.base, self.x_predictive_error, inf_1.use_jax)
        error_train, error_predictive = self.compute_error(model_l0, real_nodes, inf_1)
        self.error_real_model = error_train
        self.error_real_model_predictive = error_predictive
        
    def compute_error(self, model_l0 : S.Exploration, nodes, inf_1: inf.SFFI):
        #F_train_ = model_l0.compute_force_set(nodes, real_phi=self.x_train)
        F_predictive_error = model_l0.compute_force_set(nodes, real_phi=self.x_predictive_error)
        #error_train = inf.real_error(F_train_, self.F_real, A=inf_1.A_normalisation)
        error_train = 1
        error_predictive = inf.real_error(F_predictive_error, self.F_real_predictive_error, A=inf_1.A_normalisation)
        return float(error_train), float(error_predictive)
    
    def compute_all_error(self, model_l0 : S.Exploration, nodes, inf_1: inf.SFFI):
        error_train, error_predictive = self.compute_error(model_l0, nodes, inf_1)
        ratio_error_real_model = error_train/self.error_real_model
        ratio_error_real_model_predictive = error_predictive/self.error_real_model_predictive
        return error_train, error_predictive, ratio_error_real_model, ratio_error_real_model_predictive
    
    def copy(self):
        """Return a copy of the object"""
        return deepcopy(self)
        
@dataclass
class EstimationError:
    """Output of the inference algorithm"""
    dt : float
    time : float
    base_infered : list
    real_base : list
    error : float
    method : str
    type_D : str
    D_strength : float
    experimental_noise : float
    ratio_error_real_model : float
    predictive_error : float
    ratio_error_real_model_predictive : float
    SBR_finds_real_model : bool
    real_model_on_pareto_front : bool
    SBR_found_better_minimum : bool
    init_params : dict|None
    time_to_compute : float

    def __init__(self, inference_para : InferenceParameter, base_infered = [], real_base = [], error = 0.0, ratio_error_real_model = 0.0, method = "Unknown", 
                predictive_error = 0.0, ratio_error_real_model_predictive = 0.0, SBR_finds_real_model = False, real_model_on_pareto_front = False, SBR_found_better_minimum = False,
                time_to_compute = 0.0):
        self.dt = inference_para.dt
        self.time = inference_para.time
        self.type_D = inference_para.type_D
        self.D_strength = inference_para.D_strength
        self.experimental_noise = inference_para.experimental_noise
        self.init_params = inference_para.init_params
        self.base_infered = deepcopy(list(base_infered))
        self.real_base = list(real_base)
        self.error = float(error)
        self.method = method
        self.ratio_error_real_model = float(ratio_error_real_model)
        self.predictive_error = float(predictive_error)
        self.ratio_error_real_model_predictive = float(ratio_error_real_model_predictive)
        self.SBR_finds_real_model = SBR_finds_real_model
        self.real_model_on_pareto_front = real_model_on_pareto_front
        self.SBR_found_better_minimum = SBR_found_better_minimum
        self.time_to_compute = time_to_compute
        
    def update_error_from_model_l0(self, model_l0 : S.Exploration, inference_para : InferenceParameter, inf_1: inf.SFFI):
        self.base_infered = deepcopy(list(model_l0.best_nodes))
        self.real_base = list(inference_para.index_real_base)
        self.error, self.predictive_error, self.ratio_error_real_model, self.ratio_error_real_model_predictive = inference_para.compute_all_error(model_l0, model_l0.best_nodes, inf_1)

        
