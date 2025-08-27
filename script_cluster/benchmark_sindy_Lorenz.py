import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.lorenz import Lorenz
from _helper._simu_and_save import simu_and_save
from SFFI.sffi import PolynomeInfo
import pysindy as ps
from _helper._database import InferenceParameter
from _global_param import num_dot, threshold_sindy

diffusion_strength = 100
dt = 0.0001
n = 50_000_000

r,b,s = 28., 7/3, 10.
model_critical = Lorenz(r=r, b=b, s=s, diffusion_strength=diffusion_strength, dt=dt, n=n)
name_csv = "lorenz_critical_constant_noise"
if model_critical.diffusion_constant is False:
    name_csv = "lorenz_critical_variable_noise"
name_csv += "_benchmark_sindy"

l_diffusion_strength = np.geomspace(0.01, 10_000, num_dot)
l_n = np.geomspace(10_000, n, num_dot, dtype=int).astype(int)
l_dt = None #np.round(np.geomspace(model_non_critical.dt, 0.05, num_dot) / model_non_critical.dt) * model_non_critical.dt
l_experimental_noise = None #np.geomspace(0.001, 1, num_dot) 
l_threshold_sindy = np.unique(np.round(np.linspace(0.01, 2, num_dot), 2))
l_threshold_lasso = np.unique(np.round(np.linspace(1, 50, num_dot), 2))

para_inference = InferenceParameter(model_critical, threshold_sindy=threshold_sindy, use_sindy=True,
                                             use_PASTIS=True, use_AIC=False, use_BIC=False, use_CrossValidation=True,
                                             use_lasso=True, l_threshold_lasso=l_threshold_lasso, l_threshold_sindy=l_threshold_sindy)
para_inference.benchmark_CV_k_validation = True
para_inference.use_loop_on_p_PASTIS = True

# para_inference.use_AIC = False
# para_inference.use_BIC = False
# para_inference.use_CrossValidation = True
# para_inference.use_lasso = False
# para_inference.use_sindy = False
# para_inference.use_PASTIS = False
    
if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise,  "conventions": None}

    kwargs["conventions"] = [("Ito", "Constant") #("Ito_trapeze_large_dt", "Constant_time_correction")#, ("Strato", "Multiplicative"),
                             ]

    kwargs["l_dt"] = None
    kwargs["l_experimental_noise"] = None
    kwargs["l_diffusion_strength"] = None
    kwargs["l_n"] = [l_n[-1]]
    kwargs["l_diffusion_vs_time"] = None #[l_diffusion_strength, np.geomspace(100_000, n, num_dot).astype(int)]
    
    if len(sys.argv) == 1:
        kwargs["l_n"] = np.geomspace(50_000_000, 50_000_000, 3, dtype=int).astype(int)
        para_inference.model.diffusion_strength = 1
        kwargs["name_csv"] += "_small_diffusion_strength"
        para_inference.use_lasso = False
        para_inference.use_sindy = False
        print(para_inference.model.get_parameter_simulation())
        simu_and_save(para_inference, **kwargs)
    else:
        simu_and_save(para_inference, number_simu=sys.argv[1], **kwargs)
        para_inference.model.diffusion_strength = 0.01
        kwargs["name_csv"] += "_small_diffusion_strength"
        kwargs["l_n"] = np.geomspace(10_000, n/10, num_dot, dtype=int).astype(int)
        simu_and_save(para_inference, number_simu=sys.argv[1], **kwargs)
        
        
