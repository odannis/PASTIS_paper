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

r,b,s = 10., 1., 3.
model_non_critical = Lorenz(r=r, b=b, s=s, diffusion_strength=diffusion_strength, dt=dt, n=n)
name_csv_non_critical = "lorenz_non_critical_constant_noise"

r,b,s = 28., 7/3, 10.
model_critical = Lorenz(r=r, b=b, s=s, diffusion_strength=diffusion_strength, dt=dt, n=n)
name_csv_critical = "lorenz_critical_constant_noise"
if model_critical.diffusion_constant is False:
    name_csv_critical = "lorenz_critical_variable_noise"

l_diffusion_strength = np.geomspace(0.01, 10_000, num_dot)
l_n = np.geomspace(50_000, n, num_dot, dtype=int).astype(int)
l_dt = None #np.round(np.geomspace(model_non_critical.dt, 0.05, num_dot) / model_non_critical.dt) * model_non_critical.dt
l_experimental_noise = None #np.geomspace(0.001, 1, num_dot) 
l_thresholds_sindy = None #np.unique(np.round(np.linspace(0.01, 1, num_dot), 2))

para_inference_critical = InferenceParameter(model_critical, threshold_sindy=threshold_sindy, use_sindy=True, use_PASTIS=True, use_AIC=True, use_BIC=True, use_CrossValidation=True)

if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv_non_critical, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise, "l_thresholds_sindy" : l_thresholds_sindy}

    kwargs["conventions"] = [("Ito", "Constant") #("Ito_trapeze_large_dt", "Constant_time_correction")#, ("Strato", "Multiplicative"),
                             ]

    kwargs["l_dt"] = None
    kwargs["l_experimental_noise"] = None
    kwargs["l_diffusion_strength"] = None
    kwargs["l_diffusion_vs_time"] =  [l_diffusion_strength, np.geomspace(10_000, n, num_dot).astype(int)]

    if len(sys.argv) == 1:
        kwargs["l_n"] = None #np.geomspace(50_000_000, 50_000_000, 1, dtype=int).astype(int)
        kwargs["l_diffusion_vs_time"] = [ np.geomspace(0.01, 10_000, 10), np.geomspace(1_000, 10_000, 1).astype(int)]
        simu_and_save(para_inference_critical, **kwargs)
    else:
        #simu_and_save(para_inference_non_critical, number_simu=sys.argv[1], **kwargs)
        kwargs["name_csv"] = name_csv_critical
        simu_and_save(para_inference_critical, number_simu=sys.argv[1], **kwargs)
