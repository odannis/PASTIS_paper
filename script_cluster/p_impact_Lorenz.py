import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.lorenz import Lorenz
from _helper._simu_and_save import simu_and_save
from _helper._database import InferenceParameter
from _global_param import num_dot

diffusion_strength = 100
dt = 0.0001
n = 50_000_000

r,b,s = 28., 7/3, 10.
model_critical = Lorenz(r=r, b=b, s=s, diffusion_strength=diffusion_strength, dt=dt, n=n)
name_csv_critical = "lorenz_critical_constant_noise_p"

threshold_sindy = 0.5 # For SINDy inference
l_diffusion_strength = np.geomspace(0.01, 10_000, num_dot)
l_n = np.geomspace(10_000, n, num_dot, dtype=int).astype(int)
l_dt = None #np.round(np.geomspace(model_non_critical.dt, 0.05, num_dot) / model_non_critical.dt) * model_non_critical.dt
l_experimental_noise = np.geomspace(0.001, 1, num_dot) 
l_thresholds_sindy = None #np.unique(np.round(np.linspace(0.01, 1, num_dot), 2))

para_inference_critical = InferenceParameter(model_critical, threshold_sindy=threshold_sindy,
            use_sindy=False, use_PASTIS=True, use_AIC=False, use_BIC=False, use_CrossValidation=False)
para_inference_critical.use_loop_on_p_PASTIS = True
    
if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv_critical, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise, "l_thresholds_sindy" : l_thresholds_sindy}

    kwargs["conventions"] = [("Ito", "Constant") #("Ito_trapeze_large_dt", "Constant_time_correction")#, ("Strato", "Multiplicative"),
                             ]

    kwargs["l_dt"] = None
    kwargs["l_experimental_noise"] = None
    kwargs["l_diffusion_strength"] = None
    #kwargs["l_n"] = None
    kwargs["l_diffusion_vs_time"] = None#[l_diffusion_strength, np.geomspace(100_000, n, num_dot).astype(int)]
    
    if len(sys.argv) == 1:
        kwargs["l_n"] = np.geomspace(1_000, 10_000, 3, dtype=int).astype(int)
        simu_and_save(para_inference_critical, **kwargs)
    else:
        simu_and_save(para_inference_critical, number_simu=sys.argv[1], **kwargs)
