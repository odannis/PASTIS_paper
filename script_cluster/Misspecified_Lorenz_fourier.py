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
from SFFI.util_inference.fourier import Fourier

diffusion_strength = 100
dt = 0.0002
n = 50_000_000

r,b,s = 28., 7/3, 10.
model_critical = Lorenz(r=r, b=b, s=s, diffusion_strength=diffusion_strength, dt=dt, n=n, diffusion_constant=True,
                        total_base=Fourier(3, order=2))
name_csv_critical = "lorenz_critical_constant_noise_fourier"

threshold_sindy = 0.5 # For SINDy inference
num_dot = 10
l_diffusion_strength = None#np.geomspace(0.01, 10_000, num_dot)
l_n = np.geomspace(10_000, n, num_dot).astype(int)
l_dt = None #np.round(np.geomspace(model_critical.dt, 0.05, num_dot) / model_critical.dt) * model_critical.dt
l_experimental_noise = None #np.geomspace(0.001, 1, num_dot) 
l_thresholds_sindy = None #np.unique(np.round(np.linspace(0.01, 1, num_dot), 2))

para_inference_critical = InferenceParameter(model_critical, threshold_sindy=threshold_sindy, use_sindy=True, use_PASTIS=True, use_AIC=True,
                                                use_ensemble_sindy=False, use_weak_sindy=False, use_CrossValidation=True, use_BIC=True)
    
if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv_critical, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise, "l_thresholds_sindy" : l_thresholds_sindy}

    kwargs["conventions"] = [("Ito", "Constant"), #("Strato", "Multiplicative"),
                             ]

    #kwargs["l_thresholds_sindy"] = None
    #kwargs["l_diffusion_strength"] = None
    kwargs["l_dt"] = None
    kwargs["l_experimental_noise"] = None
    #kwargs["l_n"] = None
    
    if len(sys.argv) == 1:
        model_critical.n = 100
        kwargs["l_n"] = np.geomspace(10_000, 100_000, 1, dtype=int).astype(int)
        simu_and_save(para_inference_critical, **kwargs)
    else:
        simu_and_save(para_inference_critical, number_simu=sys.argv[1], **kwargs)
