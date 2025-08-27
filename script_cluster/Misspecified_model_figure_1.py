import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.figure_1_model import Figure1Model
from _helper._simu_and_save import simu_and_save
from _helper._database import InferenceParameter

model = Figure1Model(dt=0.001, n=1_000_000, diffusion_strength=0.4)

name_csv = "Misspecified_model_figure_1"

num_dot = 10
l_diffusion_strength = np.geomspace(0.001, 0.5, num_dot)
l_n = np.geomspace(1_000, model.n*10, num_dot).astype(int)
l_dt = None#np.round(np.geomspace(model.dt, 10, num_dot) / model.dt) * model.dt
l_experimental_noise = None #np.geomspace(0.001, 10, num_dot) 
l_thresholds_sindy = None #np.unique(np.round(np.linspace(0.01, 1, num_dot), 2))

if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise, "l_thresholds_sindy" : l_thresholds_sindy
    }
    kwargs["conventions"] = [("Ito", "Constant")]
    para_inference = InferenceParameter(model, use_PASTIS=True, use_AIC=True, use_BIC=True, use_sindy=True, use_CrossValidation=True)
    
    if len(sys.argv) == 1:
        simu_and_save(para_inference, **kwargs)
    else:
        simu_and_save(para_inference, number_simu=sys.argv[1], **kwargs)
