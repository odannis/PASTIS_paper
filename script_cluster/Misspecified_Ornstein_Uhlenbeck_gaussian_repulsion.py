import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.Ornstein_Uhlenbeck import OrnsteinUhlenbeck
from _helper._simu_and_save import simu_and_save
from _helper._database import InferenceParameter

omega = np.array([[2., 2. ],
                  [-2., 2.]])

model = OrnsteinUhlenbeck(diffusion_strength=1, dt=0.01, n=10**6, omega=omega,
                          use_repulsive_gaussian=True, degree_polynome=10)

name_csv = "OrnsteinUhlenbeck_repulsive_gaussian"

threshold_sindy = 0.5 # For SINDy inference
num_dot = 20
l_diffusion_strength = None #np.geomspace(0.1, 100, num_dot)
l_n = np.geomspace(1_000, model.n*10, num_dot).astype(int)
l_dt = None#np.round(np.geomspace(model.dt, 10, num_dot) / model.dt) * model.dt
l_experimental_noise = None #np.geomspace(0.001, 10, num_dot) 
l_thresholds_sindy = None #np.unique(np.round(np.linspace(0.01, 1, num_dot), 2))

if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise, "l_thresholds_sindy" : l_thresholds_sindy
    }
    kwargs["conventions"] = [("Ito", "Constant")]
    para_inference = InferenceParameter(model, threshold_sindy=threshold_sindy, use_PASTIS=True, use_AIC=True, use_BIC=True,
                                        use_sindy=True, use_CrossValidation=True)
    if len(sys.argv) == 1:
        #kwargs["l_diffusion_strength"] = None
        # kwargs["l_experimental_noise"] = None
        # kwargs["l_dt"] = None
        # kwargs["l_n"]  = None #np.geomspace(10_000, 1_00_000, 10).astype(int)
        # print(kwargs["l_n"])
        # kwargs["l_thresholds_sindy"] = None
        # kwargs["name_csv"] = "Test"
        # kwargs["conventions"] = ["Ito"]
        # para_inference.use_ensemble_sindy = False
        # para_inference.use_weak_sindy = False
        simu_and_save(para_inference, **kwargs)
    else:
        simu_and_save(para_inference, number_simu=sys.argv[1], **kwargs)
