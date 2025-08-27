import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.gray_scott import GrayScott
from _helper._simu_and_save import simu_and_save
from _helper._database import InferenceParameter

diffusion_strength = 10**(-3)
dt = 0.01
n = 5_000
shape_image = (2, 100, 100)

model = GrayScott(dt=dt, n=n, diffusion_strength=diffusion_strength, shape_image=shape_image, over_sampling=10, thermalised_time=10,)
name_csv = "grayscott_p"

num_dot = 20
l_diffusion_strength = np.geomspace(diffusion_strength/100, diffusion_strength*100, num_dot)
l_n = np.geomspace(int(0.1/dt), n, num_dot).astype(int)
l_dt = None#np.geomspace(model.dt, model.dt*100, num_dot) 
l_experimental_noise = np.geomspace(0.001, 1, num_dot) 
l_thresholds_sindy = None #np.unique(np.round(np.linspace(0.01, 1, num_dot), 2))

para_inference_critical = InferenceParameter(model,
            use_sindy=False, use_PASTIS=True, use_AIC=False, use_BIC=False, use_CrossValidation=False)
para_inference_critical.use_loop_on_p_PASTIS = True
    
if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise, "l_thresholds_sindy" : l_thresholds_sindy}

    kwargs["conventions"] = [("Ito", "Constant"), #("Ito_trapeze", "Multiplicative"),
                             ]

    kwargs["l_dt"] = None
    kwargs["l_experimental_noise"] = None
    kwargs["l_diffusion_strength"] = None
    #kwargs["l_n"] = None
    #kwargs["l_diffusion_vs_time"] = [l_diffusion_strength, np.geomspace(100_000, n, num_dot).astype(int)]
    
    if len(sys.argv) == 1:
        model.n = 100
        kwargs["l_n"] = np.linspace(10, 100, 10, dtype=int)
        kwargs["l_diffusion_strength"] = None
        simu_and_save(para_inference_critical, **kwargs)
    else:
        simu_and_save(para_inference_critical, number_simu=sys.argv[1], **kwargs)
