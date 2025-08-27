import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.lorenz import Lorenz
from simulation_models.Ornstein_Uhlenbeck import OrnsteinUhlenbeck
from simulation_models.lotka_volterra import LotkaVolterra
from _helper._simu_and_save import simu_and_save
from SFFI.sffi import PolynomeInfo
import pysindy as ps
from _helper._database import InferenceParameter
from Lotka_Volterra_simple import model as model_LV
from _global_param import num_dot

diffusion_strength = 100
dt = 0.0001
n = 50_000_000
r,b,s = 28., 7/3, 10.
model = Lorenz(r=r, b=b, s=s, diffusion_strength=diffusion_strength, dt=dt, n=n, diffusion_constant=True)

#model = model_LV
# omega = random_interaction_matrix(10)
# model = OrnsteinUhlenbeck(diffusion_strength=100, dt=0.001, n=10**6, diffusion_constant=True, omega=omega)
max_dt = 0.1

name_csv = ""
if type(model) is Lorenz:
    name_csv = "lorenz_critical_constant_noise"
    if model.diffusion_constant is False:
        name_csv = "lorenz_critical_variable_noise"
elif type(model) is OrnsteinUhlenbeck:
    name_csv = "OrnsteinUhlenbeck_dim_%s"%model.omega.shape[0]
elif type(model) is LotkaVolterra:
    max_dt = 1
    name_csv = "LotkaVolterra"

num_dot = 10
l_dt = np.round(np.geomspace(model.dt, max_dt, num_dot) / model.dt) * model.dt
# l_dt = l_dt[np.argsort(-1*l_dt)]
l_experimental_noise = np.geomspace(0.001, 5, num_dot) 

para_inference_critical = InferenceParameter(model, use_sindy=False, use_PASTIS=True, use_AIC=False)
    
if __name__ == "__main__":
    kwargs_dt = {"name_csv" : name_csv, "l_dt" : l_dt, 
                 "conventions": [("Ito_trapeze_large_dt", "Constant_time_correction"), ("Ito", "Constant"), ]}
                                 #("Strato_trapeze", "Constant_time_correction"),]} ("Ito_trapeze_large_dt", "Constant_time_correction"), 
                                 #("Ito_trapeze_large_dt", "Multiplicative_time_correction"), ("Strato_trapeze", "Multiplicative_time_correction"),]}
    
    kwargs_noise = {"name_csv" : name_csv, "l_experimental_noise" : l_experimental_noise, 
                    "conventions": [("Strato", "Multiplicative_Vestergaard"), ("Ito", "Constant")]} #("Strato", "Constant_Vestergaard")


    if len(sys.argv) == 1:
        model.n = 10000
        simu_and_save(para_inference_critical, **kwargs_dt)
        simu_and_save(para_inference_critical, **kwargs_noise)
    else:
        simu_and_save(para_inference_critical, number_simu=sys.argv[1], **kwargs_dt)
        simu_and_save(para_inference_critical, number_simu=sys.argv[1], **kwargs_noise)
