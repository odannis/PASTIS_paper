import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.Ornstein_Uhlenbeck import OrnsteinUhlenbeck
from _helper._simu_and_save import simu_and_save
from SFFI.sffi import PolynomeInfo
import pysindy as ps
from _helper._database import InferenceParameter
try:
    from ._global_param import num_dot
except:
    from _global_param import num_dot

def random_interaction_matrix(dim, interaction_strength=1, density=0.1, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        seed = np.random.randint(0, 1000)
        rng = np.random.default_rng()
    omega = np.zeros((dim, dim))
    for i in range(dim):
        omega[i, i] = 1     
    while (np.sum(omega!=0) - dim)/(dim*(dim - 1)) < density:
        i,j = rng.integers(0, dim), rng.integers(0, dim)
        if i != j:
            omega[i, j] = rng.choice([-interaction_strength, interaction_strength])
    # for i in range(dim):
    #     omega[i, i] = max(np.sum(np.abs(omega[i, :])) - 1, 0)
    if not np.all(np.real(np.linalg.eigvals(omega)) >= 0.00001):
        print("The matrix is not stable %s" % omega)
        return random_interaction_matrix(dim, interaction_strength, density, seed + 1)
    else:
        return omega
        
model = OrnsteinUhlenbeck(diffusion_strength=100, dt=0.01, n=10**6, diffusion_constant=True)
name_csv = "OrnsteinUhlenbeck_p"

threshold_sindy = 0.5 # For SINDy inference
l_diffusion_strength = None #np.geomspace(0.001, 100, num_dot)
l_n = np.geomspace(100, model.n, num_dot).astype(int)
l_dt = None #np.round(np.geomspace(model.dt, 10, num_dot) / model.dt) * model.dt
l_experimental_noise = None #np.geomspace(0.001, 10, num_dot) 
l_thresholds_sindy = None #np.unique(np.round(np.linspace(0.01, 1, num_dot), 2))
max_dim = 11

if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise, "l_thresholds_sindy" : l_thresholds_sindy}
    
    kwargs["conventions"] = [("Ito", "Constant")#, ("Strato", "Multiplicative")]#, ("Strato", "Multiplicative")]
                            ]
    if len(sys.argv) == 1:
        number_simu = ""
    else:
        number_simu = sys.argv[1]
        
    for dim in range(10, 11):
        kwargs["name_csv"] = name_csv + f"_dim_{dim}"
        omega = random_interaction_matrix(dim, seed=0)   
        model = OrnsteinUhlenbeck(diffusion_strength=model.diffusion_strength, dt=model.dt, omega=omega, n=model.n)   
        para_inference = InferenceParameter(model, threshold_sindy=threshold_sindy, use_sindy=False, use_PASTIS=True, use_AIC=False, use_BIC=False, use_CrossValidation=False)
        para_inference.use_loop_on_p_PASTIS = True
        simu_and_save(para_inference, number_simu=number_simu, **kwargs)
