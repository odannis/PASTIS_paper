import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.lotka_volterra import LotkaVolterra
from _helper._simu_and_save import simu_and_save
from SFFI.sffi import PolynomeInfo
from _helper._database import InferenceParameter
try:
    from ._global_param import num_dot
except:
    from _global_param import num_dot

def construct_A(alpha_1, alpha__1, dim):
   A = np.zeros((dim, dim))
   for i in range(dim):
      A[i, i] = 1
      A[i, (i+1)%dim] = alpha_1
      A[i, (i-1)%dim] = alpha__1
   return A

alpha_1 = 1
alpha__1 = -1
dim = 7

A = construct_A(alpha_1, alpha__1, dim)
r = np.ones(A.shape[0])
A[0,2] = 1
A[2,0] = 1
A[2,4] = 1
A[4,2] = 1

model =  LotkaVolterra(A=A, r=r, diffusion_strength=0.05, dt = 0.01, n=10**6, simple_base=True)
name_csv= "lotka_volterra_dim_%s"%A.shape[0]
name_csv += "_benchmark_sindy"


l_diffusion_strength = np.geomspace(model.diffusion_strength/100, model.diffusion_strength*10, num_dot)
l_n = np.geomspace(model.n/1000, model.n, num_dot).astype(int)
l_dt = np.round(np.geomspace(model.dt, model.dt*100, num_dot) / model.dt) * model.dt
l_experimental_noise = None#np.geomspace(0.01, 1, num_dot)
l_threshold_sindy = np.unique(np.round(np.linspace(0.001, 1, 20), 2))
l_threshold_lasso = np.unique(np.round(np.linspace(0.001, 0.1, 20), 2))

if __name__ == "__main__":
    kwargs = {"name_csv" : name_csv, "l_diffusion_strength" : l_diffusion_strength, "l_n" : l_n, "l_dt" : l_dt,
              "l_experimental_noise" : l_experimental_noise,  "conventions" : None }
    kwargs["conventions"] = [("Ito", "Constant")#, ("Strato", "Multiplicative"), #("Strato", "Multiplicative_Vestergaard"),
                            # ("Strato", "Multiplicative_time_correction"), 
                             ]
    kwargs["l_n"] = [l_n[-1]]
    kwargs["l_dt"] = None
    kwargs["l_diffusion_strength"] = None
    kwargs["l_experimental_noise"] = None
    para_inference = InferenceParameter(model, use_sindy=True, use_PASTIS=True, use_AIC=False, 
                                        use_BIC=False, use_CrossValidation=True, use_lasso=True,
                                         l_threshold_sindy=l_threshold_sindy, l_threshold_lasso=l_threshold_lasso)
    para_inference.benchmark_CV_k_validation = True
    para_inference.use_loop_on_p_PASTIS = True
    
    if len(sys.argv) == 1:
        kwargs["l_diffusion_strength"] = [0]
        kwargs["l_dt"] = None
        model.n = 1_00
        kwargs["name_csv"] = "Test_lotka"
        simu_and_save(para_inference, **kwargs)
    else:
        print("Number of simulation %s"%sys.argv[1])
        #kwargs["l_dt"] = None
        simu_and_save(para_inference, number_simu=sys.argv[1], **kwargs)