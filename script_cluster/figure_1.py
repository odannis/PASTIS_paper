import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import SFFI.simulation as simu
import SFFI.sffi as inf
import SFFI.util_plot as ut_plot
import numpy as np
from tqdm import tqdm
from jax import jit 
import jax.numpy as jnp
from SFFI import SBR_progression as sbr
import SFFI.util_inference.inference_help as ih
import pandas as pd
from dataclasses import dataclass
import os, sys

@jit
def power_8(x):
    return x**8

@jit
def power_7(x):
    return x**7

@jit
def power_6(x):
    return x**6

@jit
def power_5(x):
    return x**5

@jit
def power_4(x):
    return x**4

@jit
def power_2(x):
    return x**2

@jit
def power_3(x):
    return x**3

@jit
def power_1(x):
    return x

@jit
def power_0(x):
    return jnp.ones_like(x)

d_dim = 1
D = 0.4 #* np.random.random((d_dim,d_dim))
n_simu = 10

@jit
def F_t(x):
    return -x/(1-x**2)**2

@jit
def tan_func(x):
    return jnp.tan(x*np.pi/2)

parameter_simulation = {
    "shape_image" : [d_dim], "n" : 40_000, "dt" : 0.001, "sqrt_diffusion" : np.sqrt(D),
     "first_image" : [0]*d_dim if d_dim > 1 else [0],
    "base" : [F_t], "coefficient" : [1], "over_sampling" : 10, "force_numba" : F_t}

para = {**parameter_simulation,
        "n_differential_equation" : d_dim, "n_futur" : 0, 
        "first_image" : [0]*d_dim if d_dim > 1 else [0], 
        }

start_point = int(1/para["dt"])
n_times = np.linspace(start_point, para["n"], 100, dtype=int)
    
base = para["base"]
base_1 = [power_0, power_1, power_2, power_3, power_4, power_5, power_6, power_7, power_8]
base_2 = [tan_func]

name_base_1 = "\mathcal{B}_{poly}" #"x^0,.,x^{%s}"%len(base_1) #"$x^1, ..., x^{%s}$"%(len(base_1))#ut_plot.get_name_list_function(base_1)
name_base_2 = "\mathcal{B}_{\\tan}" #"\\tan(x)" #ut_plot.get_name_list_function(base_2)
color_base_1 = ut_plot.cmaps[1]
color_base_2 = ut_plot.cmaps[2]


tau = n_times*para["dt"]

@dataclass
class InferenceData:
    time : float
    L: float
    error: float
    delta_I_AIC: float
    I_AIC_1 : float
    I_AIC_2 : float
    base: str
    error_base_1: float
    error_base_2: float
    F_real_square: float = 0
    
def compute_force_set(sffi : inf.SFFI, phi = None):
    if phi is None:
        phi = sffi.phi
    else:
        phi = np.copy(phi)
        phi = phi[:, np.newaxis]
    F = np.zeros_like(phi)
    base_evaluated, _ = ih.evaluate_all_base(sffi.base, phi, sffi.use_jax)
    for i_coeff in range(sffi.coefficients.shape[0]):
        F += sffi.coefficients[i_coeff] * base_evaluated[i_coeff]
    return F

def infer_coefficient_and_cost_function_diff(base_1: list, base_2 : list,  n_times : np.ndarray | list, base_name) -> list[list[inf.SFFI]]:
    l = []
    base_total = base_1 + base_2
    index_base_1 = tuple(i for i in range(len(base_1)))
    index_base_2 = tuple(i for i in range(len(base_1), len(base_total)))
    for n in tqdm(n_times):
        ll = []
        for phi in l_phi:
            inf_tot = inf.SFFI(base_total, phi[:n], para["dt"],
                            use_jax=True, n_futur=para["n_futur"], A_normalisation=True, clean=False,
                            convention="Ito_not_trapeze", diffusion=D)
            sbr_tot = sbr.L0_SBR(inf_tot, start=False)
            F_esti_1 = sbr_tot.compute_force_set(index_base_1)
            F_esti_2 = sbr_tot.compute_force_set(index_base_2)
            
            F_real = para["coefficient"] * F_t(inf_tot.phi)

            F_real_2_avg = np.mean(F_real**2/ (4 * D))
            error_1 = np.mean((F_real - F_esti_1)**2 / (4 * D)) 
            I_AIC_1 = sbr_tot.compute_information(index_base_1)

            error_2 = np.mean((F_real - F_esti_2)**2 / (4 * D))
            I_AIC_2 = sbr_tot.compute_information(index_base_2)
            
            ll.append(InferenceData(n*inf_tot.delta_t, 0, error_1 - error_2, I_AIC_1 - I_AIC_2, I_AIC_1, I_AIC_2, base_name, error_1, error_2, F_real_2_avg))
        l.extend(ll)
    return l


if __name__ == "__main__":    
    if len(sys.argv) == 1:
        i_simu = 1
    else:
        i_simu = int(sys.argv[1])
    l_phi = np.empty((n_simu, parameter_simulation["n"], *parameter_simulation["shape_image"])) if parameter_simulation["shape_image"] != [1] else np.empty((n_simu, parameter_simulation["n"]))
    for i in tqdm(range(n_simu)): 
        phi, delta_t = simu.simulate(**parameter_simulation)
        l_phi[i] = phi
        
    l_data = infer_coefficient_and_cost_function_diff(base_1, base_2, n_times, "Diff")
    data = pd.DataFrame(l_data)
    dir2 = os.path.dirname(os.path.abspath(''))
    name_csv_save = ( dir2 + "/csv/" + "figure_1"
        + ".pkl__" + str(i_simu))
    print(name_csv_save)
    try:
        data.to_pickle(name_csv_save)
    except OSError as e:
        print("Error : ", e)