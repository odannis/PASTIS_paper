import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

from dataclasses import dataclass
from tqdm import tqdm
from script_figure import AIC_length_model_OU as AIC_length_model
from script_cluster.Ornstein_Uhlenbeck_dimension import random_interaction_matrix
from simulation_models.Ornstein_Uhlenbeck import OrnsteinUhlenbeck
import numpy as np
import sys, os
import pandas as pd

@dataclass
class Data:
    nodes : tuple
    info : float
    info_delta : float
    i_simu : int

if __name__ == "__main__":    
    if len(sys.argv) == 1:
        i_simu = 1
    else:
        i_simu = int(sys.argv[1])
    
    for dim in range(3, 4):
        omega = random_interaction_matrix(dim, seed=0)
        if dim==3:
            model = OrnsteinUhlenbeck(omega=omega, diffusion_strength=100, dt=0.01, n=10_000)
        else:
            model = OrnsteinUhlenbeck(omega=omega, diffusion_strength=100, dt=0.01, n=30_000)
        name_csv = "large_deviation_dim_%s" % model.omega.shape[0]

        rng = np.random.default_rng(i_simu)    
        l_data = []
        seed = rng.integers(0, 2000)
        systematic_exploration = True
        if model.omega.shape[0] > 3:
            systematic_exploration = False
        model_l0, model = AIC_length_model.simu_for_plotting(model=model, seed=seed, verbose=False, n=model.n,
                                systematic_exploration=systematic_exploration,
                                convention="Ito", diffusion="Constant")
        #model_l0.systematic_exploration(max_length=len(model.index_real_base)+2, model_base=model.index_real_base)
        for nodes in model_l0.d_information.keys():
            l_data.append(
                Data(
                    tuple(sorted(nodes)),
                    model_l0.compute_information(nodes),
                    model_l0.compute_information(nodes) - model_l0.compute_information(tuple(model.index_real_base)),
                    i_simu
                    )
                )

        data = pd.DataFrame(l_data)
        dir2 = os.path.dirname(os.path.abspath(''))
        name_csv_save = ( dir2 + "/csv/" + name_csv
            + ".pkl__" + str(i_simu))
        print(name_csv_save)
        try:
            data.to_pickle(name_csv_save)
        except OSError as e:
            print("Error : ", e)