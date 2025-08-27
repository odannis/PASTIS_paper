import os, sys
dir2 = os.path.abspath("")
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import SFFI.util_plot as ut_plot
import numpy as np
import importlib

cmaps = np.array([
    (235, 172, 35),
    (184, 0, 88),
    (0, 140, 249),
    (0, 110, 0),
    (0, 187, 173),
    (209, 99, 230),
    (178, 69, 2),
    (255, 146, 135),
    (89, 84, 214),
    (0, 198, 248),
    (135, 133, 0),
    (0, 167, 108),
    (189, 189, 189)
])/255

cmaps = ut_plot.cmaps

def s_color(index_color, prefactor):
    return tuple(np.array(cmaps[index_color])*prefactor)

diffusion = "Multiplicative"
conv = "Ito"
select = 2
pen = 2

global_dict = {
            f"AIC_Ito_A_True_diffusion_Constant" : ("AIC", cmaps[9]),#cmaps[1]),
             "PASTIS_p_0.001_Ito_A_True_diffusion_Constant" : ("PASTIS", cmaps[4]),
             "PASTIS_p_0.001_Ito PASTIS_exact_A_True_diffusion_Constant": ("PASTIS_g", s_color(5, 0.5)),
             "PASTIS_p_0.001_Ito Pierre_idea_A_True_diffusion_Constant" : ("PASTIS_pierre",  s_color(6, 0.59)),
   
            "STLSQ_threshold_0.5" : ("SINDy", cmaps[5]),
            "Lasso_threshold_1.0" : ("Lasso", cmaps[7]),
            "CrossValidation_threshold_7" : ("CV", cmaps[6]),
            
            f"BIC_True_Ito_A_True_diffusion_Constant" : ("BIC", cmaps[0]),

            "PASTIS_p_0.001_Strato_A_True_diffusion_Multiplicative_Vestergaard" : ("PASTIS-$\sigma$", cmaps[1]),
            "PASTIS_p_0.001_Ito_trapeze_large_dt_A_True_diffusion_Multiplicative_time_correction" : ("PASTIS-$\Delta t$", cmaps[3]),
            
            "PASTIS_p_0.001_Strato_trapeze_A_True_diffusion_Constant_time_correction"  : ("PASTIS $\Delta t$ Strato", cmaps[7]),
            f"PASTIS_p_0.001_Ito_trapeze_A_True_diffusion_Constant_time_correction" : ("PASTIS tr D Cor", cmaps[8]),
            
            f"Total_model_Ito_A_True_diffusion_Constant" : ("Complete basis", cmaps[8]),
            #f"Real_model_{conv}_A_True_n_fut_0_diffusion_{diffusion}" : ("Exact basis", cmaps[9]),
            f"Real_model_Ito_A_True_diffusion_Constant" : ("Exact basis", cmaps[2]),
            
            f"Total_model_Ito_shift_A_True_n_fut_1_diffusion_{diffusion}" : ("Total model shift", s_color(7, 0.99)),
            f"Real_model_Ito_shift_A_True_n_fut_1_diffusion_{diffusion}" : ("Real model shift", s_color(8, 0.99)),
            f"Total_model_{conv}_A_True_n_fut_0_diffusion_{diffusion}_Vestergaard" : ("Total model Vest.", s_color(7, 0.98)),
            f"Real_model_{conv}_A_True_n_fut_0_diffusion_{diffusion}_Vestergaard" : ("Real model Vest.", s_color(8, 0.98)),
            
            # f"Total_model_{method}_A_True_n_fut_0" : s_color(7, 0.97),
            # f"Real_model_{method}_A_True_n_fut_0" : s_color(8, 0.97),
            
            f"Total_model_{conv}_A_True_n_fut_0_diffusion_Constant_time_correction" : ("Total model $\Delta t$", s_color(7, 0.5)),
            f"Real_model_{conv}_A_True_n_fut_0_diffusion_Constant_time_correction" : ("Real model $\Delta t$", s_color(8, 0.5)),        
            }

color_dict = {key : value[1] for key, value in global_dict.items()}
color_dict_name_to_color = {value[0] : value[1] for key, value in global_dict.items()}

d_correct_label = {"D_strength" : r"$D$",
                  "time" : r"Total time $\tau$",
                  "Accuracy_model" : "Accuracy",
                  "error" : "Error $\mathcal{E}_r$",
                  "ratio_error_real_model" : r"$\mathcal{E}(\hat{F}_{\mathcal{B}})/\mathcal{E}(\hat{F}_{\mathcal{B}^*})$",#"$\\frac{\mathcal{E}_r(\hat{f})}{\mathcal{E}_r(\hat{f}_{true})}$",
                  "Exact_model_found" : "Exact Match", #Accuracy",#"Strict accuracy",
                  "experimental_noise" : "Measurement noise $\sigma$",
                  "dt" : "$\Delta t$",
                  "ratio_error_real_model_predictive" : "Prediction error/True model error",
                  "predictive_error" : "Prediction error",
                  }

label_pareto_front = "" #r"$\text{Unreachable}$" "\n" r"$\text{accuracy}$" #r"$\text{True model} \notin$" "\n" r"$\text{Pareto front}$"

def find_key_for_value_in_global_dict(target_value):
    for key, value in global_dict.items():
        if value[0] == target_value:
            return key
    return None
