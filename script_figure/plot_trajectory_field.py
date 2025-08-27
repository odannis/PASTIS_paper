import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

from brokenaxes import BrokenAxes
import numpy as np
import matplotlib.pyplot as plt
from _helper._load_csv import read_csv
from simulation_models._common_class import Model
from SFFI import util_plot as ut_plot
from .helper_figure import add_letter
import jax
    
def plot_trajectory_field(name_csv, model : type[Model],  parameter = "D_strength", parameters_value_plot = None,
                           ax=None, max_time=None, df=None,  cmap="seismic", start_time=False, x=None):
    if ax is None:
        fig, ax = plt.subplots()
    if df is None:
        df = read_csv(name_csv)

    parameter_values = np.sort(list(df[parameter].unique()))
    if parameters_value_plot is None:
        index_parameters = [parameter_values.shape[0]-1] #[0, parameter_values.shape[0]//2, parameter_values.shape[0]-1]
        parameters_value_plot = [parameter_values[i] for i in index_parameters]
    else:
        index_parameters = [0]
        parameter_values = parameters_value_plot

    row_selected = df[df[parameter] == df[parameter].max()].iloc[0]
    dict_para = row_selected.to_dict()
    model_wrapper = model(**dict(dict_para["init_params"]))
    dt = dict_para["dt"]
    if max_time is not None:
        dict_para["time"] = max_time
        dict_para["n"] = int(max_time/dict_para["dt"])
    print("index real base" , len(model_wrapper.index_real_base))
    print("total base", len(model_wrapper.total_base))
    
    for i, index in enumerate(index_parameters):
        print(parameter_values)
        dict_para[parameter] = parameter_values[index]
        model_wrapper.rng = np.random.default_rng(0)
        
        if x is None:
            x, dt = model_wrapper.simu_from_EstimationError(dict_para)
        vmin = np.min(x)
        vmax = np.max(x)
        if np.isnan(np.sum(x)):
            print("Nan value")
        times = dt*np.arange(x.shape[0])
        if start_time:
            ax.imshow(x[0,0], cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            ax.imshow(x[-1,0], cmap=cmap, vmin=vmin, vmax=vmax)
        # ax.margins(x=0)
        # ax.margins(y=0)
        # ax.set_xlabel('Time')
        ax.set_yticks([])
        ax.set_xticks([])   
        # for spine in ax.spines.values():
        #     spine.set_visible(False)    
    return x
        