import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from simulation_models.Ornstein_Uhlenbeck import OrnsteinUhlenbeck
from SFFI.simulation import simulate
import matplotlib.pyplot as plt
import SFFI.SBR_progression as l
import SFFI.sffi as sffi
import matplotlib.markers as ma
from .helper_figure import add_letter
from . import labels_plot as lp

def simu_for_plotting(model = None, seed=1, systematic_exploration=True, convention="Ito",
                      diffusion="Constant", verbose=False, diffusion_constant=True, n=10_000, start=True):
    if model is None:
        alpha = 1
        omega = np.array([
            [1.   , 0.   , -alpha],
            [0.0   , 1.   , 0],
            [0  , 0   , 1. ]])
        model = OrnsteinUhlenbeck(omega=omega, diffusion_strength=1, dt=0.01, n=n, diffusion_constant=diffusion_constant)
    model.fast_random = np.random.default_rng(seed)
    # model.thermalised_time = 0
    para_simu = model.get_parameter_simulation()
    x_simu, dt = simulate(**para_simu)
    x_simu_use = x_simu #+ 1 * np.random.normal(size=x_simu.shape)
    total_base = model.total_base
    #time = dt * np.array(list(range(x_simu.shape[0])))
    # for i in range(omega.shape[0]):
    #     plt.plot(time, x_simu[:,i] )#+ 100*i*model.diffusion_strength*model.dt)
    # plt.plot(x_simu)
    # plt.legend()
    inf_1 = sffi.SFFI(total_base, x_simu_use, dt, use_jax=True,
                      convention=convention, diffusion=diffusion)

    model_l0 = l.L0_SBR(inf_1, use_AIC=True, verbose=verbose, start=start)
    model_l0.compute_information(tuple(model.index_real_base))
    if systematic_exploration:
        model_l0.systematic_exploration()
    model_l0.best_nodes = model_l0.get_best_nodes()
    if verbose:
        print("         " + str(model.index_real_base))
        model_l0.print_top_best_nodes(use_name_base=False)
        model_l0.print_best_nodes()
    
    return model_l0, model

# Create the dict of pareto front for model_l0.d_information which give me the best nodes of each size
def create_pareto_front(model_l0 : "l.L0_SBR"):
    d_pareto_front = {}
    for nodes in model_l0.d_information.keys():
        if len(nodes) not in d_pareto_front.keys():
            d_pareto_front[len(nodes)] = nodes
        else:
            if model_l0.d_information[nodes] > model_l0.d_information[d_pareto_front[len(nodes)]]:
                d_pareto_front[len(nodes)] = nodes
    return dict(sorted(d_pareto_front.items()))

def plot_AIC_vs_n_parameters(model_l0 : l.L0_SBR, index_real_base, zoom=False, ax=None, use_BIC=False, ratio_bottom=0.9, ratio_top=1.1,
                             for_paper=True, letter="", dominant_set=False, pareto_front=True, add_true_model=True):
    if ax is None:
        ax = plt.subplot()
    d_size_domination = model_l0.d_size_domination
    d_information = model_l0.d_information
    if for_paper:
        dx_dt = (model_l0._sffi.phi[1:] - model_l0._sffi.phi[:-1])/model_l0._sffi.delta_t
        dx_dt2 = np.sum(dx_dt*(dx_dt@model_l0._sffi.A_normalisation))*model_l0._sffi.delta_t #type: ignore
        d_information = {key: dx_dt2 - value for key, value in d_information.items()}
    highlight_keys = list((base for key in d_size_domination.keys() for base in d_size_domination[key]))
    key_lengths = [len(key) for key in d_information.keys()]
    values = list((d_information.values()))
    marker = ma.MarkerStyle(marker=".", fillstyle="full")
    marker_dominant = ma.MarkerStyle(marker=".", fillstyle="full")
    marker_aic = ma.MarkerStyle(marker="P", fillstyle="full")
    marker_aicf = ma.MarkerStyle(marker="*", fillstyle="full")
    size_rond = 35
    ax.scatter(key_lengths, values, marker=marker, alpha=0.4, s=size_rond, linewidths=0, label="All models", color="#999999")#color="#0173B2")
    
        ## Dominant set
    if dominant_set:
        values_dominant = [d_information[n] for n in highlight_keys]
        key_lengths_dominant = [len(key) for key in highlight_keys]
        ## Create a dictionary with the number of domninant with the same size
        ax.scatter(key_lengths_dominant, values_dominant, marker=marker_dominant,
                label="Dominant set")
    if pareto_front:
        pareto_front = create_pareto_front(model_l0)
        values_pareto = [d_information[pareto_front[key]] for key in pareto_front.keys()]
        key_lengths_pareto = list(pareto_front.keys())
        ax.scatter(key_lengths_pareto, values_pareto, marker=marker_dominant, s=size_rond,
                label="Pareto front", color=lp.cmaps[-1])
    if add_true_model:
        ax.scatter(len(index_real_base), d_information[tuple(index_real_base)],
                label='True model', marker=marker_aicf, s=size_rond/6, color="green")
    maxima = max if not for_paper else min
    AIC_model = maxima(d_information.keys(), key=d_information.get) #type: ignore
    
    key_min = np.min(list(d_size_domination.keys()))
    PASTIS_models = d_size_domination[key_min]
    if len(PASTIS_models) > 1:
        generator = d_size_domination[key_min]
        PASTIS_model = maxima(generator, key=d_information.get)#type: ignore
    else:
        PASTIS_model = PASTIS_models[0]
    
    # ax.scatter(len(PASTIS_model), d_information[PASTIS_model],
    #            label="PASTIS choice", marker=marker_aicf)
    
    if zoom:
        keys = [i for i in d_size_domination.keys() if i > len(PASTIS_model)]
        bottom = np.min([list((d_information[base] for key in keys for base in d_size_domination[key]))])
        ax.set_ylim(bottom=bottom*ratio_bottom, top=d_information[AIC_model]*ratio_top)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xlabel('Number of parameters')
    ylabel = 'BIC Score' if use_BIC else '$\mathcal{I}$'
    if for_paper:
        ylabel = "$\mathcal{I}$"
        title = "Bias-free loss function"
        if zoom:
            title += " (zoom)"
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    if letter != "":
        add_letter(ax, letter)
    return ax
    
    
if __name__ == "__main__":
    model_l0, model = simu_for_plotting()
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    plot_AIC_vs_n_parameters(model_l0.d_information, model_l0.d_size_domination, model.index_real_base, ax=axs[0])#type: ignore
    plot_AIC_vs_n_parameters(model_l0.d_information, model_l0.d_size_domination, model.index_real_base, zoom=True, ax=axs[1])#type: ignore
    plt.show()