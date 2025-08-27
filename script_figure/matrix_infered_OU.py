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
import matplotlib.gridspec as gridspec

    
def simu_for_plotting(model = None):
    if model is None:
        alpha = 0.4
        omega = np.array([
            [1.   , 0.   , alpha],
            [alpha   , 1.   , 0],
            [0  , -alpha   , 1. ]])
        model = OrnsteinUhlenbeck(omega=omega, diffusion_strength=10, dt=0.01, n=1_000_000)
    model.fast_random = np.random.default_rng(4)
    para_simu = model.get_parameter_simulation()
    x_simu, dt = simulate(**para_simu)
    x_simu_use = x_simu #+ 1 * np.random.normal(size=x_simu.shape)
    total_base = model.total_base
    #time = dt * np.array(list(range(x_simu.shape[0])))
    # for i in range(omega.shape[0]):
    #     plt.plot(time, x_simu[:,i] )#+ 100*i*model.diffusion_strength*model.dt)
    # plt.plot(x_simu)
    # plt.legend()
    inf_1 = sffi.SFFI(total_base, x_simu_use, dt, para_simu["shape_image"][0], use_jax=True)

    model_l0 = l.L0_SBR(inf_1)
    model_l0.systematic_exploration()
    model_l0.best_nodes = model_l0.get_best_dominating_node()
    print("         " + str(model.index_real_base))
    model_l0.print_top_best_nodes(use_name_base=False)
    model_l0.print_best_nodes()
    
    return model_l0, model

def obtain_infered_matrix_and_real_matrix(l_index_base, dim, coefficients):
    infered_matrix = np.zeros((dim, dim))
    for i, index_base in enumerate(l_index_base):
        index_variable = (index_base // dim) - 1
        index_dimension = index_base % dim
        if index_variable != -1: #Not a constant infered
            infered_matrix[index_dimension, index_variable] = -coefficients[i]
    return infered_matrix

def plot_matrix_model_l0(model_l0 : l.L0_SBR, model : OrnsteinUhlenbeck, 
                         cmap='bwr', subplot_spec = gridspec.GridSpec(1, 1)[0], fig=None):
    if fig is None:
        fig = plt.figure(layout="constrained")
    
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=subplot_spec)
    big_ax = fig.add_subplot(subplot_spec)
    big_ax.axis('off')
    
    l_ = [model.omega]
    title = ["True", "AIC", "Pastis"]
    for i, selectivity in enumerate([0, model_l0.selectivity]):
        model_l0.selectivity = selectivity
        model_l0.best_nodes = model_l0.get_best_dominating_node()
        dim = model.omega.shape[0]
        coefficients_model_l0 = model_l0.get_coefficients_model(model_l0.best_nodes)
        matrix_l0 = obtain_infered_matrix_and_real_matrix(model_l0.best_nodes, dim, coefficients_model_l0)
        l_.append(matrix_l0)
    vmin = np.min([np.min(matrix) for matrix in l_])
    vmax = np.max([np.max(matrix) for matrix in l_])
    v = max(abs(vmin), abs(vmax))
    l_ax = []
    
    l_ = [l_[0], l_[2], l_[1]]
    title = [title[0], title[2], title[1]]
    for i, diff_matrix in enumerate(l_):
        ax = fig.add_subplot(inner[i])
        l_ax.append(ax)
        im = ax.imshow(diff_matrix, cmap=cmap, interpolation=None, vmin=-v, vmax=v)
        ax.set_title(title[i], pad=1)
        ax.set_yticks([])
        ax.set_xticks([])
    colorbar = fig.colorbar(im, ax=l_ax, shrink=0.5)
    return big_ax

def plot_error_matrix_model_l0(model_l0 : l.L0_SBR, model : OrnsteinUhlenbeck, 
                         cmap='bwr', subplot_spec = gridspec.GridSpec(1, 1)[0], fig=None):
    if fig is None:
        fig = plt.figure(layout="constrained")
    
    inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=subplot_spec)

    l_ = []
    title = ["Error AIC", "Error Pastis"]
    for i, selectivity in enumerate([0, model_l0.selectivity]):
        model_l0.selectivity = selectivity
        model_l0.best_nodes = model_l0.get_best_dominating_node()
        dim = model.omega.shape[0]
        coefficients_model_l0 = model_l0.get_coefficients_model(model_l0.best_nodes)
        matrix_l0 = obtain_infered_matrix_and_real_matrix(model_l0.best_nodes, dim, coefficients_model_l0)
        diff_matrix = (model.omega - matrix_l0)
        ## Divide diff_matrix by the maximum of the absolute value between the two matrices for each scalar in matrix
        #diff_matrix = diff_matrix / np.maximum(np.abs(model.omega), np.abs(matrix_l0))
        ## Replace nan by 0
        #diff_matrix = np.nan_to_num(diff_matrix)
        l_.append(diff_matrix)
    vmin = min([np.min(matrix) for matrix in l_])
    vmax = max([np.max(matrix) for matrix in l_])
    v = max(abs(vmin), abs(vmax))
    l_ax = []
    for i, diff_matrix in enumerate(l_):
        ax = fig.add_subplot(inner[i])
        l_ax.append(ax)
        im = ax.imshow(diff_matrix, cmap=cmap, interpolation=None, vmin=-v, vmax=v)
        ax.set_title(title[i])
        ax.set_yticks([])
        ax.set_xticks([])
    colorbar = fig.colorbar(im, ax=l_ax, shrink=0.5)


if __name__ == "__main__":
    model_l0, model = simu_for_plotting()
    plot_matrix_model_l0(model_l0, model)
    plt.show()
    print("Done")
    