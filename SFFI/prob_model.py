from scipy.special import erf
import numpy as np 
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from . import SBR_progression as sbr
    
def obtain_gap(model_l0 : "sbr.L0_SBR"):
    d_pareto_front = create_pareto_front(model_l0)
    sorted_size_nodes = sorted(d_pareto_front.keys(), reverse=False)
    sorted_nodes = [d_pareto_front[k] for k in sorted_size_nodes]
    l_info_sorted_nodes = [model_l0.d_information[best_nodes] for best_nodes in sorted_nodes]
    return sorted_nodes, l_info_sorted_nodes

def create_pareto_front(model_l0: "sbr.L0_SBR"):
    full_model = tuple(range(model_l0._sffi.coefficients.shape[0]))
    model_l0.compute_information(full_model)
    d_pareto_front = {len(full_model): full_model}
    current_model = full_model
    while len(current_model) > 0:
        explore_around(model_l0, current_model)
        next_candidates = [
            mod for mod in model_l0.d_information.keys()
            if len(mod) == len(current_model) - 1
        ]
        if not next_candidates:
            break
        best_candidate = max(next_candidates, key=lambda mod: model_l0.d_information[mod])
        d_pareto_front[len(best_candidate)] = best_candidate
        current_model = best_candidate
    return dict(sorted(d_pareto_front.items()))

def explore_around(model_l0 : "sbr.L0_SBR", best_nodes : tuple, use_permutation=False):
    for i in best_nodes:
        initial_nodes = set(best_nodes)
        initial_nodes.remove(i)
        model_l0.compute_information(tuple(initial_nodes))
        for j in range(model_l0._sffi.coefficients.shape[0]):
            if j not in best_nodes:
                nodes_try = initial_nodes.copy()
                nodes_try.add(j)
                nodes_try = tuple(sorted(set(nodes_try)))
                model_l0.compute_information(nodes_try)
    if use_permutation:
        for j in range(model_l0._sffi.coefficients.shape[0]):
            if j not in best_nodes:
                nodes_try = set(best_nodes).copy()
                nodes_try.add(j)
                nodes_try = tuple(sorted(set(nodes_try)))
                model_l0.compute_information(nodes_try)

def find_best_model(model_l0 : "sbr.L0_SBR", plot=False, ax=None):
    sorted_nodes, l_info_sorted_nodes = obtain_gap(model_l0)
    l_model, l_proba_model = [], []
    # select the nodes with the maximum value in self.d_information
    for i in range(len(sorted_nodes)):
        proba = get_proba_model(i, l_info_sorted_nodes)
        #print(f"Model {sorted_nodes[i]} : {l_info_sorted_nodes[i]} with proba {proba}")
        l_proba_model.append(proba)
        l_model.append(sorted_nodes[i])
    if plot:
        if ax is None:
            fig, ax = plt.subplots()
        for x, (info, proba) in enumerate(zip(l_info_sorted_nodes, l_proba_model)):
            ax.plot(x, info, 'o')
            ax.annotate(f"{proba:.2f}", (x, info), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=5)
        
    model_out = max(l_model, key=lambda x: l_info_sorted_nodes[l_model.index(x)])
    return model_out

def get_proba_model(i, l_info_sorted_nodes):
    max_free_parameters = len(l_info_sorted_nodes) - i
    min_free_parameters = 1
    proba_no_FN, proba_no_FP = 1, 1
    if i != len(l_info_sorted_nodes) - 1:
        gap_model_more_complex = l_info_sorted_nodes[i+1] - l_info_sorted_nodes[i]
        proba_no_FN = 1 - cdf_delta_I(gap_model_more_complex, max_free_parameters)
        #print(f"Proba no FN : {proba_no_FN}, gap : {gap_model_more_complex}")
    if i != 0:
        max_free_parameters = len(l_info_sorted_nodes) - i + 1
        gap_model_less_complex = l_info_sorted_nodes[i] - l_info_sorted_nodes[i-1]
        proba_no_FP = cdf_delta_I(gap_model_less_complex, max_free_parameters)
        #print(f"Proba no FP : {proba_no_FP}, gap : {gap_model_less_complex}")
    return proba_no_FN*proba_no_FP
    r#eturn proba_no_FN

def cdf_delta_I(delta_I, free_parameters):
    return erf(np.sqrt(delta_I))**(free_parameters)

def plot_proba_model(best_nodes, model_l0 : "sbr.L0_SBR"):
    get_proba_model(best_nodes, model_l0)