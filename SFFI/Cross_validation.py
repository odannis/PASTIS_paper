from matplotlib.pylab import permutation
import numpy as np
from .sffi import SFFI
from .util_inference import inference_help as ih
import matplotlib.pyplot as plt
from functools import cache
from tqdm import tqdm
from ._exploration_models import Exploration
import time
from jax import numpy as jnp

def shrink_matrix(M : np.ndarray, nodes : list[int], nodes_2 : list[int]|None = None):
    M = M[nodes, :]
    if nodes_2 is None:
        M = M[:, nodes]
    else:
        M = M[:, nodes_2]
    return M

class CrossValidation(Exploration): # https://arxiv.org/pdf/1406.4802.pdf
    def __init__(self,  sffi : SFFI, verbose : bool = False, start = True, l_seed_model = [],
                hard_stop = True, fast_exploration = False, use_STLSQ : bool = False, k_validation : int = 7):
        self.verbose = verbose
        self._sffi = sffi 
        self.D = sffi.D
        self.use_STLSQ = use_STLSQ
        self.d_information = {}
        self.d_helper = {}
        self.d_size = {}
        self.k_validation = k_validation

        phi_start = np.asarray(sffi.phi[:-1])
        delta_phi_start = np.asarray((sffi.phi[1:] - sffi.phi[:-1])/sffi.delta_t)
        assert phi_start.shape[0] == delta_phi_start.shape[0]
        permutation = np.random.permutation(phi_start.shape[0])
        phi_start = phi_start[permutation]
        delta_phi_start = delta_phi_start[permutation]
        phi_part = split_array_into_equal_parts(phi_start, k_validation)
        delta_phi_part = split_array_into_equal_parts(delta_phi_start, k_validation)
        self.l_B_and_F_projected = []
        t = time.time()
        for phi, delta_phi in zip(phi_part, delta_phi_part):
            B, F_projected = np.zeros_like(sffi.B), np.zeros_like(sffi.F_projected)
            memory_size_GB = phi.nbytes*sffi.n_base / 10**9 
            n_cut = int(memory_size_GB // 2 + 1)
            for _n_cut in range(n_cut):
                i_start = _n_cut*phi.shape[0]//n_cut
                i_end = (_n_cut+1)*phi.shape[0]//n_cut
                base_evaluated, _= ih.evaluate_all_base(sffi.base, phi[i_start:i_end], sffi.use_jax)
                base_evaluated = jnp.asarray(base_evaluated, dtype=jnp.float64)
                _F_projected = ih.project_delta_phi(delta_phi[i_start:i_end], base_evaluated, sffi.A_normalisation, sffi.delta_t)
                _B = ih.compute_projection_matrix(
                        base_evaluated, base_evaluated, sffi.A_normalisation, sffi.delta_t
                    )
                _B.block_until_ready()
                _F_projected.block_until_ready()
                B += np.asarray(_B)
                F_projected += np.asarray(_F_projected)
            self.l_B_and_F_projected.append((B, F_projected))
        if verbose: print("Time to compute the projection matrix and the projected force %s" % (time.time() - t))
        super().__init__(sffi=sffi, verbose=verbose, start=start, hard_stop=hard_stop, fast_exploration=fast_exploration, 
                    use_STLSQ=use_STLSQ, l_seed_model=l_seed_model)  # Explicitly passing args to the base class

    
    def compute_information(self, nodes_tuple : tuple) -> float:
        if nodes_tuple not in self.d_information.keys():
            nodes = list(nodes_tuple)
            F_F = 0.0
            coeff = 0
            for _k in range(self.k_validation):
                B_test, F_projected_test = self.l_B_and_F_projected[_k]
                B_train, F_projected_train = np.zeros_like(B_test), np.zeros_like(F_projected_test)
                for _j in range(self.k_validation):
                    if _j != _k:
                        B_train += self.l_B_and_F_projected[_j][0]
                        F_projected_train += self.l_B_and_F_projected[_j][1]
                coeff = ih.invert_matrix(shrink_matrix(B_train, nodes)) @ F_projected_train[nodes]
                F_F += 2*np.dot(coeff, F_projected_test[nodes]) - np.dot(coeff, shrink_matrix(B_test, nodes)@coeff)
            #F_F /= self.k_validation
            Information = F_F
            self.d_information[nodes_tuple] = Information
            self.d_helper[nodes_tuple] = F_F, coeff, 0
            
            #if self.verbose: print("Information  %s : %s --> penalty %s"%(nodes_tuple, Information, 2 * np.sum(B_inv*B_D)))
        return self.d_information[nodes_tuple]

            
def split_array_into_equal_parts(array, num_parts=7):
    """
    Split a NumPy array into a specified number of equal-sized parts.
    
    Parameters:
    - array: np.ndarray, the array to be split.
    - num_parts: int, the number of equal-sized parts to split the array into.
    
    Returns:
    - A list of np.ndarray, each being an equal-sized part of the original array.
    """
    # Calculate the size of each part
    part_size = len(array) // num_parts
    
    # Initialize a list to hold the parts
    parts = []
    
    for i in range(num_parts):
        # Calculate the start and end indices for each part
        start_index = i * part_size
        end_index = start_index + part_size
        
        # Append the part to the list
        parts.append(array[start_index:end_index])
    
    return parts