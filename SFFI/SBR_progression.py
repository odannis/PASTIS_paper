import numpy as np
from .sffi import SFFI
import matplotlib.pyplot as plt
from .util_inference import inference_help as ih
from tqdm import tqdm
from . import prob_model as pm
from typing import Literal
from scipy.special import erfinv
from ._exploration_models import Exploration

def shrink_matrix(M, nodes : list[int], nodes_2 : list[int]|None = None):
    M = M[nodes, :]
    if nodes_2 is None:
        M = M[:, nodes]
    else:
        M = M[:, nodes_2]
    return M

def timeit(method):
    import time
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f s' % (method.__name__, (te - ts)))
        return result
    return timed

class L0_SBR(Exploration): # https://arxiv.org/pdf/1406.4802.pdf
    def __init__(self, sffi : SFFI, verbose : bool = False, start = True,  
                hard_stop = True, fast_exploration = False, use_STLSQ : bool = False,
                use_BIC = False, use_AIC = False, p = 0.001, special_convention = None):
        self.verbose = verbose
        self._sffi = sffi
        self.D = sffi.D
        self.use_BIC = use_BIC
        self.use_AIC = use_AIC
        self.special_convention = special_convention
        if self.use_BIC and self.use_AIC:
            raise ValueError("You can't use both AIC and BIC")
        self.p = p
        super().__init__(sffi=sffi, verbose=verbose, start=start, hard_stop=hard_stop, fast_exploration=fast_exploration, 
                         use_STLSQ=use_STLSQ)  # Explicitly passing args to the base class

    def compute_information(self, nodes_tuple : tuple) -> float:
        sffi = self._sffi
        if nodes_tuple not in self.d_information.keys():
            nodes = list(nodes_tuple)
            B = shrink_matrix(sffi.B, nodes) ## Or select B for n_futur ?
            if "large_dt" in sffi.convention:
                B_info = shrink_matrix(sffi.B_large_dt, nodes)
            else:
                B_info = B
            
            try:
                coeff = np.linalg.lstsq(B, sffi.F_projected[nodes], rcond=-1)[0]
            except Exception as e:
                print(e)
                B_inv = ih.invert_matrix(B)
                coeff = B_inv @ sffi.F_projected[nodes]
                
            F_projected = sffi.F_projected[nodes]
            F_F = 2*np.dot(coeff, F_projected) - np.dot(coeff, B_info@coeff)

            if "Strato" in sffi.convention:
                B_info = shrink_matrix(sffi.B, nodes)
                F_projected_strato = sffi.F_projected[nodes] #+ sffi.correction_strato[nodes]
                F_F = 2*np.dot(coeff, F_projected_strato) - np.dot(coeff, B_info@coeff) # type: ignore
                pass
            elif "large_dt" in sffi.convention:
                F_projected_strato = sffi.F_projected_strato[nodes] #+ sffi.correction_strato[nodes]
                F_F = 2*np.dot(coeff, F_projected_strato)  - np.dot(coeff, B_info@coeff) # type: ignore
                pass
            if self.use_BIC == True:
                penalty = 1/2 * len(nodes)*np.log(sffi.delta_t*sffi.phi.shape[0])
            elif self.use_AIC == True:
                penalty = len(nodes)
            elif self.special_convention == "PASTIS_exact":
                n_0 = sffi.coefficients.shape[0]
                n_eff = n_0 - len(nodes) + 1
                try:
                    penalty = len(nodes)*erfinv((1-self.p)**(1/(n_eff + 1)))**2
                except Exception as e:
                    print("PASTIS_exact", e)
                    penalty = len(nodes)*np.log(self._sffi.n_base/self.p)
            elif self.special_convention == "Ando":
                penalty_1 = len(nodes)*(np.log(self._sffi.n_base/self.p))
                #F_F = 2*np.dot(sffi.coefficients, sffi.F_projected) - np.dot(sffi.coefficients, sffi.B@sffi.coefficients)
                penalty_2 = len(nodes)*self.p*F_F
                penalty = min(penalty_1, penalty_2)
            elif self.special_convention == "Pierre_idea":
                F_F_here = 2*np.dot(sffi.coefficients, sffi.F_projected) - np.dot(sffi.coefficients, sffi.B@sffi.coefficients)
                penalty = len(nodes)*(np.log(F_F_here))

            elif self.special_convention == "no_penalty":
                penalty = 0
            else:
                penalty = len(nodes)*(np.log(self._sffi.n_base/self.p))
                
            #print(penalty)
            Information = F_F - penalty
            self.d_information[nodes_tuple] = Information
            self.d_helper[nodes_tuple] = F_F, coeff, penalty
            
        return self.d_information[nodes_tuple]
