import numpy as np
from .sffi import SFFI
import matplotlib.pyplot as plt
from .util_inference import inference_help as ih
from tqdm import tqdm
from . import prob_model as pm
from typing import Literal
from scipy.special import erfinv
from abc import ABC, abstractmethod

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

class Exploration(ABC): # https://arxiv.org/pdf/1406.4802.pdf
    def __init__(self, sffi : SFFI, verbose : bool = False, start = True, 
                hard_stop = True, fast_exploration = False, use_STLSQ : bool = False,
                l_seed_model : list[tuple[int]] = []):
        self.verbose = verbose
        self._sffi = sffi
        self.use_STLSQ = use_STLSQ
        self.hard_stop = hard_stop
        self.fast_exploration = fast_exploration
        self.l_seed_model = l_seed_model
        
        self.d_information = {}
        self.d_helper = {}
        self.d_size = {}
        self.l_nodes_selected : list[tuple] =  [tuple(range(sffi.coefficients.shape[0]))]#
        self.compute_information(tuple(range(sffi.coefficients.shape[0])))
        self.best_nodes = tuple()
        self.base_max_score = self.l_nodes_selected[0]
        if start:
            self.initialize_set()
            self.compute_information(self.base_max_score)
            if not self.use_STLSQ:
                self.exploration_null_and_full()  
            else:
                self.STLSQ()
                
            self.best_nodes = self.global_best_nodes()
            
        if verbose:
            self.print_best_nodes()

    @abstractmethod
    def compute_information(self, nodes_tuple : tuple) -> float:
        raise Exception("compute_information not implemented")
    
    def exploration_null_and_full(self): 
        n_coef = self._sffi.coefficients.shape[0]
        n_copy = n_coef
        if not self.fast_exploration:
            #self.l_nodes_selected : list[tuple]  = [tuple(sorted(set(np.random.choice(n_coef, size=np.random.randint(0, n_coef), replace=False)))) for i in range(n_copy)]#
            self.l_nodes_selected : list[tuple]  = [tuple(sorted(set(np.random.choice(n_coef, size=i, replace=False)))) for i in range(n_coef)]#
            self.l_nodes_selected.append(tuple())
            self.l_nodes_selected.append(tuple(range(n_coef)))
            self.l_nodes_selected.extend(self.l_seed_model)

            model_selected = self.explore_nodes(allow_substitution=False)
            self.l_nodes_selected = [model_selected]     
        else:
            self.l_nodes_selected : list[tuple]  = [tuple()]
            self.l_nodes_selected.extend(self.l_seed_model)
            model_selected = self.explore_nodes(allow_substitution=False)
            self.l_nodes_selected = [model_selected] 
        
    def explore_nodes(self, allow_substitution : bool = True):    
        self.find_new_set(allow_substitution=allow_substitution)
        model_selected = self.get_best_nodes()
        return model_selected
        
    def global_best_nodes(self):
        self.best_nodes = self.get_best_nodes()
        return self.best_nodes
        
    def print_best_nodes(self):
        sffi = self._sffi
        try:
            l_name = np.array(sffi.l_name_base)
            d_name = {}
            coefficients = np.around(self.d_helper[self.best_nodes][1], 2)
            for i, name in enumerate(l_name[list(self.best_nodes)]):
                dim, func = name.split(":")
                d_name[dim] = d_name.get(dim, "") + str(coefficients[i]) + func + " +"
            msg = "Best nodes : \n"
            for key in np.sort(list(d_name.keys())):
                msg += "         %s : %s\n"%(key, d_name[key])
            print(msg)
        except Exception as e:
            print(e)
            
    def initialize_set(self):
        for nodes in self.l_nodes_selected:
            info = self.compute_information(nodes)
        if self.verbose : print("Initial set : ", self.l_nodes_selected)

    def find_new_set(self, allow_substitution=True):
        sffi = self._sffi
        l_previous_previous_nodes = []
        best_nodes_here = ()
        c = 0
        while self.l_nodes_selected != l_previous_previous_nodes:
            if c > sffi.coefficients.shape[0] * 1000:
                print("Warning : infinite loop break it !")
                print("Previous nodes : ", l_previous_previous_nodes)
                print("Actual nodes : ", self.l_nodes_selected)
                break
            c += 1
            l_previous_nodes = self.l_nodes_selected.copy()
            best_nodes_here = self.new_branches(substitution=allow_substitution, hard_stop=self.hard_stop)
            if self.l_nodes_selected == l_previous_nodes:
                l_previous_previous_nodes = l_previous_nodes.copy()
                # if allow_substitution:
                #     if self.verbose : print("Look for substitution")
                #     best_nodes_here = self.new_branches(substitution=True, hard_stop=self.hard_stop)
        return best_nodes_here
    
    def new_branches(self, substitution=False, hard_stop = False):
        sffi = self._sffi
        previous_l_nodes_selected = self.l_nodes_selected.copy()
        best_nodes = max(self.d_information.keys(), key=self.d_information.get) # type: ignore #previous_l_nodes_selected[0]
        l_nodes_selected_next = []
        for inital_tuple in previous_l_nodes_selected:
            self.compute_information(inital_tuple)
            l_new_nodes = [inital_tuple]
            suffle_coeff = np.random.permutation(sffi.coefficients.shape[0])
            for i in suffle_coeff:
                c = 0
                if i not in inital_tuple:
                    tuple_try = tuple((inital_tuple + (i, )))
                else:
                    set_try = set(inital_tuple)
                    set_try.remove(i)
                    tuple_try = tuple(set_try,)
                    
                tuple_try = tuple(sorted(tuple_try)) #order tuple
                self.compute_information(tuple_try)
                l_new_nodes.append(tuple_try)
                if hard_stop and self.difference_information(tuple_try, inital_tuple) > 0:
                    #if self.verbose : print("hard stop")
                    break
                if substitution:
                    best_found = False
                    initial_list = list(inital_tuple)
                    np.random.shuffle(initial_list)
                    for k in initial_list:
                        c += 1
                        set_try = set(inital_tuple)
                        set_try.remove(k)
                        if i in set_try:
                            set_try.remove(i)
                        else:
                            set_try.add(i)
                        tuple_try = tuple(sorted(set_try))
                        self.compute_information(tuple_try)
                        l_new_nodes.append(tuple_try)
                        if self.difference_information(tuple_try, best_nodes) > 0:
                            best_found = True
                            break
                        if c > 1000:
                            print("too much iterations")
                            break
                    if best_found:
                        break

            # for node in l_nodes_selected_next:
            #     if node in l_new_nodes:
            #         l_new_nodes.remove(node)
            
            if len(l_new_nodes)>0: 
                l_nodes_selected_next.append(self.get_best_nodes(new_nodes=l_new_nodes))
            
        best_nodes = max(l_nodes_selected_next, key=self.d_information.get) # type:ignore
        if self.verbose : print("Actual best nodes info=%s : %s  "%(self.compute_information(best_nodes), best_nodes))
        self.l_nodes_selected = l_nodes_selected_next
        return best_nodes
    
    def systematic_exploration(self, max_length=None, model_base=None):
        import itertools
        sffi = self._sffi
        if max_length is None:
            max_length = sffi.coefficients.shape[0]
        max_length = min(max_length, sffi.coefficients.shape[0])
        for i in tqdm(range(1, max_length+1), desc="Dumb mode"):
            for comb in itertools.combinations(list(range(sffi.coefficients.shape[0])), i):
                if model_base is not None:
                    if not set(model_base).issubset(set(comb)):
                        continue
                self.compute_information(tuple(sorted(comb)))
        self.best_nodes = self.get_best_nodes()
                
    def difference_information(self, nodes_1 : tuple, nodes_2 : tuple):
        """
        I(nodes_1) - I(nodes_2) +- selectivity * sd(nodes_1, nodes_2)
        """ 
        I_1 = self.compute_information(nodes_1)
        I_2 = self.compute_information(nodes_2)
        return I_1 - I_2
        
    def compute_force_set(self, nodes : tuple|list, real_phi : np.ndarray|None = None, base_evaluated : np.ndarray|None = None):
        sffi = self._sffi
        if real_phi is not None:
            phi = real_phi
        else:
            phi = sffi.phi
        nodes = list(nodes)
        F = np.zeros_like(phi)
        F_projected = np.asarray(sffi.F_projected)
        B = shrink_matrix(sffi.B, nodes)
        coeff = np.asarray(ih.invert_matrix(B) @ F_projected[nodes])
        n_base = len(sffi.base) if type(sffi.base) is list else sffi.base.n_base # type: ignore
        memory_size_GB = F.nbytes * n_base / 10**9 
        limit_memory = 1 #GB
        if base_evaluated is None:
            n_cut = int(memory_size_GB // limit_memory + 1)
            for i in range(n_cut):
                i_start = i*phi.shape[0]//n_cut
                i_end = (i+1)*phi.shape[0]//n_cut
                base_evaluated, _ = ih.evaluate_all_base(sffi.base, phi[i_start:i_end], sffi.use_jax)
                base_evaluated = np.asarray(base_evaluated)
                for i_coeff in range(len(nodes)):
                    F[i_start:i_end] += coeff[i_coeff] * base_evaluated[nodes[i_coeff]]
        else:
            for i_coeff in range(len(nodes)):
                F += coeff[i_coeff] * base_evaluated[nodes[i_coeff]]
        F = np.asarray(F)
        return F

    def get_best_nodes(self, new_nodes=None) -> tuple:
        generator = self.d_information.keys()
        if new_nodes is not None:
            generator = new_nodes
        best_nodes = max(generator, key=self.d_information.get) #type: ignore
        return best_nodes
    
    def possible_best_nodes(self, use_name_base=True, print_msg = True):
        sffi = self._sffi
        max_info_nodes = max(self.d_information, key=self.d_information.get) # type: ignore
        info_max = self.d_information[max_info_nodes]
        l_possible_best_nodes = []
        for key in self.d_information.keys():
            if self.difference_information(key, max_info_nodes) >= 0:
                l_possible_best_nodes.append(key)
        msg = "Possible best nodes : \n"
        name_base = np.array(sffi.l_name_base)
        for nodes in l_possible_best_nodes:
            if use_name_base:
                try:
                    msg += "         %s : %s\n"%(name_base[list(nodes)], self.d_information[nodes])
                except:
                    msg += "         %s : %s\n"%(nodes, self.d_information[nodes])
            else:
                msg += "         %s : %s\n"%(nodes, self.d_information[nodes])
        if print_msg:
            print(msg)
        return []#l_possible_best_nodes
                    
    def get_list_best_nodes(self) -> dict:
        best_nodes = self.get_best_nodes()
        best_info = self.d_information[best_nodes]
        d_set = {}
        d_set[tuple(sorted(best_nodes))] = best_info
        for key in self.d_information.keys():
            info = self.d_information[key]
            if info > 0:
                if info > best_info:
                    d_set[key] = info
        return d_set
    
    def print_top_best_nodes(self, use_name_base=True):
        sffi = self._sffi
        nodes = list(self.d_information.keys())
        infos_sd = list(self.d_information.values())
        infos = [i for i in infos_sd]
        arg = np.argsort(infos)
        l_info_x, l_info_shown, l_shown, l_sd = [], [], [], []
        print(" Top set %s"%str(self.best_nodes))
        max_info_nodes = max(self.d_information, key=self.d_information.get) # type: ignore
        info_max = self.d_information[max_info_nodes]
        name_base = np.array(sffi.l_name_base)
        for i in range(len(arg)):
            index = -i -1 
            if len(nodes[arg[index]]) not in l_shown:
                l_shown.append(len(nodes[arg[index]]))
                l_info_x.append(str(nodes[arg[index]]))
                l_info_shown.append(infos[arg[index]])
                nodes_tuple = tuple((nodes[arg[index]]))
                self.compute_information(nodes_tuple)
                F_F, coeff, penalty = self.d_helper[nodes_tuple]
                start = "    "
                if tuple(nodes[arg[index]]) == self.best_nodes:
                   start = "Best"
                name_nodes = nodes_tuple
                if use_name_base:
                    name_nodes = name_base[list(nodes_tuple)]
                #print(start + " Set %s : dif info = %s penalty %s, sd with best %s"%(nodes_tuple, infos[arg[index]] - self.d_information[self.best_nodes], penalty, sd_1))
                print(start + " Set %s : info = %s penalty %s"%(name_nodes, infos[arg[index]], penalty))
        # plt.figure()
        # #plt.plot(l_info_x, l_info_shown, marker=".")
        # plt.errorbar(l_info_x, l_info_shown, yerr=l_sd, marker=".", linestyle="-")
        # plt.hlines(self.d_information[self.best_nodes], 0, len(l_info_x), linestyle="--")
        # plt.yscale("log")
        # plt.show()

    def STLSQ(self):
        sffi = self._sffi
        nodes = list(range(sffi.coefficients.shape[0]))
        for i in range(len(nodes)):
            nodes = sorted(nodes)
            self.compute_information(tuple(nodes))
            coeff = self.d_helper[tuple(nodes)][1]
            delete_node = np.argmin(np.abs(coeff))
            nodes.pop(delete_node)
            
    def get_coefficients_model(self, nodes : tuple) -> np.ndarray:
        self.compute_information(nodes)
        return self.d_helper[nodes][1]