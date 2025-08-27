from .util_inference import inference_help as inf
import numpy as np
import numpy.typing as npt
from .util_inference.polynome import PolynomeInfo
from .util_inference.fourier import Fourier
from jax import numpy as jnp
from jax import jit
import functools
        
class SFFI:
    def __init__(self,
                base : list | PolynomeInfo | Fourier,
                phi : np.ndarray,
                delta_t : float,
                diffusion : np.ndarray|str|float = "Constant", 
                use_jax : bool = True,
                n_futur : int = 0,
                convention : str = "Ito",
                verbose : bool = False,
                direct_force : np.ndarray | None = None,
                clean : bool = True,
                A_normalisation : bool | None = True,
                limit_memory = 2 #GB,
                ) -> None:
        """
        key arguments:
        diffusion : "Multiplicative" "Vestergaard" "time_correction"
        convention : "Strato" "Ito" "trapeze" "noise" "large_dt" "shift"
        """
        self.base = base
        if len(phi.shape) == 1:
            self.phi = jnp.array(phi[:, np.newaxis], dtype=jnp.float64)
        else:
            self.phi  = jnp.array(phi, dtype=jnp.float64)
        #self.phi = self.phi.astype(jnp.float64)
        self.delta_t : float = jnp.float64(delta_t)
        self.use_jax : bool = use_jax
        self.n_futur : int = n_futur
        self.time_correction : bool = False
        if "trapeze" in convention:
            self.time_correction = True
            
        self.D = diffusion
        self.verbose = verbose
        self.direct_force = direct_force
        self.convention = convention
        self.tau = self.delta_t * self.phi.shape[0]  

        if "noise" in self.convention and "Strato" not in self.convention or "shift" in self.convention:
            self.n_futur = 1
            
        A_norm_estimated = self.compute_A()
        
        if A_normalisation == True:
                self.A_normalisation = A_norm_estimated
                #assert type(self.D) == float or type(self.D) == int, "Diffusion should be a float or an int for a single equation !"
        elif A_normalisation == False or A_normalisation is None:
            self.A_normalisation = None
        else:
            self.A_normalisation = A_normalisation

        #self.D_compute = self.try_to_diag_D(self.D_compute)
        
        n_base = len(self.base) if type(self.base) is list else self.base.n_base # type: ignore
        self.n_base = n_base
        i_parallel = 3
        memory_size_GB = self.phi.nbytes*self.n_base*2*i_parallel / 10**9 
        n_cut = int(memory_size_GB // limit_memory + 1)
        if "large_dt" in self.convention or "noise" in self.convention or "Strato" in self.convention:
            memory_size_GB += 2*self.phi.shape[-1]*self.phi.nbytes*n_base / 10**9 
        n_cut = int(memory_size_GB // limit_memory + 1)
        #self.B_D = np.zeros((n_base, n_base)),
        
        D_compute = jnp.zeros((n_base, n_base),)
        self.B = jnp.zeros((n_base, n_base),)
        self.B_large_dt = jnp.zeros((n_base, n_base),)
        self.F_projected, self.F_projected_strato = jnp.zeros(n_base,), jnp.zeros(n_base,)
        for i in range(n_cut):
            start = i*self.phi.shape[0]//n_cut
            end = (i+1)*self.phi.shape[0]//n_cut
            if i == n_cut - 1:
                end = self.phi.shape[0]
            _phi = self.phi[start: end]
            D_compute = self.compute_D(_phi)
            if "large_dt" in self.convention or "Strato" in self.convention or self.n_futur != 0:
                d_base_dx = inf.evaluate_deriv_all_base(self.base, _phi, order_deriv=1)
            else:
                d_base_dx = None
            base_evaluated = self.evaluate_base(_phi)
            #self.B_D += self.compute_B_D(base_evaluated, D_compute)
            base_projection, F_projected = self.compute_F_projected(_phi, base_evaluated, D_compute, d_base_dx)
            self.F_projected += F_projected
            B = self.compute_projection_matrix(base_evaluated, base_projection)
            self.B += B
            if (i+1)%i_parallel == 0:
                self.B.block_until_ready()
            
            if "large_dt" in self.convention:
                B_large_dt, F_projected_strato = self.compute_B_large_dt(_phi, d_base_dx, base_evaluated, D_compute)
                self.B_large_dt += B_large_dt
                self.F_projected_strato += F_projected_strato

            del base_evaluated, base_projection
           
        self.F_projected = np.asarray(self.F_projected)
        self.B = np.asarray(self.B)
        self.B_large_dt = np.asarray(self.B_large_dt) 
        self.F_projected_strato = np.asarray(self.F_projected_strato)
        
        self.infer_coefficient()
        
        if clean:
            ## Free memory
            base_evaluated = None
            self.D_compute = np.array([])
            D_compute = np.array([])
        else:
            self.D_compute = D_compute
        
    def evaluate_base(self, phi):
        base_evaluated, self.l_name_base = inf.evaluate_all_base(self.base, phi, self.use_jax)
        self.non_zero_dim = None
        self.simple_base_name = None
        if (type(self.base) is PolynomeInfo  or type(self.base) is Fourier) and len(self.base.non_zero_dim) > 0:
            self.non_zero_dim = self.base.non_zero_dim
            self.simple_base_name = [x.split(":")[1] for x in self.l_name_base]
        return base_evaluated
           
    def compute_F_projected(self, phi, base_evaluated, D_compute, d_base_dx): 
        delta_phi = (phi[1:] - phi[:-1])/self.delta_t
        if self.n_futur != 0:
           delta_phi =  delta_phi[self.n_futur:] 
        
        if "Strato" in self.convention:
            base_projection = (base_evaluated[:, :-1] + base_evaluated[:, 1:])/2
        else:
            base_projection = base_evaluated
    
        base_projection = base_projection[:,:-1]
        F_projected = inf.project_delta_phi(delta_phi, base_projection, self.A_normalisation, self.delta_t)
        if "Strato" in self.convention and self.n_futur == 0:
            correction_strato = inf.project_delta_phi(D_compute, d_base_dx, self.A_normalisation, self.delta_t) 
            F_projected -= correction_strato
            #F_projected_info -= correction_strato

        return base_projection, F_projected
    
    def compute_projection_matrix(self, base_evaluated, base_projection):
        if self.time_correction:
            real_base = (base_evaluated[:, self.n_futur:-1] + base_evaluated[:, 1 + self.n_futur:])/2
        elif self.n_futur != 0:
            real_base = base_evaluated[:, self.n_futur:]
        else:
            real_base = base_evaluated
        
        if "Strato_B_asymetrique" in self.convention:
            base_projection = base_evaluated
        
        t_max = min(base_projection.shape[1], real_base.shape[1])
        if t_max != base_projection.shape[1] or t_max != real_base.shape[1]:
            base_projection = (base_projection[:,:t_max])
            real_base = (real_base[:,:t_max])
        B = inf.compute_projection_matrix(
                 base_projection, real_base, self.A_normalisation, self.delta_t
                )
        return B

    def infer_coefficient(self):
        self.B_inv = inf.invert_matrix(self.B)
        self.coefficients = self.B_inv @ self.F_projected 
        #self.F = jnp.einsum("i, i...", self.coefficients, self.base_evaluated)
        #self.FF = np.mean(self.F**2)

    def compute_D(self, phi):
        if type(self.D) is str:
            D = compute_D_jax(phi, self.D, self.delta_t)
        else:
            D = self.D * jnp.tile(jnp.ones(self.n_differential_equation), (delta_phi.shape[0]-1, 1, 1)) #type: ignore
        return D
    
    def compute_B_large_dt(self, phi, d_base_dx, base_evaluated, D_compute):
        delta_phi = (phi[1:] - phi[:-1])/self.delta_t
        base_strato = (base_evaluated[:, :-1] + base_evaluated[:, 1:])/2
        B_strato = inf.compute_projection_matrix(
            base_strato, base_strato, self.A_normalisation, self.delta_t
            )
        F_projected_strato = inf.project_delta_phi(delta_phi, base_strato, self.A_normalisation, self.delta_t)
        correction_strato = inf.project_delta_phi(D_compute, d_base_dx, self.A_normalisation, self.delta_t)
        F_projected_strato -= correction_strato
        return B_strato, F_projected_strato
        
    def compute_A(self):
        A = 1
        var_noise = 0
        phi = np.asarray(self.phi, dtype=np.float64)
        delta_phi = (phi[1:] - phi[:-1])/self.delta_t
        delta_phi = delta_phi.reshape(delta_phi.shape[0], -1)
        delta_phi = np.asarray(delta_phi, dtype=np.float64)
        if delta_phi.shape[-1] > 100:
            D = self.delta_t/2 * np.mean(delta_phi**2)
            A = float(1/(4*D))
        else:
            if type(self.D) is str:
                D = self.delta_t/(2) * np.einsum("...ti,...tj->...ij", delta_phi, delta_phi)/(delta_phi.shape[0])
                if "time_correction" in self.D:
                    # delta_phi_1 = (self.phi[1:] - self.phi[:-1])[:-1]
                    # delta_phi_2 = (self.phi[2:] - self.phi[:-2])
                    # D = (4*jnp.einsum("...ti,...tj->...ij", delta_phi_1, delta_phi_1) - jnp.einsum("...ti,...tj->...ij", delta_phi_2, delta_phi_2))/(4*self.delta_t*delta_phi_1.shape[0])
                    delta_phi_correct = (phi[2:] + phi[:-2] - 2*phi[1:-1])
                    if not "Multiplicative" in self.D:
                        D = (np.einsum("...ti,...tj->...ij", delta_phi_correct, delta_phi_correct))/(4*self.delta_t*delta_phi_correct.shape[0])
                    
                if "Vestergaard" in self.D:
                    var_noise = -self.delta_t**2 * np.einsum("...ti,...tj->...ij", delta_phi[:-1], delta_phi[1:])/(delta_phi.shape[0]-1)
                    var_noise = (var_noise + var_noise.T)/2  
                    D -= var_noise / self.delta_t
      
                A = inf.invert_matrix(4*D)
                A = (A + A.T)/2 # Ensure that A is symmetric
            else:
                if not callable(self.D):
                    if type(self.D) is not np.ndarray and type(self.D) is not str:
                        if self.D == 0:
                            A = 1
                        else:
                            A = np.diag(np.ones(self.phi.shape[1])) * (4*float(self.D))**(-1) #* np.ones_like(self.phi)
                    else:
                        D_inv = inf.invert_matrix(4*np.array(self.D))
                        A = D_inv # * np.ones((*self.phi.shape, self.n_differential_equation))
                
                else:
                    D_compute = np.array([self.D(self.phi[i]) for i in range(self.phi.shape[0])])
                    # A = np.linalg.pinv(self.D_compute)
                    D_inv_average = inf.invert_matrix(4*np.mean(D_compute, axis=0))
                    A = D_inv_average * np.ones((*self.phi.shape, self.phi.shape[1]))
                
        return jnp.asarray(A, dtype=jnp.float64)
    
    def compute_B_D(self, base_evaluated, D_compute):
        if self.A_normalisation is None:
            two_DA = 2*D_compute
        elif len(self.A_normalisation.shape) == 2: # type: ignore
            two_DA = 2*D_compute@self.A_normalisation
        else:
            raise Exception("Ill defined A_normalisation")
        if np.all(two_DA[-1] == two_DA[0]): #np.all(np.diag(np.diag(two_DA[-1])) == two_DA[-1]) and 
            A_normalisation = two_DA[0]
            if self.A_normalisation is not None:
                A_normalisation = self.A_normalisation@two_DA[0]
            B_D = inf.compute_projection_matrix(
                base_evaluated, base_evaluated, A_normalisation, self.delta_t)
        else:
            _min = min(two_DA.shape[0], base_evaluated.shape[1])
            left_term = jnp.einsum("tij, btj -> bti", two_DA[:_min], base_evaluated[:, :_min])
            B_D = inf.compute_projection_matrix(
                base_evaluated, left_term,  self.A_normalisation, self.delta_t) 
        return B_D
    
            
def real_error(F_estimated, real_F, non_noisy_phi : np.ndarray|None = None,
               A = None, sffi : SFFI|None = None):
    if non_noisy_phi is not None:
        base_evaluated, _ = inf.evaluate_all_base(sffi.base, non_noisy_phi, sffi.use_jax)# type: ignore
        assert len(base_evaluated) == len(sffi.coefficients) # type: ignore
        F_estimated = np.einsum("i, i...", sffi.coefficients, base_evaluated) # type: ignore

    _min = min(real_F.shape[0], F_estimated.shape[0])
    diff_F = (real_F[:_min] - F_estimated[:_min])
    if A is not None and type(A) is np.ndarray:
        _norm = np.mean((real_F@A)*real_F) #max(np.mean(real_F**2), np.mean(F_estimated**2))
        return float(np.mean((diff_F@A)*diff_F) / _norm)
    else:
        _norm = np.mean(real_F**2) #max(np.mean(real_F**2), np.mean(F_estimated**2))
        return float(np.mean( (diff_F**2)) / _norm)
        

def compute_D_jax(phi, D_type, delta_t):
    var_noise = 0
    delta_phi = (phi[1:] - phi[:-1])/delta_t
    delta_phi = jnp.array(delta_phi.reshape(delta_phi.shape[0], -1))
    
    if delta_phi.shape[-1] > 100:
        D = delta_t/(2) *jnp.mean(delta_phi**2, axis=0) # For SPDE
    else:
        if "Vestergaard" in D_type:
            if "Multiplicative" in D_type:
                var_noise = -delta_t**2 * (jnp.einsum("...ti,...tj->...tij", delta_phi[:-1], delta_phi[1:]) + jnp.einsum("...ti,...tj->...tij", delta_phi[1:], delta_phi[:-1]))/2
            else:
                var_noise = -delta_t**2 * jnp.einsum("...ti,...tj->...ij", delta_phi[:-1], delta_phi[1:])/(delta_phi.shape[0]-1)
                var_noise = (var_noise + var_noise.T)/2
        if "Multiplicative" in D_type:
            D = delta_t/(2) * jnp.einsum("...i,...j->...ij", delta_phi, delta_phi)[:-1]
        else:
            D = delta_t/(2) * jnp.einsum("...ti,...tj->...ij", delta_phi, delta_phi)/(delta_phi.shape[0])
            #D = D * jnp.tile(jnp.ones(delta_phi.shape[1]), (delta_phi.shape[0]-1, 1, 1))
        D -= var_noise / delta_t
        
        if "time_correction" in D_type:
            delta_phi_correct = (phi[2:] + phi[:-2] - 2*phi[1:-1])
            if not "Multiplicative" in D_type:
                D = (jnp.einsum("...ti,...tj->...ij", delta_phi_correct, delta_phi_correct))/(4*delta_t*delta_phi_correct.shape[0])
                D = D * jnp.tile(jnp.ones(delta_phi.shape[1]), (delta_phi.shape[0]-1, 1, 1))
            else:
                D = (jnp.einsum("...i,...j->...ij", delta_phi_correct, delta_phi_correct))/(4*delta_t)
                D = jnp.insert(D, 0, D[0], axis=0)
            D -= 6/4 * var_noise / delta_t
    return D