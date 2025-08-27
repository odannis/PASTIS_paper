
from abc import ABC, abstractmethod 
from typing import Callable
import numpy as np
import SFFI.simulation as simu
from typing import List
from SFFI.sffi import PolynomeInfo
import pysindy as ps
import jax

class Model(ABC):
    shape_image : list
    first_image : np.ndarray
    force_numba : Callable
    dt : float 
    n_differential_equation : int
    total_base : PolynomeInfo|List[Callable]
    index_real_base : list
    feature_library : ps.PolynomialLibrary
    index_real_base_sindy : list
    index_real_base_sindy_weak_form : list
    n : int
    diffusion_strength : float
    drift_term : Callable
    sqrt_diffusion : Callable|float|int
    rng : np.random.Generator = np.random.default_rng()
    over_sampling : int = 10
    fast_random : bool|np.random.Generator = False
    init_params : dict = {}
    
    def __init__(self, dt : float = 0.0002, n : int = 100_000, over_sampling : int = 10,
                 fast_random : bool = False, diffusion_strength : float = 1, thermalised_time = 10.0, **kwargs):
        self.dt = dt
        self.n = n
        self.over_sampling = over_sampling
        self.fast_random = fast_random
        self.diffusion_strength = diffusion_strength
        self.thermalised_time = thermalised_time
        if "kwargs" in kwargs:
            for key, value in kwargs["kwargs"].items():
                self.__setattr__(key, value)
                
    def get_parameter_simulation(self) -> dict:
        key = jax.random.key(self.rng.integers(0, 2**32 - 1))
        parameter_simulation =  {
            "shape_image" : self.shape_image,  "first_image" : self.first_image,
            "base" : [self.drift_term], "coefficient" : [1], "over_sampling" : self.over_sampling,
             "dt" : self.dt, "n" : self.n, "sqrt_diffusion": self.sqrt_diffusion, "key" : key,
            "diffusion_strength" : self.diffusion_strength, "thermalised_time" : self.thermalised_time,
            }
        return parameter_simulation
            
    def simu_from_EstimationError(self, d_estimation_error : dict):
        parameter_simulation = self.get_parameter_simulation()
        parameter_simulation["diffusion_strength"] = float(d_estimation_error["D_strength"])
        parameter_simulation["n"] = int(d_estimation_error["time"] / d_estimation_error["dt"])
        parameter_simulation["dt"] = d_estimation_error["dt"]
        x_train, dt = simu.simulate(**parameter_simulation)

        if float(d_estimation_error["experimental_noise"]) != 0:
            x_train += np.random.normal(scale=d_estimation_error["experimental_noise"], size=x_train.shape)
        return x_train, dt