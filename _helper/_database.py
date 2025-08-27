try:
    import SFFI.simulation as simu
    import SFFI.sffi as inf
    import SFFI.SBR_progression as S
    import SFFI.Cross_validation as CV
except ImportError:
    import sys
    import os
    dir2 = os.path.abspath('')
    dir1 = os.path.dirname(dir2)
    if not dir1 in sys.path: sys.path.append(dir1)
    if not dir2 in sys.path: sys.path.append(dir2)
    import SFFI.simulation as simu
    import SFFI.sffi as inf
    import SFFI.SBR_progression as S
    import SFFI.Cross_validation as CV
    
import time
import sys
import numpy as np
from tqdm import tqdm
import pysindy as ps
import pandas as pd
from scipy.special import comb
from sklearn.linear_model import Lasso, LassoCV
import jax
try:
    from ._class import  EstimationError, InferenceParameter
except ImportError:
    from _class import  EstimationError, InferenceParameter
import os
from typing import Any
from simulation_models._common_class import Model
import gc
from itertools import combinations

def loop_over_parameter_simulation(parameter_inference : InferenceParameter, l_parameter, name_parameter : str, name_csv,
                                   replicate=1, number_simu=""):
    l_tot = []
    parameter_inference = parameter_inference.copy()
    gen = np.random.default_rng()
    parameter_inference.experimental_noise = 0
    parameter_inference.loop_on_p_PASTIS = False
    try:
        if int(number_simu)%2 == 0:
            l_parameter = np.flip(l_parameter)
    except:
        pass
    for i in range(replicate):
        para_simu = parameter_inference.model.get_parameter_simulation()
        if name_parameter == "dt":
            para_simu[name_parameter] = np.min(l_parameter)
        elif name_parameter == "n":
            #para_simu[name_parameter] = np.max(l_parameter)
            if parameter_inference.use_loop_on_p_PASTIS:
                parameter_inference.loop_on_p_PASTIS = True
        #para_simu["first_image"] += gen.normal(loc=0, scale=para_simu["first_image"]/10, size=para_simu["shape_image"]) if para_simu["shape_image"] != [1] else gen.normal(loc=0, scale=para_simu["first_image"]/10)
        x_simu = np.array([]) # Avoid type error
        if name_parameter == "dt"  or name_parameter == "experimental_noise" :#or name_parameter == "n":
            x_simu, dt = simu.simulate(**para_simu)
        for parameter in tqdm(l_parameter, desc=str(number_simu) + " Simulation " + name_parameter + " " + name_csv):
            if name_parameter == "dt":
                over_sampling = int(parameter / np.min(l_parameter))
                x_train = x_simu[::over_sampling]
                para_simu[name_parameter] = over_sampling*np.min(l_parameter)
            # elif name_parameter == "n":
            #     x_train = x_simu[:parameter]
            elif name_parameter == "experimental_noise":
                parameter_inference.experimental_noise = parameter
                x_train = x_simu + gen.normal(0, parameter, x_simu.shape)
            else:
                para_simu = parameter_inference.model.get_parameter_simulation()
                para_simu[name_parameter] = parameter
                x_train, dt = simu.simulate(**para_simu)
            if np.isnan(np.sum(x_train)):
                print("Nan in simulation")
                continue
                
            _para_simu_ = para_simu.copy()
            _para_simu_["n"] = min(parameter_inference.model.get_parameter_simulation()["n"], 10**6)
            # if "GrayScott" in str(parameter_inference.model):
            #     _para_simu_["n"] = min(1000, parameter_inference.model.get_parameter_simulation()["n"])
            _para_simu_["key"] = jax.random.key(gen.integers(0, 2**32 - 1))
            x_predictive_error, dt = simu.simulate(**_para_simu_)
            
            if name_parameter == "experimental_noise":
                if len(x_simu.shape) == 1:
                    x_simu = x_simu[:,np.newaxis]
                    x_train = x_train[:,np.newaxis]
                    x_predictive_error = x_predictive_error[:,np.newaxis]
            else:
                if len(x_train.shape) == 1:
                    x_train = x_train[:,np.newaxis]
                    x_predictive_error = x_predictive_error[:,np.newaxis]

            parameter_inference._populate(x_train, x_predictive_error, para_simu)
            l_tot.extend(update_estimation_error(x_train, parameter_inference))

            if len(l_tot) > 0:
                data = pd.DataFrame(l_tot)
                dir2 = os.path.dirname(os.path.abspath(''))
                name_save = ( dir2 + "/csv/" + name_csv
                    + "_" + name_parameter + ".pkl__" + number_simu)
                try:
                    #data.to_csv(name_csv_save, mode='a', header=not os.path.isfile(name_csv_save))
                    # Save data to a pickle file
                    if os.path.isfile(name_save):
                        # Append data to existing file
                        existing_data = pd.read_pickle(name_save)
                        data = pd.concat([existing_data, data], ignore_index=True)
                    data.to_pickle(name_save)
                    l_tot = []
                    del data
                except OSError as e:
                    print("Error : ", e)
            else:
                print("Warning : no data to save")
    parameter_inference.experimental_noise = 0
    del l_tot
    gc.collect()

def model_sindy(x_train, dt, para : InferenceParameter, optimizer : Any = ps.STLSQ(), verbose=False, **kwargs):
    x_train = np.array(x_train)
    model = ps.SINDy(optimizer=optimizer, **kwargs)
    model.fit(x_train, t=dt)
    if verbose:
        model.print()

    if optimizer.__class__ is ps.optimizers.base.EnsembleOptimizer:
        coeff_selected = np.where(np.abs(model.coefficients().flatten()) >= para.threshold_sindy)[0]# Bug with weak method : coeff are not set to 0
    else:
        coeff_selected = np.where(model.coefficients().flatten() != 0)[0]
    if para.x_no_experimental_noise is not None:
        return coeff_selected, model.predict(para.x_no_experimental_noise), np.array(model.predict(para.x_predictive_error))
    else:
        return coeff_selected, np.array(model.predict(x_train)), np.array(model.predict(para.x_predictive_error))


def update_estimation_error(x_train : np.ndarray, para : InferenceParameter, verbose=False):
    l = []
    dt = para.dt
    if verbose: print("SFFI")
    time_SFFI = time.time()
    inf_1 = inf.SFFI(para.total_base, x_train, dt, convention=para.convention, A_normalisation=para.A_normalisation, n_futur=para.n_future, diffusion=para.diffusion, clean=True)
    time_SFFI = time.time() - time_SFFI
    if verbose: print("SFFI time", time_SFFI)
    model_l0 = S.L0_SBR(inf_1, start=False)
    para.update_error_real_model(model_l0, tuple(para.index_real_base), inf_1)
    
    ## Real nodes ###
    model_l0.best_nodes = tuple(para.index_real_base)
    method = "Real_model" + "_"  + para.convention + "_A_" + str(para.A_normalisation) + "_diffusion_" + str(para.diffusion)
    _estimation_error = EstimationError(para, method=method)
    _estimation_error.update_error_from_model_l0(model_l0, para, inf_1)
    l.append(_estimation_error)
    
    ## Total nodes ###
    model_l0.best_nodes = tuple(list(range(inf_1.coefficients.shape[0])))
    method = "Total_model" + "_" + para.convention  + "_A_" + str(para.A_normalisation) + "_diffusion_" + str(para.diffusion)
    _estimation_error = EstimationError(para, method=method)
    _estimation_error.update_error_from_model_l0(model_l0, para, inf_1)
    l.append(_estimation_error)
    
    if verbose: print("PASTIS")
    if para.use_PASTIS:
        l_special_convention = [None, "Pierre_idea"]
        for special_convention in l_special_convention:
            if para.loop_on_p_PASTIS and special_convention is None:
                _gen = np.geomspace(0.001, 5, 20)
                _gen = np.append(_gen, [0.001, 0.01, 0.1, 1])
                _gen = np.unique(_gen)    
            else:
                _gen = [0.001]
            convention = para.convention
            if special_convention is not None:
                convention += " " + special_convention
            for p in _gen:
                time_SBR = time.time()
                model_l0 = S.L0_SBR(inf_1, p=p, special_convention=special_convention, verbose=verbose) #type: ignore
                time_SBR = time.time() - time_SBR
                method = ("PASTIS_p_%s"%p + "_" + convention + "_A_" + str(para.A_normalisation) + "_diffusion_" + str(para.diffusion))
                SBR_finds_real_model = tuple(para.model.index_real_base) in model_l0.d_information.keys()
                
                real_model_on_pareto_front = False
                if p == 0.001 and special_convention is None:
                    real_model_on_pareto_front = explore_model_l0(model_l0, inf_1, para)
                
                SBR_found_better_minimum = model_l0.compute_information(model_l0.best_nodes) >= model_l0.compute_information(tuple(para.index_real_base))
                _estimation_error = EstimationError(para, method=method, 
                                                    real_model_on_pareto_front=real_model_on_pareto_front, SBR_finds_real_model=SBR_finds_real_model,
                                                    SBR_found_better_minimum=SBR_found_better_minimum, time_to_compute=time_SFFI+time_SBR)
                _estimation_error.update_error_from_model_l0(model_l0, para, inf_1)
                l.append(_estimation_error)
        
    if verbose: print("BIC")
    if para.use_BIC:
        use_BIC = True
        time_SBR = time.time()
        model_l0 = S.L0_SBR(inf_1, use_BIC=use_BIC, verbose=verbose) #type: ignore
        time_SBR = time.time() - time_SBR
        method = ("BIC_" + str(use_BIC) + "_" + para.convention + "_A_" + str(para.A_normalisation) +  "_diffusion_" + str(para.diffusion))
        _estimation_error = EstimationError(para, method=method, time_to_compute=time_SFFI+time_SBR)
        _estimation_error.update_error_from_model_l0(model_l0, para, inf_1)
        l.append(_estimation_error)
        
    if verbose: print("AIC")
    if para.use_AIC:
        time_SBR = time.time()
        model_l0 = S.L0_SBR(inf_1, use_AIC=True, verbose=verbose) #type: ignore
        time_SBR = time.time() - time_SBR
        method="AIC_" + para.convention + "_A_" + str(para.A_normalisation) + "_diffusion_" + str(para.diffusion)
        _estimation_error = EstimationError(para, method=method, time_to_compute=time_SFFI+time_SBR)
        _estimation_error.update_error_from_model_l0(model_l0, para, inf_1)
        l.append(_estimation_error)
    
    if verbose: print("CV")
    if para.use_CrossValidation:
        _gen_k_validation = [7]
        if para.benchmark_CV_k_validation:
            _gen_k_validation = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        for k_validation in _gen_k_validation:
            time_SBR = time.time()
            model_CV = CV.CrossValidation(inf_1, k_validation=k_validation, verbose=verbose)
            time_SBR = time.time() - time_SBR
            model_CV.best_nodes = model_CV.get_best_nodes()
            method = "CrossValidation_threshold_"+ str(k_validation)
            _estimation_error = EstimationError(para, method=method, time_to_compute=time_SFFI+time_SBR)
            _estimation_error.update_error_from_model_l0(model_CV, para, inf_1)
            l.append(_estimation_error)
            del model_CV
    
    A_normalisation = inf_1.A_normalisation
    del inf_1
    del model_l0
    gc.collect()
                
    if verbose: print("SINDy")    
    if para.use_sindy:
        _gen = list(para.l_threshold_sindy.copy())
        _gen.append(para.threshold_sindy)
        for threshold_sindy in _gen:
            time_sindy = time.time()
            optimizer_STLSQ = ps.STLSQ(threshold=threshold_sindy)
            try:
                base_infered, F_estimated, F_prediction = model_sindy(x_train, dt, para, optimizer=optimizer_STLSQ, feature_library=para.feature_library)
                time_sindy = time.time() - time_sindy
                error = inf.real_error(F_estimated, para.F_real, A=A_normalisation)
                predictive_error = inf.real_error(F_prediction, para.F_real_predictive_error, A=A_normalisation)
                ratio_error_real_model = error / para.error_real_model 
                ratio_error_real_model_predictive = predictive_error / para.error_real_model_predictive
                l.append(
                    EstimationError(para, error=error, ratio_error_real_model=ratio_error_real_model, 
                                    predictive_error=predictive_error, ratio_error_real_model_predictive=ratio_error_real_model_predictive,
                                    base_infered=base_infered, real_base=para.index_real_base_sindy,
                                    method="STLSQ_threshold_" + str(threshold_sindy), time_to_compute=time_sindy))
            except Exception as e:
                print("Error in SINDy with STLSQ", e)
                continue
    
    if verbose: print("Lasso")
    if para.use_lasso:
        _gen = list(para.l_threshold_lasso.copy())
        _gen.append(para.threshold_lasso)
        for threshold_lasso in _gen:
            time_lasso = time.time()
            optimizer = Lasso(alpha=threshold_lasso, fit_intercept=False)
            base_infered, F_estimated, F_prediction = model_sindy(x_train, dt, para, optimizer=optimizer, feature_library=para.feature_library) 
            time_lasso = time.time() - time_lasso
            error = inf.real_error(F_estimated, para.F_real, A=A_normalisation)
            ratio_error_real_model = error / para.error_real_model 
            l.append(
                EstimationError(para, error=error, ratio_error_real_model=ratio_error_real_model,
                                base_infered=base_infered, real_base=para.index_real_base_sindy,
                                method="Lasso_threshold_" + str(threshold_lasso), time_to_compute=time_lasso))
        
    if para.use_ensemble_sindy:
        optimizer_ensemble = ps.EnsembleOptimizer(
            ps.STLSQ(threshold=para.threshold_sindy),
            bagging=True,
        )
        base_infered, F_estimated, F_prediction = model_sindy(x_train, dt, para, optimizer=optimizer_ensemble, feature_library=para.feature_library)  
        error = inf.real_error(F_estimated, para.F_real, A=A_normalisation)
        ratio_error_real_model = error / para.error_real_model 
        l.append(
            EstimationError(para, error=error, ratio_error_real_model=ratio_error_real_model,
                            base_infered=base_infered, real_base=para.index_real_base_sindy,
                            method="Ensemble_STLSQ_threshold_" + str(para.threshold_sindy))
            )
    if para.use_weak_sindy:
        t_train =  np.array(range(x_train.shape[0])) * dt
        #library_functions = [lambda x: x, lambda x, y: x * y, lambda x: x ** 2]
        #library_function_names = [lambda x: x, lambda x, y: x + y, lambda x: x + x]
        ode_lib = ps.WeakPDELibrary(
            function_library=para.feature_library,
            #library_functions=library_functions,
            #function_names=library_function_names,
            spatiotemporal_grid=t_train,
            include_bias=True,
            is_uniform=True,
        )
        optimizer_ensemble = ps.EnsembleOptimizer(
            ps.STLSQ(threshold=para.threshold_sindy),
            bagging=True,
        )
        base_infered, F_estimated, F_prediction = model_sindy(x_train, dt, para, optimizer=optimizer_ensemble, feature_library=ode_lib) 
        error = inf.real_error(F_estimated, para.F_real, A=A_normalisation)
        ratio_error_real_model = error / para.error_real_model
        l.append(EstimationError(para, error=error, ratio_error_real_model=ratio_error_real_model,
                                 base_infered=base_infered, real_base=para.model.index_real_base_sindy_weak_form,
                method="Weak_Ensemble_STLSQ_threshold_" + str(para.threshold_sindy))
            )
    
    if para.use_ConstrainedSR3:
        optimizer = ps.ConstrainedSR3(
            threshold=para.threshold_sindy,
            constraint_rhs=para.constraint_rhs, constraint_lhs=para.constraint_lhs, max_iter=1000
            )
        base_infered, F_estimated, F_prediction = model_sindy(x_train, dt, para, optimizer=optimizer, feature_library=para.feature_library) 
        error = inf.real_error(F_estimated, para.F_real, A=A_normalisation)
        ratio_error_real_model = error / para.error_real_model
        l.append(EstimationError(para, error=error, ratio_error_real_model=ratio_error_real_model,
                                 base_infered=base_infered, real_base=para.index_real_base_sindy,
                method="ConstrainedSR3_threshold_" + str(para.threshold_sindy))
            )
    return l

def explore_model_l0(model_l0, inf_1, para):
    all_combinations = combinations(range(inf_1.coefficients.shape[0]), len(para.index_real_base))
    if comb(inf_1.coefficients.shape[0], len(para.index_real_base)) > 10**5:
        all_combinations = []
        other_indices = [j for j in range(inf_1.coefficients.shape[0]) if j not in para.index_real_base]
        for i in para.index_real_base:
            all_combinations.extend([tuple(np.sort(para.index_real_base[:i] + para.index_real_base[i+1:] + [j])) for j in other_indices])
    info_real_model = model_l0.compute_information(tuple(para.index_real_base))
    max_key = max((k for k in model_l0.d_information if len(k) == len(para.index_real_base)), key=model_l0.d_information.get, default=None) #type: ignore
    real_model_on_pareto_front = (list(np.sort(max_key)) == list(para.index_real_base)) #type: ignore
    if real_model_on_pareto_front:
        for combo in all_combinations:
            _info = model_l0.compute_information(tuple(set(combo)))
            if _info > info_real_model and tuple(list(np.sort(combo))) != tuple(para.index_real_base):
                #print("Model on pareto front but not the best %s"%str(combo))
                break
    max_key = max((k for k in model_l0.d_information if len(k) == len(para.index_real_base)), key=model_l0.d_information.get, default=None) #type: ignore
    real_model_on_pareto_front = (list(np.sort(max_key)) == list(para.index_real_base)) #type: ignore
    return real_model_on_pareto_front
            