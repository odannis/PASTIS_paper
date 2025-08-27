import numpy as np
try:
    from ._database import InferenceParameter, loop_over_parameter_simulation
except ImportError:
    from _database import InferenceParameter, loop_over_parameter_simulation
import numpy as np
import warnings
import jax
import psutil, os

def show_ram():
    # # Récupérer les informations sur la mémoire virtuelle
    # ram_info = psutil.virtual_memory()
    # # Afficher la mémoire utilisée en pourcentage
    # print(f"Pourcentage de mémoire utilisée : {ram_info.percent}%")
    # # Afficher la mémoire utilisée en Go
    # print(f"Mémoire utilisée (Go) : {ram_info.used / (1024 ** 3):.2f} Go")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Memory usage in MB
    print(f"Memory used: {mem_info.rss / (1024 ** 2):.2f} MB")
#ignore by message
warnings.filterwarnings("ignore", message="Sparsity")

def simu_and_save(para : InferenceParameter, number_simu : str = "0", name_csv = "test", 
            l_diffusion_strength = None, l_n = None, l_dt = None,
            l_experimental_noise = None, l_thresholds_sindy = None,
            conventions=[("Ito", "Constant"), ("Strato", "Multiplicative")],
            l_diffusion_vs_time = None,
            ):
    model = para.model
    if name_csv != "test":
        try:
            model.rng = np.random.default_rng(int(number_simu))
        except ValueError:
            print("Error: number_simu must be a number")
            pass
        
    for i, convention in enumerate(conventions):#["Strato", "Strato_large_dt", "Ito", "Ito_large_dt"]:
        para.convention = convention[0]
        para.diffusion = convention[1]
        if l_n is not None:
            loop_over_parameter_simulation(para, l_n, "n", name_csv, number_simu=number_simu)
        if l_dt is not None:
            loop_over_parameter_simulation(para, l_dt, "dt", name_csv, number_simu=number_simu)
        if l_experimental_noise is not None:
            loop_over_parameter_simulation(para, l_experimental_noise, "experimental_noise", name_csv, number_simu=number_simu)
        if l_diffusion_strength is not None:
            loop_over_parameter_simulation(para, l_diffusion_strength, "diffusion_strength", name_csv, number_simu=number_simu)
        if l_diffusion_vs_time is not None:
            l_diffusion, l_time = l_diffusion_vs_time
            for diffusion_strength in l_diffusion:
                #show_ram()
                para.model.diffusion_strength = diffusion_strength
                loop_over_parameter_simulation(para, l_time, "n", name_csv + "_diffusion_vs_time", number_simu=number_simu)
                jax.clear_caches()
            # for n in l_time:
            #     n_save = para.model.n
            #     para.model.n = n
            #     loop_over_parameter_simulation(para, l_diffusion, "diffusion_strength", name_csv + "_diffusion_vs_time", number_simu=number_simu)
            #     para.model.n = n_save
            #     jax.clear_caches()

        
if __name__ == "__main__":
    ### Test function ###
    from simulation_models.lorenz import Lorenz
    r,b,s = 10., 1., 3.
    num = 3
    model = Lorenz(r=r, b=b, s=s, n=1_000_000, dt=0.01)
    l_diffusion_strength = np.geomspace(0.000001, 100, num)
    l_n = np.geomspace(1_000, 100_000, 20).astype(int)
    l_dt = np.round(np.geomspace(model.dt, 0.1, num) / model.dt) * model.dt
    l_experimental_noise = np.geomspace(0.001, 0.1,  num)
    para = InferenceParameter(model)
    simu_and_save(para, l_diffusion_strength=l_diffusion_strength)
