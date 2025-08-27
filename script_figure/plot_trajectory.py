from .helper_figure import add_letter
from SFFI import util_plot as ut_plot
from simulation_models._common_class import Model
from _helper._load_csv import read_csv
import matplotlib.pyplot as plt
import numpy as np
from brokenaxes import BrokenAxes
import os
import sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path:
    sys.path.append(dir1)
if not dir2 in sys.path:
    sys.path.append(dir2)


def plot_trajectory_OU(x, ax, delta_t=0.1, subplot_spec=None, fig=None,
                       time_start=10, time_end=1, letter="", **kwargs):
    delta = (np.max(x) - np.min(x))/x.shape[1]
    times = delta_t*np.arange(x.shape[0])
    lim_x_1 = min(int(time_start/delta_t), times.shape[0]-1)
    lim_x_2 = min(int(time_end/delta_t), times.shape[0]-1)

    bax = BrokenAxes(subplot_spec=subplot_spec, xlims=((0, times[lim_x_1]), (times[-lim_x_2], times[-1])),
                     d=False, despine=False, fig=fig,
                     wspace=-1, **kwargs)

    for i in range(x.shape[1]):
        ax = bax.plot(times, x[:, i] + i*delta)

    for i, ax in enumerate(bax.axs):
        if i == 0:
            ax.set_title("   Trajectory OU ")
            ax.set_xlabel("  Time ")
            ax.set_ylabel("    $ $ ")

    # bax.set_xlabel('Time', labelpad=20)
    bax.axs[0].set_yticks([])
    # if fig is not None:
    #     fig.tight_layout()
    #     for handle in bax.diag_handles:
    #         handle.remove()
    #     bax.draw_diags()
    if letter != "":
        add_letter(bax.axs[0], letter)
    return bax


def plot_trajectory_OU_simple(x, ax, delta_t=0.1,
                              letter="", ):
    times = delta_t*np.arange(x.shape[0])
    ax.plot(times, x)
    # ax.set_title("Trajectory OU ")
    ax.set_xlabel("Time ")
    ax.set_ylabel("")
    ax.margins(x=0)
    ax.margins(y=0)
    ax.set_yticks([])

    if letter != "":
        add_letter(ax, letter)
    return ax


def round_number(number):
    # Convert to scientific notation
    scientific_notation = "{:e}".format(number)
    # Split the number into the base and exponent
    base, exponent = scientific_notation.split("e")
    # Adjust the base to one decimal place
    base = "{:.0f}".format(float(base))
    # Format the exponent by removing '+' sign and leading zeros
    exponent = exponent.replace("+", "").lstrip("0")
    if exponent == "":
        exponent = "0"
    exponent = int(exponent)
    # Combine into the desired format
    if base == "1":
        formatted_number = f"${{10}}^{{{exponent}}}$"
    else:
        formatted_number = f"${base}.{10}^{{{exponent}}}$"
    return formatted_number


def plot_trajectory_Lorenz(name_csv, model: type[Model],  parameter="D_strength", parameters_value_plot=None,
                           subplotspec=None, func_plot_trajectory=ut_plot.plot_trajectory, cmaps_trajectory="viridis",
                           correct_name_short=None, time_max=None, aspect_color_bar=20, colorbar=False, add_title=True
                           ):
    df = read_csv(name_csv)
    parameter_values = np.sort(list(df[parameter].unique()))
    if parameters_value_plot is None:
        # [0, parameter_values.shape[0]//2, parameter_values.shape[0]-1]
        index_parameters = [parameter_values.shape[0] //
                            5,  8*parameter_values.shape[0]//10]
        parameters_value_plot = [parameter_values[i] for i in index_parameters]
    else:
        index_parameters = [min(range(len(parameter_values)), key=lambda i: abs(
            parameter_values[i] - value)) for value in parameters_value_plot]


    dict_para = df.iloc[0].to_dict()
    model_wrapper = model(**dict_para["init_params"])
    if time_max is not None:
        dict_para["time"] = time_max
    fig = plt.gcf()
    if subplotspec is None:
        subplotspec = fig.add_gridspec(1, 1)[0]
    gs1 = subplotspec.subgridspec(1, len(parameters_value_plot))
    im = None
    l_ax = []
    for i, value_parameter in enumerate(parameters_value_plot):
        norm = None if im is None else im.norm
        if parameter == "dt":
            norm = None  # Correct bug
        dict_para[parameter] = value_parameter
        print(dict_para)
        model_wrapper.fast_random = np.random.default_rng(0)
        x, dt = model_wrapper.simu_from_EstimationError(dict_para)
        ax = fig.add_subplot(gs1[i])
        l_ax.append(ax)
        im = func_plot_trajectory(
            x, ax=ax, delta_t=dt, norm=norm, cmap=cmaps_trajectory)

        name_short = correct_name_short if correct_name_short is not None else parameter
        if add_title:
            ax.set_title("%s = %s" %
                         (name_short, round_number(parameters_value_plot[i])))
        if i == len(index_parameters)-1 and im is not None and colorbar:
            plt.colorbar(im, label="Time", ax=ax,
                         location="right", aspect=aspect_color_bar)
        if i == 0:
            # placeholder_ax.text(-0.18, 0.5, '2D projected data', transform=placeholder_ax.transAxes, ha='center', va='center', rotation=90)
            # placeholder_ax.set_ylabel("2D projected data")
            pass
    return l_ax


def plot_trajectory_Lorenz_simple(name_csv, model: type[Model],  ax=None, parameter="D_strength", parameters_value_plot=None,
                                  func_plot_trajectory=ut_plot.plot_trajectory, cmaps_trajectory="viridis",
                                  correct_name_short=None, time_max=None, aspect_color_bar=20, colorbar=False, add_title=True,
                                  d_update_para = {}, **kwargs
                                  ):
    if ax is None:
        fig, ax = plt.subplots()
    df = read_csv(name_csv)
    parameter_values = np.sort(list(df[parameter].unique()))
    if parameters_value_plot is None:
        # [0, parameter_values.shape[0]//2, parameter_values.shape[0]-1]
        index_parameters = [8*parameter_values.shape[0]//10]
        parameters_value_plot = [parameter_values[i] for i in index_parameters]
    else:
        index_parameters = [min(range(len(parameter_values)), key=lambda i: abs(
            parameter_values[i] - value)) for value in parameters_value_plot]

    dict_para = df.iloc[0].to_dict()
    model_wrapper = model(over_sampling=10, thermalised_time=10, **dict_para["init_params"])
    dict_para[parameter] = parameters_value_plot[0]
    dict_para.update(d_update_para)
    if time_max is not None:
        dict_para["time"] = time_max
    model_wrapper.fast_random = np.random.default_rng(0)
    print(dict_para)
    x, dt = model_wrapper.simu_from_EstimationError(dict_para)

    im = func_plot_trajectory(x, ax=ax, delta_t=dt,
                              cmap=cmaps_trajectory, **kwargs)

    name_short = correct_name_short if correct_name_short is not None else parameter
    if add_title:
        ax.set_title("%s = %s" %
                     (name_short, round_number(parameters_value_plot[0])))
    if colorbar:
        plt.colorbar(im, label="Time", ax=ax,
                     location="right", aspect=aspect_color_bar)
    return ax


def plot_trajectory_Lotka_Volterra(name_csv, model: type[Model],  parameter="D_strength", parameters_value_plot=None,
                                   subplot_spec=None, time_start=10, time_end=2, max_time=None, **kwargs):
    df = read_csv(name_csv)
    parameter_values = np.sort(list(df[parameter].unique()))
    if parameters_value_plot is None:
        # [0, parameter_values.shape[0]//2, parameter_values.shape[0]-1]
        index_parameters = [parameter_values.shape[0]-1]
    else:
        index_parameters = [min(range(len(parameter_values)), key=lambda i: abs(
            parameter_values[i] - value)) for value in parameters_value_plot]

    index_parameters = [index_parameters[0]]
    parameters_value_plot = [parameter_values[i] for i in index_parameters]

    row_selected = df[df[parameter] == df[parameter].max()].iloc[0]
    dict_para = row_selected.to_dict()
    model_wrapper = model(**dict_para["init_params"])
    if max_time is not None:
        dict_para["time"] = max_time

    bax = None
    for i, index in enumerate(index_parameters):
        dict_para[parameter] = parameter_values[index]
        model_wrapper.fast_random = np.random.default_rng(0)
        x, dt = model_wrapper.simu_from_EstimationError(dict_para)

        times = dt*np.arange(x.shape[0])
        time_start = min(time_start, times[-10])
        lim_x_1 = int(time_start/dt)
        lim_x_2 = int(time_end/dt)

        bax = BrokenAxes(subplot_spec=subplot_spec, xlims=((0, times[lim_x_1]), (times[-lim_x_2], times[-1])),
                         d=False, despine=False, wspace=-1, **kwargs)

        bax.plot(times, x)
        bax.set_xlabel('Time', labelpad=20)

        for i, ax in enumerate(bax.axs):
            if i == 0:
                ax.set_title("   Trajectory ")
                ax.set_xlabel("  Time ")
                ax.set_ylabel("    $ $ ")
    return bax 


def plot_trajectory_Lotka_Volterra_simple(name_csv, model: type[Model],  parameter="D_strength", parameters_value_plot=None,
                                          ax=None, max_time=None, df=None, time_max=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    if df is None:
        df = read_csv(name_csv)
    parameter_values = np.sort(list(df[parameter].unique()))
    if parameters_value_plot is None:
        # [0, parameter_values.shape[0]//2, parameter_values.shape[0]-1]
        index_parameters = [parameter_values.shape[0]-1]
        parameters_value_plot = [parameter_values[i] for i in index_parameters]
    else:
        index_parameters = [min(range(len(parameter_values)), key=lambda i: abs(
            parameter_values[i] - value)) for value in parameters_value_plot]

    row_selected = df[df[parameter] == df[parameter].max()].iloc[0]
    dict_para = row_selected.to_dict()
    print(dict_para["init_params"])
    if time_max is not None:
        dict_para["time"] = time_max
    model_wrapper = model(**dict(dict_para["init_params"]))
    print("index real base" , len(model_wrapper.index_real_base))
    print("total base", len(model_wrapper.total_base))
    for parameter_values in parameters_value_plot:
        dict_para[parameter] = parameter_values
        if max_time is not None:
            dict_para["time"] = max_time
            dict_para["n"] = int(max_time/dict_para["dt"])
        model_wrapper.fast_random = np.random.default_rng(0)
        x, dt = model_wrapper.simu_from_EstimationError(dict_para)
        if np.isnan(np.sum(x)):
            print("Nan value")
        times = dt * np.arange(x.shape[0])
        ax.plot(times, x, **kwargs)
        ax.margins(x=0)
        ax.margins(y=0)
        ax.set_xlabel('Time')
        # ax.set_yticks([0])
        ax.set_xticks([int(times[0]), int(times[-1])])
    return ax
