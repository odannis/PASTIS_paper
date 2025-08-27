import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import numpy as np
from _helper._load_csv import  read_csv
from _helper._figures import rename_legend, move_legend_on_line
import SFFI.util_plot as ut_plot
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from . import labels_plot as lp
from .helper_figure import add_letter
from typing import List, Tuple, Dict
from matplotlib import ticker as mticker

d_correct_label = lp.d_correct_label


def plot_from_csv(csv_use : str, l_y_plot = ["Accuracy_model", "error"], parameter = "time", errorbar = ("ci", 95), 
                  method_to_plot = ["AIC", "PASTIS", "PASTIS AAIC", "Total model", "Real model"], add_legend_on_line = True, axs = None, max_x_axis = None, min_x_axis=None,
                  max_y_axis = None, min_y_axis = None, verbose = False, label_dict = None, add_real_model = True, letter = [""], dashes : bool|List|Tuple = True,
                  hue = "method", color_area = False, color_under_area = "grey", hatch = None, legend : str|bool = "auto", df = None, method_to_put_last="PASTIS",
                  alpha_min_distance_label=1, **kwargs) -> List[Axes]:
    if df is None:
        df = read_csv(csv_use)
    if verbose : print("method in csv avalaible %s"%df.method.unique())
    if label_dict is not None:
        labels_plot = {key : value[0] for key, value in label_dict.items() if value[0] in method_to_plot}
        color_dict = {key : value[1] for key, value in label_dict.items()}
    else:
        labels_plot = {key : value[0] for key, value in lp.global_dict.items() if value[0] in method_to_plot}
        color_dict = {key : value[1] for key, value in lp.global_dict.items()}
    if verbose : print(labels_plot)
    if axs is None: 
        subplot = (len(l_y_plot), 1)
        fig = plt.figure(figsize=ut_plot.set_size("article", subplots=subplot, scale_height=1), layout="constrained") #type: ignore
        gs0 = fig.add_gridspec(*subplot)
        axs = [fig.add_subplot(gs0[i]) for i in range(len(l_y_plot))]
    l_g = []
    if max_x_axis is not None:
        mask = df[parameter] <= max_x_axis
        df = df[mask]
    if min_x_axis is not None:
        mask = df[parameter] >= min_x_axis
        df = df[mask]
    for i, y_plot in enumerate(l_y_plot):
        ax_1 : Axes = axs[i]
        hue_order = list(labels_plot.keys())
        if not "error" in y_plot or not add_real_model:
            for key, value in labels_plot.items():
                # if "model" in value:
                #     hue_order.pop(hue_order.index(key))
                pass
        if len(hue_order) == 0:
            hue_order = None
        else:
            # Find pastis in hue_order and put it first
            try:
                # Find the key for the value "PASTIS" in the dict labels_plot
                key_pastis = next(key for key, value in labels_plot.items() if value == method_to_put_last)
                index_pastis = hue_order.index(key_pastis)
                hue_order.pop(index_pastis)
                hue_order.append(key_pastis)
            except Exception as e:
                print(e)
        if verbose : print("hue_order %s"%hue_order)
        if verbose : print("y_plot %s"%y_plot)
        try:
            g = sns.lineplot(x=parameter, y=y_plot, hue=hue, data=df, ax=ax_1, hue_order=hue_order, errorbar=errorbar,
                            palette=color_dict, dashes=dashes, legend=legend, **kwargs) #type: ignore
        except Exception as e:
            print(e, color_dict, hue_order)
            print(df[hue].unique())
            #add_legend_on_line = False
            g = sns.lineplot(x=parameter, y=y_plot, hue=hue, data=df, ax=ax_1, errorbar=errorbar, hue_order=hue_order, dashes=dashes, legend=legend, **kwargs) #type: ignore

        if max_y_axis is not None:
            ax_1.set_ylim(bottom=min_y_axis, top=max_y_axis)
        if "error" in y_plot or "time" in y_plot:
            ax_1.set_yscale("log")
        if max_y_axis is not None:
            ax_1.set_ylim(bottom=min_y_axis, top=max_y_axis)
        
        if y_plot in d_correct_label.keys():
            g.set_ylabel(d_correct_label[y_plot])
        if parameter in d_correct_label.keys():
            g.set_xlabel(d_correct_label[parameter])
        
        if color_area:
            i = -1
            line = g.lines[i]
            x = line.get_xdata()
            y = line.get_ydata()
            g.fill_between(x, y, y2=np.max(y), interpolate=True, color=color_under_area, alpha=0.15, hatch=hatch)
            line.remove()
        
        if legend is not False:
            rename_legend(g.legend(), labels_plot, ax=g)
        ax_1.set_xscale("log")
        
        if add_legend_on_line:
            move_legend_on_line(g, alpha=alpha_min_distance_label)
        else:
            # box = g.get_position()
            # g.set_position((box.x0, box.y0, box.width * 0.8, box.height))
            # g.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            pass
            
        if letter[0] != "":
            try:
                add_letter(ax_1, letter[i])
            except Exception as e:
                print(e)
        
        g.xaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=10))
        #g.xaxis.set_major_formatter(mticker.LogFormatter())
        l_g.append(g)
        g.margins(x=0)
    return l_g

def add_pareto_zone(ax, name_csv_time, x_text = 5, y_text = 0.5, fontsize = 8, two_zone_plot = False, one_zone_plot="real_model_on_pareto_front", **kwargs):
    """
    Add pareto zone on the plot
    one_zone_plot : str (default : "SBR_found_better_minimum" or "real_model_on_pareto_front")
    """
    name_pastis = lp.find_key_for_value_in_global_dict("PASTIS")
    
    if not two_zone_plot:
        label_dict = {name_pastis: ("something", "black")}
        plot_from_csv(name_csv_time, 
                    parameter="time",
                    method_to_plot=["something"],
                    l_y_plot=[one_zone_plot],
                    axs=[ax],
                    add_legend_on_line=False,
                    label_dict=label_dict,
                    color_area=True,
                    color_under_area="grey",
                    errorbar=None,
                    legend=False,
                    **kwargs
                    )
    else:
        label_dict = {name_pastis: ("M*=SBR", "black")}
        label_dict_1 = {name_pastis : ("M*=Pareto", "black")}
        plot_from_csv(name_csv_time, 
                    parameter="time",
                    method_to_plot=["M*=Pareto"],
                    l_y_plot=["real_model_on_pareto_front"],
                    axs=[ax],
                    add_legend_on_line=False,
                    label_dict=label_dict_1,
                    color_area=True,
                    color_under_area="grey",
                    errorbar=None,
                    legend=False,
                    hatch="XX",
                    **kwargs
                    )
        try:
            plot_from_csv(name_csv_time, 
                        parameter="time",
                        method_to_plot=["M*=SBR"],
                        l_y_plot=["SBR_pareto_found"],
                        axs=[ax],
                        add_legend_on_line=False,
                        label_dict=label_dict,
                        color_area=True,
                        errorbar=None,
                        legend=False,
                        color_under_area="grey",
                        **kwargs,
                        )
        except Exception as e:
            print("fail to add sbr zone")
            print(e)

    # get lim axes
    text = lp.label_pareto_front
    x_text = ax.get_xlim()[0]*1.15
    ax.text(x_text, y_text, text, rotation=0.,
            ha="left", va="center",
            fontsize=fontsize
            )
    
if __name__ == "__main__":
    name_csv = "OrnsteinUhlenbeck_dim"
    csv_use = "{}_{}_n.csv".format(name_csv, 4)
    plot_from_csv(csv_use, method_to_plot=["AIC", "PASTIS", "PASTIS AAIC", "Total model", "Real model"])
    plt.show()