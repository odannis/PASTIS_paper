import os, sys
dir2 = os.path.abspath('')
dir1 = os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)
if not dir2 in sys.path: sys.path.append(dir2)

import pandas as pd
import numpy as np
from _helper._load_csv import  read_csv
from _helper._figures import rename_legend, move_legend_on_line
import SFFI.util_plot as ut_plot
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns

try:
    from . import labels_plot as lp
except:
    from script_figure import labels_plot as lp


def concatenate_dim(name_base, save_csv = False, type_csv = "csv"):
    # Initialize an empty list to store the dataframes
    dataframes = []

    # Loop through the files and add the 'dim' column
    for dim in range(1, 11):  # From 3 to 8 inclusive
        try:
            filename = f'csv/{name_base}_dim_{dim}_n.{type_csv}'
            df = read_csv(filename)  # Load the CSV file
            df['dim'] = dim  # Add the 'dim' column
            dataframes.append(df)  # Append the dataframe to the list
        except FileNotFoundError:
            print(f"File {filename} not found")

    # Concatenate all dataframes into one
    aggregated_df = pd.concat(dataframes, ignore_index=True)
    if save_csv:
        aggregated_df.to_csv(f'csv/{name_base}_aggregated.csv', index=False)
    return aggregated_df

# Initialize a list to store the dimension and time to reach 95% average exact model found
def plot_time_vs_key(parameter="D_strength", percent_accept = 0.95, name_csv='lorenz_critical_constant_noise_diffusion_vs_time_n.pkl',
                accuracy = "Accuracy_model", method_to_plot = ["PASTIS", "AIC", "SINDy", "BIC"], ax=None, df = None, bottom = True,  top = True,
                aggregate_csv = False, dashes = True, label_dict = None, legend : str|bool = "auto", move_legend = False,
                min_label_distance = "auto", xscale = "log", yscale = "log", marker=".", **kwargs):
    if label_dict is None:
        label_dict = lp.global_dict
    labels_plot = {key : value[0] for key, value in label_dict.items() if value[0] in method_to_plot}
    color_dict = {key : value[1] for key, value in label_dict.items()}
    if ax is None:
        fig, ax = plt.subplots()
    time_to_95_avg = []
    if aggregate_csv:
        df = concatenate_dim(name_csv, type_csv="pkl")
    elif df is None:
        df = read_csv(name_csv)
    nan_value = 10**10
    # Loop through each dimension
    for i in df[parameter].unique():
        # Read the data for each dimension
        mask = df[parameter] == i
        time = df[mask].time.unique().max()
        method = df[mask].method.unique()[0]
        mask_h = (df.method == method) & (df[parameter] == i) & (df.time == time)
        print("Average on %s samples for %s with time %s"%(len(df[mask_h]), i, time))
        avg_exact_model_found_all = df[mask].groupby(['time', "method"])[accuracy].mean().reset_index()
        for method in avg_exact_model_found_all.method.unique():
            mask = avg_exact_model_found_all.method == method
            avg_exact_model_found = avg_exact_model_found_all[mask]
            time_95_avg = avg_exact_model_found[(avg_exact_model_found[accuracy] >= percent_accept)]['time']
            time_95_avg = time_95_avg.min() if not time_95_avg.empty else np.nan
            if pd.notna(time_95_avg):
                pass
                # time_before = avg_exact_model_found[avg_exact_model_found.time < time_95_avg]["time"].max()
                # if pd.notna(time_before):
                #     percent_before = avg_exact_model_found[avg_exact_model_found.time == time_before][accuracy].values[0] 
                #     percent_after = avg_exact_model_found[avg_exact_model_found.time == time_95_avg][accuracy].values[0]
                #     if percent_before < percent_accept:
                #         time_95_avg_after = time_before + (time_95_avg - time_before) * (percent_accept - percent_before)/(percent_after - percent_before)
                #         time_95_avg = time_95_avg_after
                #     else:
                #         print("er")
            else:
                time_95_avg = nan_value
            time_to_95_avg.append({parameter: i, 'Time': time_95_avg, "method": method})

    # Create a DataFrame from the collected times
    time_to_95_avg_df = pd.DataFrame(time_to_95_avg)
    mask = None
    for key in labels_plot.keys():
        if mask is None:
            mask = time_to_95_avg_df['method'].str.endswith(key) 
        else:
            mask += time_to_95_avg_df['method'].str.endswith(key) 
        
    g = sns.lineplot(time_to_95_avg_df[mask], x=parameter, y='Time', hue='method', marker=marker, #type: ignore
                     palette=color_dict, ax=ax, dashes=dashes, legend=legend, **kwargs) #type: ignore
    if top:
        g.set_ylim(top=np.max(df['time']))
    if bottom:
        print("min", np.min(df["time"]))
        g.set_ylim(bottom=np.min(df["time"]), top=np.max(df['time']))
        
    if parameter in lp.d_correct_label.keys():
        g.set_xlabel(lp.d_correct_label[parameter])
        
    rename_legend(g.legend(), labels_plot, ax=g)
    g.set_xscale(xscale)
    g.set_yscale(yscale)
    if move_legend:
        move_legend_on_line(g, min_label_distance=min_label_distance)
    try:
        #g.set_title("%s %s"%(percent_accept, lp.d_correct_label[accuracy]))
        g.set_ylabel("Time to reach %s \n %s"%(percent_accept, lp.d_correct_label[accuracy]))
    except:
        pass
    else:
        sns.despine(ax=g)
        g.set_xmargin(0)
        g.set_ymargin(0)
    return g

def add_Pareto_front(ax : Axes, name_csv="lotka_volterra_simple", aggregate_csv=True, x_text=None, y_text=70, xscale="linear",
                     fontsize=8, parameter="dim", percent_accept=0.99, method=r"PASTIS", accuracy="real_model_on_pareto_front"):
    name_Pastis = lp.find_key_for_value_in_global_dict(method)
    label_dict_1 = {name_Pastis : ("M*=Pareto", "black")}
    bottom, top = ax.get_ylim()
    g = plot_time_vs_key(name_csv=name_csv, parameter=parameter,
                        percent_accept=percent_accept, accuracy=accuracy,
                        method_to_plot=["M*=Pareto"], ax=ax,
                        aggregate_csv=aggregate_csv, bottom=False, label_dict=label_dict_1,
                        legend=False, xscale=xscale)
    ax.legend().remove()

    i = -1
    # for j in range(len(g.lines)):
    #     print(g.lines[j])
    #     print(g.lines[j].get_ydata())
    line = g.lines[i]
    x = line.get_xdata()
    y = line.get_ydata()
    if x_text is None:
        x_text = np.min(x) + (np.max(x) - np.min(x))/2
    print("x_text", x_text)

    g.fill_between(x, y, y2=bottom, interpolate=True, color='grey', alpha=0.2)#hatch="XX")
    #g.fill_between(x, y, y2=bottom, interpolate=True,  facecolor="none", hatch="XX", edgecolor="grey", linewidth=0.0, alpha=0.5)
    line.remove()
    g.set_ylim(bottom=bottom)
    text = lp.label_pareto_front
    g.text(x_text, y_text, text, rotation=0.,
            ha="center", va="center", fontsize=fontsize,
            # bbox=dict(boxstyle="round",
            #         ec=(0, 0, 0),
            #         fc=(1., 1, 1),
            #         )
            )

if __name__ == "__main__":
    plot_time_vs_key()
    plt.show()