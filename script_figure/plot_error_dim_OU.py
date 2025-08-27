import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from .helper_figure import add_letter
from  . import labels_plot as lp
from _helper._load_csv import read_csv
d_correct_label = lp.d_correct_label

def plot_error_dim(name_csv="OrnsteinUhlenbeck_dim", max_dim=13, ax=None,
                   colormap = colormaps.get_cmap("viridis"), 
                min_dim=3, method_to_plot="PASTIS", y="Accuracy_model", x="time",
                letter="", type_file="pkl"):
    if ax is None:
        fig, ax = plt.subplots()
    color_steps = np.linspace(0, 1, max_dim - 2)
    method_name = [key for key, value in lp.global_dict.items() if value[0] == method_to_plot][0]
    # Plotting loop
    for i, color_step in zip(range(min_dim, max_dim), color_steps):
        try:
            df = read_csv(f'csv/{name_csv}_{i}_n.{type_file}')
            mask = df['method'].str.fullmatch(method_name)
            df = df[mask]
            sns.lineplot(x=x, y=y, data=df, color=colormap(color_step),
                        ax=ax, label=f'$d = {i}$')
            ax.set_xscale('log')
        except Exception as e:
            print(e)
    if y in d_correct_label.keys():
        ax.set_ylabel(d_correct_label[y])
    if x in d_correct_label.keys():
        ax.set_xlabel(d_correct_label[x])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title(method_to_plot)
    if letter != "":
        add_letter(ax, letter)
    return ax


if __name__ == "__main__":
    plot_error_dim()
    plt.show()