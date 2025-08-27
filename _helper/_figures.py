import _helper.utils as utils
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from SFFI import util_plot as ut_plot
from simulation_models._common_class import Model
from matplotlib.axes import Axes
from typing import List
import matplotlib.pyplot as plt
from ._load_csv import read_csv
from SFFI.util_inference.inference_help import evaluate_deriv_all_base

def add_last_y_data_on_line(g: Axes):
    for i in range(g.lines.__len__()-1):
        for j in range(i+1, g.lines.__len__()):
            if g.lines[i].get_color() == g.lines[j].get_color():
                try:
                    g.lines[j].set_ydata(np.array([g.lines[i].get_ydata()[-1]])) #type: ignore
                except Exception as e:
                    print(e)
                    pass
                break
            
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

def rename_legend(legend, new_labels, ax : Axes|None = None):
    for t in legend.texts:
        for k, v in new_labels.items():
            if k == t.get_text():
                t.set_text(v)
    if ax is not None:
        for child in ax.get_children():
            for k, v in new_labels.items():
                if k == child.get_label():
                    child.set_label(v)
                    
def move_legend_on_line(ax: Axes, min_label_distance="auto", alpha=1.0):
    import matplotx
    add_last_y_data_on_line(ax)
    matplotx.line_labels(ax=ax, min_label_distance=min_label_distance, alpha=alpha)
    ax.legend().remove() 
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.margins(x=0, y=0.02)

def put_letter(ax, letter, x=-0.15, y=1.1, **kwargs):
    ax.text(x, y, letter, ha='left', va='bottom', transform=ax.transAxes, fontsize=10, fontweight='bold',  **kwargs)
    
def annotate_figure(ax : Axes, annotation_text : str, x_pos_text : float, ax2 : None|Axes = None):
    ax.vlines(x_pos_text, *ax.get_ylim(), linestyles='dashed', color='black')
    point_arrow = (x_pos_text, ax.get_ylim()[1])
    ax.annotate(annotation_text, xy=point_arrow, xytext=(0, 0), textcoords='offset points', ha='center', va='bottom')
    if ax2 is not None:
        ax2.vlines(x_pos_text, *ax2.get_ylim(), linestyles='dashed', color='black')
