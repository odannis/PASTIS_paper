import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
#import mayavi.mlab
import numpy.typing as npt
from typing import Tuple
from matplotlib.axes import Axes

def set_size(width, fraction=1.0, subplots=(1, 1), scale_height = 1.0):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    scale_height : float
            Increase proportion in height
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    elif width == "article":
        width_pt = 243.77952756
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1]) * scale_height

    return (fig_width_in, fig_height_in)

def axisvector(index,dim):
    """d-dimensional vector pointing in direction index."""
    return np.array([ 1. if i == index else 0. for i in range(dim)])

def plot_process(data, tau, dim, shift=(0,0), ax : None | Axes = None, tmin=None, tmax=None, **kwargs):
    """Basic 2D plotting of the trajectory. The color gradient indicates
    time. dir1 and dir2 arguments, if specified, should be two
    orthogonal unit vectors, defining the projection plane of the
    representation. 'particle' argument indicates which particle
    should be considered, if relevant.
    """
    dir1 = axisvector(0, dim)
    dir2 = axisvector(1, dim)
    DX = data[1:] - data[:-1]
    dx = np.array([ dir1.dot(u[:]) for u in DX[tmin:tmax] ])
    dy = np.array([ dir2.dot(u[:]) for u in DX[tmin:tmax] ])
    x = np.array([ dir1.dot(u[:]) for u in data[tmin:tmax] ])[:-1]
    y = np.array([ dir2.dot(u[:]) for u in data[tmin:tmax] ])[:-1]
    dx = dx[:x.shape[0]]
    dy = dy[:y.shape[0]]
    tau = tau[tmin:tmax][:-1]
    if ax is None:
        ax = plt.gca()
    ax.axis('equal') 
    ax.set_xticks([])
    ax.set_yticks([])
    return ax.quiver(x+shift[0],y+shift[1], dx, dy, tau,
            headaxislength = 0.0, headwidth = 0., headlength = 0.0, minlength=0.,
            minshaft = 0., scale = 1.0, units = 'xy', lw=0.0,
            **kwargs)

def plot_trajectory(phi : np.ndarray, delta_t : float = 1.0, ax=None, show_time = False, cmap="viridis", width=1, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.5,5))
    if len(phi) > 10**5:
        print("reduce the number of points")
        mult = int(len(phi) / 10**5)
        phi = phi[::mult]
        delta_t *= mult
    tau = delta_t*np.array([i for i in range(len(phi))])
    im = None
    if len(phi.shape) == 1:
        ax.plot(tau, phi)
        im = ax
    else:
        d_dim = phi.shape[1]
        if phi.shape[-1] == 3 or phi.shape[-1] == 2:
            print(d_dim, width, kwargs)
            im = plot_process(phi, tau, d_dim, ax=ax, width=width, cmap=cmap, **kwargs)
            ax.axis("off")
        else:
            ax.plot(tau, phi)
            im = None
        if show_time:
            plt.colorbar(im, label="Time", location="right", fraction=0.05, pad=0.01, ax=ax, **kwargs)
    return im 


def get_name_list_function(base_to_test : list) -> str:
    l_name = [func.__name__.replace("power_", "x^") for func in base_to_test]
    return "$" + ", ".join(l_name) + "$"
    
def set_favorite_plot_config(font_scale=0.8, kwargs={}):
    ########## ASPECT #################
    plt.rcParams['figure.dpi'] = 2000
    sns.set_theme(context='paper', style='ticks', palette="colorblind", color_codes=True, font_scale=font_scale) #type: ignore
    #plt.rcParams.update(tex_fonts)
    ########## ASPECT #################
    plt.rcParams['text.usetex'] = True
    plt.rc('text.latex', preamble=r'\usepackage{physics,amsmath,amssymb, bm}')
    st_kwargs = {"xtick.major.pad": 0, "ytick.major.pad": 0, "xtick.minor.pad": 1, "ytick.minor.pad": 1,
                 "axes.labelpad" : 1}
    for key, value in st_kwargs.items():
        plt.rcParams[key] = value
    for key, value in kwargs.items():
        plt.rcParams[key] = value


width = "article"
width_pt = 430.00462
small_font = 4
big_font = 5
tex_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "arial",
    # Use 10pt font in plots, to match 10pt font in document
    "axes.labelsize": big_font,
    "font.size": big_font,
    # Make the legend/label fonts a little smaller
    "legend.fontsize": small_font,
    "xtick.labelsize": small_font,
    "ytick.labelsize": small_font,
    # Set the font used for Math in text and labels.
    "mathtext.fontset": "cm",  # Use the Computer Modern font
    # Ensure that all figure text matches the document text.
    "axes.titlesize": big_font
}

l_AF = [0, 3, 2, 8, 1, 4, 5, 6, 7, 9] ## May 10 colora in colorblind palette
cmaps = []

cmaps_sns = list(sns.color_palette("colorblind")) # type: ignore
for i in l_AF:
    cmaps.append(cmaps_sns[i])

colorblind_friendly_palette_rgb = [
    (230, 184, 0),    # Pastis yellow (original color)
    (30, 136, 229),   # Blue
    (255, 160, 0),    # Orange
    (0, 77, 64),      # Dark teal
    (216, 27, 96),    # Pink/Magenta
    (94, 53, 177),    # Purple
    (0, 172, 193),    # Cyan
    (85, 139, 47),    # Green
    (109, 76, 65),    # Brown
    (240, 98, 146)    # Light pink
]

# Normalize RGB values to 0-1 range
cmaps_plot = [(r/255, g/255, b/255) for r, g, b in colorblind_friendly_palette_rgb]

if __name__ == "__main__":
    plt.figure(figsize=(10, 1))
    for i, color in enumerate(cmaps):
        plt.axvspan(i, i+1, color=color)
        plt.text(i + 0.5, 0.5, str(i), ha='center', va='center')
    plt.xlim(0, len(cmaps))
    plt.axis('off')
    plt.show()