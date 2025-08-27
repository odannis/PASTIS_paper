from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.pyplot as plt
import numpy as np

class VideoPhi:
    def __init__(self, phi, l_ax, three_D, min_end, name_plot, global_vmin=True,
                 cmap = 'seismic', use_colorbar=True, vmax=None, vmin=None):
        self.l_ax = l_ax
        self.three_D = three_D
        self.X, self.Y = np.meshgrid(np.arange(phi.shape[1]), np.arange(phi.shape[2]))
        self.min_end = min_end
        self._phi = phi
        self.l_shw = {}
        self.colorbar = {}
        self.name_plot = name_plot
        self.cmap = cmap
        self.use_colorbar = use_colorbar 
        # Precompute vmin and vmax for each image
        self.vmin = {}
        self.vmax = {}
        for i_image in range(len(self.l_ax)):
            
            if len(self.l_ax) == 1 or global_vmin:
                movie = self._phi
            else:
                movie = self._phi[:, i_image]
            if self.min_end is not None:
                if self.min_end is True:
                    stop = -1
                else:
                    stop = None    
                _vmin = np.min(movie[stop])
                _vmax = np.max(movie[stop])
            else:
                _vmin = np.min(movie[i_image])
                _vmax = np.max(movie[i_image])
            self.vmin[i_image] = vmin if vmin is not None else _vmin
            self.vmax[i_image] = vmax if vmax is not None else _vmax

    def __call__(self, i):
        artists = []
        for i_image, ax in enumerate(self.l_ax):
            if len(self.l_ax) == 1:
                movie = self._phi
            else:
                movie = self._phi[:, i_image]

            image = movie[i]

            if i == 0:
                if self.three_D:
                    shw = ax.plot_wireframe(self.X, self.Y, image)
                    ax.set_zlim(top=self.max)
                    ax.clear()
                else:
                    shw = ax.imshow(image, vmin=self.vmin[i_image], vmax=self.vmax[i_image],
                                    cmap=self.cmap, interpolation='nearest')

                ax.axis('off')
                self.l_shw[i_image] = shw
                if not self.three_D and self.use_colorbar:
                    if i_image not in self.colorbar.keys():
                        self.colorbar[i_image] = plt.colorbar(shw, ax=ax) 
                if len(self.name_plot) == len(self.l_ax):
                    ax.set_title(self.name_plot[i_image])
                artists.append(shw)
            else:
                shw = self.l_shw[i_image]
                if self.three_D:
                    ax.clear()
                    shw = ax.plot_wireframe(self.X, self.Y, image)
                    ax.set_zlim(top=self.max)
                    artists.append(shw)
                else:
                    shw.set_data(image)
                    artists.append(shw)
        return artists

def show_video_phi(phi, number_images_video=50, fps=20, save_name=None, min_end=None, three_D=False, name_plot=[], figsize=None, ncols=None,
                   cmap='seismic', global_vmin=True, **kwargs):
    if ncols is not None:
        pass
    elif len(phi.shape) == 3:
        ncols = 1
    elif len(phi.shape) == 4:
        ncols = phi.shape[1]
    else:
        raise Exception("Phi has a strange shape") 
        
    if phi.shape[0] < number_images_video:
        number_images_video = phi.shape[0]
        print("Small data set --> Video of the whole data set")
        
    plt.set_cmap('seismic')
    if not three_D:
        fig, ax = plt.subplots(ncols=ncols, figsize=figsize)
        if ncols == 1:
            ax = [ax]
    else:
        fig = plt.figure(figsize=figsize)
        ax = plt.axes(projection='3d')
    # Use linspace to get indices for evenly spaced frames
    indices = np.linspace(0, phi.shape[0]-1, number_images_video, dtype=int)
    phi_plot = phi[indices]
    ud = VideoPhi(phi_plot, ax, three_D, min_end, name_plot, cmap=cmap, global_vmin=global_vmin, **kwargs)
    interval = int(1000 / fps)
    anim = FuncAnimation(
        fig, ud, frames=phi_plot.shape[0], interval=interval,
        repeat=False, cache_frame_data=False, blit=True
    )
    plt.tight_layout()
    if save_name is not None:
        anim.save(save_name + ".mp4", dpi=300)
        #plt.savefig(save_name + "_image.png", dpi=300)
    a = HTML(anim.to_html5_video())
    plt.clf()
    return a 
