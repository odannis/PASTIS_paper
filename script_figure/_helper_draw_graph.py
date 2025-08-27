import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import FancyArrowPatch

def matrix_special_color(matrix, real_matrix):
    rows, cols = np.where(real_matrix  != 0)
    matrix_out = np.zeros_like(matrix)
    for row, col in zip(rows, cols):
        if matrix[row, col] != 0:
            if np.sign(matrix[row, col]) == np.sign(real_matrix[row, col]):
                matrix_out[row, col] = 1
            else:
                print("here")
                matrix_out[row, col] = -0.5
        else:
            matrix_out[row, col] = 0.5
    rows, cols = np.where(matrix  != 0)
    for row, col in zip(rows, cols):
        if real_matrix[row, col] == 0:
            matrix_out[row, col] = -1
    return matrix_out

def draw_graph(matrix, ax : plt.axes = None, pos = None, color_special = False, special_lines = False, real_matrix = None,
                style_line = "-", alpha = 1, node_size=200, width_mult=3, margins=0.1, cmaps=None,
                shorten_edges=10.0, size_head=30, show_self_arrow=True, scale_graph=1, center_graph=None):
    
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(5, 5))
    
    matrix = matrix.copy()
    matrix = matrix.T
    if real_matrix is not None:
        real_matrix = real_matrix.copy()
        real_matrix = real_matrix.T
    if not show_self_arrow:
        matrix[range(matrix.shape[0]), range(matrix.shape[0])] = 0
        if real_matrix is not None:
            real_matrix[range(real_matrix.shape[0]), range(real_matrix.shape[0])] = 0
    G = nx.DiGraph()
    # Ensure all nodes are added, even if they have no edges
    G.add_nodes_from(range(matrix.shape[0]))


    norm_width = np.max(np.abs(matrix))
    if real_matrix is not None:
        norm_width = np.max(np.abs(real_matrix))
        G_real = nx.DiGraph()
        rows, cols = np.where(real_matrix != 0)
        weights = matrix[rows, cols]
        for row, col, weight in zip(rows, cols, weights):
            G_real.add_edge(col, row, weight=weight)
        pos = nx.circular_layout(G_real, scale=scale_graph, center=center_graph)


    # Step 4: Add edges to the graph with weights and colors
    rows, cols = np.where(matrix != 0)
    weights = matrix[rows, cols]
    colors = ["g" if weight < 0 else "r" if weight > 0 else "gray" for weight in weights]

    if color_special:
        out_matrix = matrix_special_color(matrix, real_matrix=real_matrix)
        print(out_matrix)
        rows, cols = np.where(out_matrix != 0)
        weights = out_matrix[rows, cols]
        colors = ["green" if weight == 1 else "r" if weight == -1 else "orange" if weight == -0.5 else "gray" for weight in weights]
        weights = real_matrix[rows, cols]
        weights = [w if w != 0 else matrix[rows[i], cols[i]] for i,w in enumerate(weights)]
    
    if special_lines:
        out_matrix = matrix_special_color(matrix, real_matrix=real_matrix)
        rows, cols = np.where(out_matrix != 0)
        weights = out_matrix[rows, cols]
        styles = [":" if (weight == 1 or weight == -0.5) else "--" if weight == -1 else "-." for weight in weights]
        colors = ["black" for weight in weights]
        weights = matrix[rows, cols]
        weights = [w if w != 0 else real_matrix[rows[i], cols[i]] for i,w in enumerate(weights)]


    styles = [style_line for weight in weights]
    alphas = [alpha for weight in weights]
    
    for row, col, weight, color in zip(rows, cols, weights, colors):
        G.add_edge(col, row, weight=weight, color=color)

    # Step 5: Draw the graph with edge colors
    if pos is None:
        pos = nx.circular_layout(G, scale=scale_graph, center=center_graph)  # Positions for all nodes in a circle

    edge_widths = width_mult#*np.abs(list(nx.get_edge_attributes(G,'weight').values()))/norm_width

    edges = G.edges(data=True)
    edge_colors = [edge[2]['color'] for edge in edges]

    # Obtain a colormap and generate a list of colors from it
    node_colors = None
    if cmaps is not None:
        node_colors = [cmaps[i] for i in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, ax=ax, node_size=node_size, margins=margins,)
    #nx.draw_networkx_labels(G, pos, font_size=font_size, font_family="sans-serif", ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, ax=ax, style=styles,
                             alpha=alphas, connectionstyle='arc3, rad = 0.2', node_size=node_size,
                             arrowstyle='-|>', min_source_margin=shorten_edges, min_target_margin=shorten_edges,
                             arrows=True, arrowsize=size_head)

    #draw_edges_with_gaps(G, pos, ax, shorten=shorten_edges, alpha=alpha, size_head=size_head, lw=lw_edge)
    ax.axis('off') 
    ax.axis('equal') 
    return pos

def get_matrix_interaction(gradient_perso):
    coeff = gradient_perso.d_helper[gradient_perso.best_nodes][1]
    d_dim = gradient_perso._sffi.phi.shape[-1]
    l_name_base = gradient_perso._sffi.l_name_base
    A_b = np.zeros((d_dim, d_dim))
    if gradient_perso.best_nodes != tuple():
        for i_base, name_func in enumerate(l_name_base[np.array(gradient_perso.best_nodes)]):
            on_dim = int(name_func.split("X_")[1].split(" ")[0])
            func = name_func.split("[")[1].split("]")[0].split(" ")
            print("for %s, on_dim = %d, func = %s" % (name_func, on_dim, func))
            if len(func) > 1 :
                func_1 = int(func[0].split("_")[1][:-1])
                func_2 = int(func[1].split("_")[1][:-1])
                if func_1 == on_dim:
                    A_b[on_dim, func_2] = -1*coeff[i_base]
                elif func_2 == on_dim:
                    A_b[on_dim, func_1] = -1*coeff[i_base]
                else:
                    print("Inferred function out of graph representation")
                    #raise ValueError("Not good")
    return A_b

# Function to add an image on a node
def add_image_on_node(ax, pos, image_path, zoom, folder_images="images_bacteria/"):
    image = plt.imread(folder_images + image_path)
    img = OffsetImage(image, zoom=zoom)
    ab = AnnotationBbox(img, (pos[0], pos[1]), frameon=False)
    ax.add_artist(ab)
    
def add_images_on_nodes(ax, pos, image_name_base="bac_1", zoom=0.1, folder_images="images_bacteria/"):
    c  = 0
    for (node, position) in pos.items():
        image_path = f"{image_name_base}_{c}.png"
        c += 1
        add_image_on_node(ax, position, image_path, zoom, folder_images)
        
def draw_edges_with_gaps(G, pos, ax, alpha=1.0, shorten=20, size_head=30.0, lw=3.0):
    edge_colors = [edge[2]['color'] for edge in G.edges(data=True)]
    for i, d in enumerate(G.edges()):
        start_node, end_node = d
        sp = np.array(pos[start_node])
        ep = np.array(pos[end_node])
        color = edge_colors[i]
        if start_node == end_node:  # Self-loop
            # Calculate loop displacement
            disp = np.array([0.1, 0.1])  # This is a simple way to get a displacement vector.
            loop_start = sp + disp
            loop_end = sp
            
            # Dynamically adjust the radius for the self-loop based on the graph layout scale
            loop_rad = np.linalg.norm(disp) * 30  # Radius based on displacement magnitude
            
            # Create a loop-like arrow
            loop = FancyArrowPatch(loop_start, loop_end, 
                         arrowstyle='o', 
                         mutation_scale=20,  # Size of the circle
                         color='red')
            ax.add_patch(loop)
        else:
            line = FancyArrowPatch(sp, ep, arrowstyle='->',
                                color=color, alpha=alpha, 
                                shrinkA=shorten, shrinkB=shorten,  # Adjust starting and ending points
                                mutation_scale=size_head, lw=lw, zorder=1,
                                connectionstyle="arc3,rad=0.1")
            ax.add_patch(line)


def create_image_with_all_color():
    from PIL import Image, ImageOps
    import matplotlib.pyplot as plt

    # Load the image
    image_path = 'script_figure/images_bacteria/bac_1.png'
    image = Image.open(image_path)

    # Define the colorblind palette from Seaborn
    colorblind_palette = [
        '#0173b2', # Blue
        '#de8f05', # Orange
        '#029e73', # Green
        '#d55e00', # Red
        '#cc78bc', # Purple
        '#ca9161', # Brown
        '#fbafe4', # Pink
        '#949494', # Gray
        '#ece133', # Yellow
        '#56b4e9'  # Light Blue
    ]

    # Function to change the color of the image
    def change_image_color(image, color):
        # Convert image to RGBA
        image = image.convert("RGBA")
        data = image.getdata()

        # Replace the color
        new_data = []
        for item in data:
            if item[3] > 0: # Only change non-transparent pixels
                new_data.append(tuple(int(color[i:i+2], 16) for i in (1, 3, 5)) + (item[3],))
            else:
                new_data.append(item)
        
        image.putdata(new_data)
        return image

    saved_images = {}
    for i, color in enumerate(colorblind_palette):
        colored_image = change_image_color(image, color)
        file_path = f'script_figure/images_bacteria/bac_1_{i}.png'
        colored_image.save(file_path, "PNG")
        saved_images[color] = file_path
        
    # #Create a figure to show all images
    # fig, axes = plt.subplots(2, 5, figsize=(20, 10))

    # # Apply each color to the image and display it
    # for ax, color in zip(axes.flatten(), colorblind_palette):
    #     colored_image = change_image_color(image, color)
    #     ax.imshow(colored_image)
    #     ax.axis('off')
    #     ax.set_title(color)

if __name__ == "__main__":
    create_image_with_all_color()
    A = np.array([[0, 1, 0, 1], [0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]])
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    pos = draw_graph(A, ax=ax, style_line="-", real_matrix=A, alpha=1,  color_special=True, node_size=0)
    add_images_on_nodes(ax, pos)
    plt.show()
