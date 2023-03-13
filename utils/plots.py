import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.cm as cm

def plot_stroke(stroke, save_name=None, return_fig=False):
    # Plot a single example.
    f, ax = plt.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        if return_fig is False:
            plt.show()
        else:
            return f
    else:
        try:
            plt.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print ("Error building image!: ", save_name)

    plt.close()

def color_map_color(colors, cmap_name='viridis', vmin=0, vmax=1):
    # norm = plt.Normalize(vmin, vmax)
    norm = matplotlib.colors.Normalize(vmin=min(colors), vmax=max(colors))
    cmap = cm.get_cmap(cmap_name)  # PiYG
    rgb = cmap(norm(colors))[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return [matplotlib.colors.rgb2hex(c) for c in rgb]

def plot_stroke_attentions(stroke, attn_weights, save_name=None, return_fig=False):
    # Plot a single example.
    f, ax = plt.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    colors = np.zeros(x.shape[0])
    colors[-attn_weights.shape[0]:] = attn_weights
    colors = color_map_color(colors)

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        # ax.scatter(x[start:cut_value], y[start:cut_value], c=colors[start:cut_value])
        for line in range(start, cut_value-1):
            ax.plot(x[line:line+2], y[line:line+2], color=colors[line], marker='')
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        if return_fig is False:
            plt.show()
        else:
            return f
    else:
        try:
            plt.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print ("Error building image!: ", save_name)

    plt.close()
