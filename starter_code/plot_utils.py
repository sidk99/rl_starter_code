import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def plot_3d_totem_scatter(data, labels, figname):
    """
        data: {(x, y): [z]}
            where there are a bunch of zs for the same (x,y) pair
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from matplotlib.ticker import MaxNLocator
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    min_z = min(min(d) for d in data.values())

    for (x, y), zs in sorted(data.items()):
        # draw points
        xs = [x for i in zs]  # repeat
        ys = [y for i in zs]  # repeat
        ax.scatter(xs, ys, zs)

        # draw poles
        for xx, yy, zz in zip(xs, ys, zs):
            ax.plot([xx, xx], [yy,yy], [min_z, zz], '-', color='black', linewidth=0.1)

    ax.set_xlabel(labels['x'])
    ax.set_ylabel(labels['y'])
    ax.set_zlabel(labels['z'])

    # assuming we are using int data
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_zlim(bottom=min_z)

    plt.savefig(figname)