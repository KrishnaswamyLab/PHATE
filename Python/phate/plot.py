# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Plotting convenience functions
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import animation, rc
from mpl_toolkits.mplot3d import Axes3D  # NOQA: F401
import pandas as pd
import numbers
from .phate import PHATE
from .utils import in_ipynb

try:
    import anndata
except ImportError:
    # anndata not installed
    pass


def _get_plot_data(data, ndim=None):
    """Get plot data out of an input object

    Parameters
    ----------
    data : array-like, `phate.PHATE` or `scanpy.AnnData`
    ndim : int, optional (default: None)
        Minimum number of dimensions
    """
    out = data
    if isinstance(data, PHATE):
        out = data.transform()
    else:
        try:
            if isinstance(data, anndata.AnnData):
                try:
                    out = data.obsm['X_phate']
                except KeyError:
                    raise RuntimeError(
                        "data.obsm['X_phate'] not found. "
                        "Please run `sc.tl.phate(adata)` before plotting.")
        except NameError:
            # anndata not installed
            pass
    if ndim is not None and out.shape[1] < ndim:
        if isinstance(data, PHATE):
            data.set_params(n_components=ndim)
            out = data.transform()
        else:
            raise ValueError(
                "Expected at least {}-dimensional data, got {}".format(
                    ndim, out.shape[1]))
    return out


def _auto_params(data, c, discrete, cmap, legend):
    """Automatically select nice parameters for a scatter plot
    """
    for d in data[1:]:
        if d.shape[0] != data[0].shape[0]:
            raise ValueError("Expected all axis of data to have the same length"
                             ". Got {}".format([d.shape[0] for d in data]))
    if c is not None and not mpl.colors.is_color_like(c):
        try:
            c = c.values
        except AttributeError:
            # not a pandas Series
            pass
        try:
            c = c.toarray()
        except AttributeError:
            # not a scipy spmatrix
            pass
        c = np.array(c).flatten()
        if not len(c) == data[0].shape[0]:
            raise ValueError("Expected c of length {} or 1. Got {}".format(
                len(c), data.shape[0]))
        if discrete is None:
            # guess
            if isinstance(cmap, dict) or \
                    not np.all([isinstance(x, numbers.Number) for x in c]):
                discrete = True
            else:
                discrete = len(np.unique(c)) <= 20
        if discrete:
            c, labels = pd.factorize(c)
            if cmap is None and len(np.unique(c)) <= 10:
                cmap = mpl.colors.ListedColormap(
                    mpl.cm.tab10.colors[:len(np.unique(c))])
            elif cmap is None:
                cmap = 'tab20'
        else:
            if not np.all([isinstance(x, numbers.Number) for x in c]):
                raise ValueError(
                    "Cannot treat non-numeric data as continuous.")
            labels = None
            if cmap is None:
                cmap = 'inferno'
        if isinstance(cmap, dict):
            if not discrete:
                raise ValueError("Cannot use dictionary cmap with "
                                 "continuous data.")
            elif np.any([l not in cmap for l in labels]):
                missing = set(labels).difference(cmap.keys())
                raise ValueError(
                    "Dictionary cmap requires a color "
                    "for every unique entry in `c`. "
                    "Missing colors for [{}]".format(
                        ", ".join([str(l) for l in missing])))
            else:
                cmap = mpl.colors.ListedColormap(
                    [mpl.colors.to_rgba(cmap[l]) for l in labels])
    else:
        labels = None
        legend = False
    if len(data) == 3:
        subplot_kw = {'projection': '3d'}
    elif len(data) == 2:
        subplot_kw = {}
    else:
        raise ValueError("Expected either 2 or 3 dimensional data. "
                         "Got {}".format(len(data)))
    return c, labels, discrete, cmap, subplot_kw, legend


def scatter(data,
            c=None, cmap=None, s=1, discrete=None,
            ax=None, legend=True, figsize=None,
            xticks=False,
            yticks=False,
            zticks=False,
            xticklabels=True,
            yticklabels=True,
            zticklabels=True,
            label_prefix="PHATE",
            xlabel=None,
            ylabel=None,
            zlabel=None,
            title=None,
            legend_title="",
            legend_loc='best',
            filename=None,
            dpi=None,
            **plot_kwargs):
    """Create a scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better. For easy access, use
    `scatter2d` or `scatter3d`.

    Parameters
    ----------
    data : list of array-like
        list containing one array for each axis (min: 2, max: 3)
    c : list-like or None, optional (default: None)
        Color vector. Can be an array of RGBA values, or a list of discrete or
        continuous values of any data type. The values in `c` will be used to
        populate the legend / colorbar
    cmap : `matplotlib` colormap, str, dict or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    s : float, optional (default: 1)
        Point size.
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: True)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar.
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    xticks : True, False, or list-like (default: False)
        If True, keeps default x ticks. If False, removes x ticks.
        If a list, sets custom x ticks
    yticks : True, False, or list-like (default: False)
        If True, keeps default y ticks. If False, removes y ticks.
        If a list, sets custom y ticks
    zticks : True, False, or list-like (default: False)
        If True, keeps default z ticks. If False, removes z ticks.
        If a list, sets custom z ticks.  Only used for 3D plots.
    xticklabels : True, False, or list-like (default: True)
        If True, keeps default x tick labels. If False, removes x tick labels.
        If a list, sets custom x tick labels
    yticklabels : True, False, or list-like (default: True)
        If True, keeps default y tick labels. If False, removes y tick labels.
        If a list, sets custom y tick labels
    zticklabels : True, False, or list-like (default: True)
        If True, keeps default z tick labels. If False, removes z tick labels.
        If a list, sets custom z tick labels. Only used for 3D plots.
    label_prefix : str or None (default: "PHATE")
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    xlabel : str or None (default : None)
        Label for the x axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    ylabel : str or None (default : None)
        Label for the y axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    zlabel : str or None (default : None)
        Label for the z axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
        Only used for 3D plots.
    title : str or None (default: None)
        axis title. If None, no title is set.
    legend_title : str (default: "")
        title for the colorbar of legend. Only used for discrete data.
    legend_loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location. Only used for discrete data.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **plot_kwargs : keyword arguments
        Extra arguments passed to `matplotlib.pyplot.scatter`.

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> ###
    >>> # Running PHATE
    >>> ###
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> ###
    >>> # Plotting using phate.plot
    >>> ###
    >>> phate.plot.scatter2d(tree_phate, c=tree_clusters)
    >>> # You can also pass the PHATE operator instead of data
    >>> phate.plot.scatter2d(phate_operator, c=tree_clusters)
    >>> phate.plot.scatter3d(phate_operator, c=tree_clusters)
    >>> ###
    >>> # Using a cmap dictionary
    >>> ###
    >>> import numpy as np
    >>> X = np.random.normal(0,1,[1000,2])
    >>> c = np.random.choice(['a','b'], 1000, replace=True)
    >>> X[c=='a'] += 10
    >>> phate.plot.scatter2d(X, c=c, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    c, labels, discrete, cmap, subplot_kw, legend = _auto_params(
        data, c, discrete,
        cmap, legend)

    plot_idx = np.random.permutation(data[0].shape[0])
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=subplot_kw)
        show = True
    else:
        fig = ax.get_figure()
        show = False
    if legend and not discrete:
        im = ax.imshow(np.linspace(np.min(data[1]), np.max(data[1]), 10).reshape(-1, 1),
                       vmin=np.min(c), vmax=np.max(c), cmap=cmap,
                       aspect='auto', origin='lower')
        im.remove()
        ax.relim()
        ax.autoscale()
    try:
        if c is not None and not mpl.colors.is_color_like(c):
            c = c[plot_idx]
        sc = ax.scatter(*[d[plot_idx] for d in data],
                        c=c,
                        cmap=cmap, s=s, **plot_kwargs)
    except TypeError as e:
        if not hasattr(ax, "get_zlim"):
            raise TypeError("Expected ax with projection='3d'. "
                            "Got 2D axis instead.")
        else:
            raise e

    if label_prefix is not None:
        if xlabel is None:
            xlabel = label_prefix + "1"
        if ylabel is None:
            ylabel = label_prefix + "2"
        if zlabel is None:
            zlabel = label_prefix + "3"

    if not xticks:
        ax.set_xticks([])
    elif xticks is True:
        pass
    else:
        ax.set_xticks(xticks)
    if not xticklabels:
        ax.set_xticklabels([])
    elif xticklabels is True:
        pass
    else:
        ax.set_xticklabels(xticklabels)
    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if not yticks:
        ax.set_yticks([])
    elif yticks is True:
        pass
    else:
        ax.set_yticks(yticks)
    if not yticklabels:
        ax.set_yticklabels([])
    elif yticklabels is True:
        pass
    else:
        ax.set_yticklabels(yticklabels)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if title is not None:
        ax.set_title(title)

    if len(data) == 3:
        if not zticks:
            ax.set_zticks([])
        elif zticks is True:
            pass
        else:
            ax.set_zticks(zticks)
        if not zticklabels:
            ax.set_zticklabels([])
        elif zticklabels is True:
            pass
        else:
            ax.set_zticklabels(zticklabels)
        if zlabel is not None:
            ax.set_zlabel(zlabel)

    if legend:
        if discrete:
            def handle(c):
                return plt.Line2D([], [], color=c, ls="", marker="o")
            ax.legend(
                handles=[handle(sc.cmap(sc.norm(i)))
                         for i in range(len(labels))],
                labels=list(labels),
                ncol=max(1, len(labels) // 10),
                title=legend_title,
                loc=legend_loc)
        else:
            plt.colorbar(im, label=legend_title)

    if show or filename is not None:
        plt.tight_layout()
    if filename is not None:
        fig.savefig(filename, dpi=dpi)
    if show:
        if not in_ipynb():
            plt.show(block=False)


def scatter2d(data, **kwargs):
    """Create a 2D scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Only the first two components will be used.
    c : list-like or None, optional (default: None)
        Color vector. Can be an array of RGBA values, or a list of discrete or
        continuous values of any data type. The values in `c` will be used to
        populate the legend / colorbar
    cmap : `matplotlib` colormap, str, dict or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    s : float, optional (default: 1)
        Point size.
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: True)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar.
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    xticks : True, False, or list-like (default: False)
        If True, keeps default x ticks. If False, removes x ticks.
        If a list, sets custom x ticks
    yticks : True, False, or list-like (default: False)
        If True, keeps default y ticks. If False, removes y ticks.
        If a list, sets custom y ticks
    zticks : True, False, or list-like (default: False)
        If True, keeps default z ticks. If False, removes z ticks.
        If a list, sets custom z ticks.  Only used for 3D plots.
    xticklabels : True, False, or list-like (default: True)
        If True, keeps default x tick labels. If False, removes x tick labels.
        If a list, sets custom x tick labels
    yticklabels : True, False, or list-like (default: True)
        If True, keeps default y tick labels. If False, removes y tick labels.
        If a list, sets custom y tick labels
    zticklabels : True, False, or list-like (default: True)
        If True, keeps default z tick labels. If False, removes z tick labels.
        If a list, sets custom z tick labels. Only used for 3D plots.
    label_prefix : str or None (default: "PHATE")
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    xlabel : str or None (default : None)
        Label for the x axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    ylabel : str or None (default : None)
        Label for the y axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    zlabel : str or None (default : None)
        Label for the z axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
        Only used for 3D plots.
    title : str or None (default: None)
        axis title. If None, no title is set.
    legend_title : str (default: "")
        title for the colorbar of legend
    legend_loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location. Only used for discrete data.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **plot_kwargs : keyword arguments
        Extra arguments passed to `matplotlib.pyplot.scatter`.

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> ###
    >>> # Running PHATE
    >>> ###
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> ###
    >>> # Plotting using phate.plot
    >>> ###
    >>> phate.plot.scatter2d(tree_phate, c=tree_clusters)
    >>> # You can also pass the PHATE operator instead of data
    >>> phate.plot.scatter2d(phate_operator, c=tree_clusters)
    >>> phate.plot.scatter3d(phate_operator, c=tree_clusters)
    >>> ###
    >>> # Using a cmap dictionary
    >>> ###
    >>> import numpy as np
    >>> X = np.random.normal(0,1,[1000,2])
    >>> c = np.random.choice(['a','b'], 1000, replace=True)
    >>> X[c=='a'] += 10
    >>> phate.plot.scatter2d(X, c=c, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    data = _get_plot_data(data, ndim=2)
    return scatter([data[:, 0], data[:, 1]], **kwargs)


def scatter3d(data, **kwargs):
    """Create a 3D scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        to be the value of the figure. Only used if filename is not None.
        Input data. Only the first three components will be used.
    c : list-like or None, optional (default: None)
        Color vector. Can be an array of RGBA values, or a list of discrete or
        continuous values of any data type. The values in `c` will be used to
        populate the legend / colorbar
    cmap : `matplotlib` colormap, str, dict or None, optional (default: None)
        matplotlib colormap. If None, uses `tab20` for discrete data and
        `inferno` for continuous data. If a dictionary, expects one key
        for every unique value in `c`, where values are valid matplotlib colors
        (hsv, rbg, rgba, or named colors)
    s : float, optional (default: 1)
        Point size.
    discrete : bool or None, optional (default: None)
        If True, the legend is categorical. If False, the legend is a colorbar.
        If None, discreteness is detected automatically. Data containing
        non-numeric `c` is always discrete, and numeric data with 20 or less
        unique values is discrete.
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    legend : bool, optional (default: True)
        States whether or not to create a legend. If data is continuous,
        the legend is a colorbar.
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    xticks : True, False, or list-like (default: False)
        If True, keeps default x ticks. If False, removes x ticks.
        If a list, sets custom x ticks
    yticks : True, False, or list-like (default: False)
        If True, keeps default y ticks. If False, removes y ticks.
        If a list, sets custom y ticks
    zticks : True, False, or list-like (default: False)
        If True, keeps default z ticks. If False, removes z ticks.
        If a list, sets custom z ticks.  Only used for 3D plots.
    xticklabels : True, False, or list-like (default: True)
        If True, keeps default x tick labels. If False, removes x tick labels.
        If a list, sets custom x tick labels
    yticklabels : True, False, or list-like (default: True)
        If True, keeps default y tick labels. If False, removes y tick labels.
        If a list, sets custom y tick labels
    zticklabels : True, False, or list-like (default: True)
        If True, keeps default z tick labels. If False, removes z tick labels.
        If a list, sets custom z tick labels. Only used for 3D plots.
    label_prefix : str or None (default: "PHATE")
        Prefix for all axis labels. Axes will be labelled `label_prefix`1,
        `label_prefix`2, etc. Can be overriden by setting `xlabel`,
        `ylabel`, and `zlabel`.
    xlabel : str or None (default : None)
        Label for the x axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    ylabel : str or None (default : None)
        Label for the y axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
    zlabel : str or None (default : None)
        Label for the z axis. Overrides the automatic label given by
        label_prefix. If None and label_prefix is None, no label is set.
        Only used for 3D plots.
    title : str or None (default: None)
        axis title. If None, no title is set.
    legend_title : str (default: "")
        title for the colorbar of legend
    legend_loc : int or string or pair of floats, default: 'best'
        Matplotlib legend location. Only used for discrete data.
        See <https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html>
        for details.
    filename : str or None (default: None)
        file to which the output is saved
    dpi : int or None, optional (default: None)
        The resolution in dots per inch. If None it will default to the value
        savefig.dpi in the matplotlibrc file. If 'figure' it will set the dpi
        to be the value of the figure. Only used if filename is not None.
    **plot_kwargs : keyword arguments
        Extra arguments passed to `matplotlib.pyplot.scatter`.

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> ###
    >>> # Running PHATE
    >>> ###
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> ###
    >>> # Plotting using phate.plot
    >>> ###
    >>> phate.plot.scatter2d(tree_phate, c=tree_clusters)
    >>> # You can also pass the PHATE operator instead of data
    >>> phate.plot.scatter2d(phate_operator, c=tree_clusters)
    >>> phate.plot.scatter3d(phate_operator, c=tree_clusters)
    >>> ###
    >>> # Using a cmap dictionary
    >>> ###
    >>> import numpy as np
    >>> X = np.random.normal(0,1,[1000,2])
    >>> c = np.random.choice(['a','b'], 1000, replace=True)
    >>> X[c=='a'] += 10
    >>> phate.plot.scatter2d(X, c=c, cmap={'a' : [1,0,0,1], 'b' : 'xkcd:sky blue'})
    """
    data = _get_plot_data(data, ndim=3)
    return scatter([data[:, 0], data[:, 1], data[:, 2]],
                   **kwargs)


def rotate_scatter3d(data,
                     filename=None,
                     elev=30,
                     rotation_speed=30,
                     fps=10,
                     ax=None,
                     figsize=None,
                     ipython_html="jshtml",
                     **kwargs):
    """Create a rotating 3D scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, `phate.PHATE` or `scanpy.AnnData`
        Input data. Only the first three dimensions are used.
    filename : str, optional (default: None)
        If not None, saves a .gif or .mp4 with the output
    elev : float, optional (default: 30)
        Elevation of viewpoint from horizontal, in degrees
    rotation_speed : float, optional (default: 30)
        Speed of axis rotation, in degrees per second
    fps : int, optional (default: 10)
        Frames per second. Increase this for a smoother animation
    ax : `matplotlib.Axes` or None, optional (default: None)
        axis on which to plot. If None, an axis is created
    figsize : tuple, optional (default: None)
        Tuple of floats for creation of new `matplotlib` figure. Only used if
        `ax` is None.
    ipython_html : {'html5', 'jshtml'}
        which html writer to use if using a Jupyter Notebook
    **kwargs : keyword arguments
        See :~func:`phate.plot.scatter3d`.

    Examples
    --------
    >>> import phate
    >>> import matplotlib.pyplot as plt
    >>> tree_data, tree_clusters = phate.tree.gen_dla(n_dim=100, n_branch=20,
    ...                                               branch_length=100)
    >>> tree_data.shape
    (2000, 100)
    >>> phate_operator = phate.PHATE(n_components=3, k=5, a=20, t=150)
    >>> tree_phate = phate_operator.fit_transform(tree_data)
    >>> tree_phate.shape
    (2000, 2)
    >>> phate.plot.rotate_scatter3d(tree_phate, c=tree_clusters)
    """
    if in_ipynb():
        # credit to
        # http://tiao.io/posts/notebooks/save-matplotlib-animations-as-gifs/
        rc('animation', html=ipython_html)

    if filename is not None:
        if filename.endswith(".gif"):
            writer = 'imagemagick'
        elif filename.endswith(".mp4"):
            writer = "ffmpeg"
        else:
            raise ValueError(
                "filename must end in .gif or .mp4. Got {}".format(filename))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize,
                               subplot_kw={'projection': '3d'})
        show = True
    else:
        fig = ax.get_figure()
        show = False

    degrees_per_frame = rotation_speed / fps
    frames = int(round(360 / degrees_per_frame))
    # fix rounding errors
    degrees_per_frame = 360 / frames
    interval = 1000 * degrees_per_frame / rotation_speed

    scatter3d(data, ax=ax, **kwargs)

    def init():
        ax.view_init(azim=0, elev=elev)
        return ax

    def animate(i):
        ax.view_init(azim=i * degrees_per_frame, elev=elev)
        return ax

    ani = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=range(frames), interval=interval, blit=False)

    if filename is not None:
        ani.save(filename, writer=writer)

    if in_ipynb():
        # credit to https://stackoverflow.com/a/45573903/3996580
        plt.close()
    elif show:
        plt.tight_layout()
        plt.show(block=False)

    return ani
