# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Plotting convenience functions
from __future__ import print_function, division
from .phate import PHATE
import warnings
import scprep

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
    if ndim is not None and out[0].shape[0] < ndim:
        if isinstance(data, PHATE):
            data.set_params(n_components=ndim)
            out = data.transform()
        else:
            raise ValueError(
                "Expected at least {}-dimensional data, got {}".format(
                    ndim, out[0].shape[0]))
    return out


def scatter(x, y, z=None,
            c=None, cmap=None, s=None, discrete=None,
            ax=None, legend=None, figsize=None,
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
    x : list-like
        data for x axis
    y : list-like
        data for y axis
    z : list-like, optional (default: None)
        data for z axis
    c : list-like or None, optional (default: None)
        Color vector. Can be a single color value (RGB, RGBA, or named
        matplotlib colors), an array of these of length n_samples, or a list of
        discrete or continuous values of any data type. If `c` is not a single
        or list of matplotlib colors, the values in `c` will be used to
        populate the legend / colorbar with colors from `cmap`
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

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

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
    warnings.warn("`phate.plot.scatter` is deprecated. "
                  "Use `scprep.plot.scatter` instead.",
                  FutureWarning)
    return scprep.plot.scatter(x=x, y=y, z=z,
                               c=c, cmap=cmap, s=s, discrete=discrete,
                               ax=ax, legend=legend, figsize=figsize,
                               xticks=xticks,
                               yticks=yticks,
                               zticks=zticks,
                               xticklabels=xticklabels,
                               yticklabels=yticklabels,
                               zticklabels=zticklabels,
                               label_prefix=label_prefix,
                               xlabel=xlabel,
                               ylabel=ylabel,
                               zlabel=zlabel,
                               title=title,
                               legend_title=legend_title,
                               legend_loc=legend_loc,
                               filename=filename,
                               dpi=dpi,
                               **plot_kwargs)


def scatter2d(data, **kwargs):
    """Create a 2D scatter plot

    Builds upon `matplotlib.pyplot.scatter` with nice defaults
    and handles categorical colors / legends better.

    Parameters
    ----------
    data : array-like, shape=[n_samples, n_features]
        Input data. Only the first two components will be used.
    c : list-like or None, optional (default: None)
        Color vector. Can be a single color value (RGB, RGBA, or named
        matplotlib colors), an array of these of length n_samples, or a list of
        discrete or continuous values of any data type. If `c` is not a single
        or list of matplotlib colors, the values in `c` will be used to
        populate the legend / colorbar with colors from `cmap`
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

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

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
    warnings.warn("`phate.plot.scatter2d` is deprecated. "
                  "Use `scprep.plot.scatter2d` instead.",
                  FutureWarning)
    data = _get_plot_data(data, ndim=2)
    return scprep.plot.scatter2d(data, **kwargs)


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
        Color vector. Can be a single color value (RGB, RGBA, or named
        matplotlib colors), an array of these of length n_samples, or a list of
        discrete or continuous values of any data type. If `c` is not a single
        or list of matplotlib colors, the values in `c` will be used to
        populate the legend / colorbar with colors from `cmap`
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

    Returns
    -------
    ax : `matplotlib.Axes`
        axis on which plot was drawn

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
    warnings.warn("`phate.plot.scatter3d` is deprecated. "
                  "Use `scprep.plot.scatter3d` instead.",
                  FutureWarning)
    data = _get_plot_data(data, ndim=3)
    return scprep.plot.scatter3d(data, **kwargs)


def rotate_scatter3d(data,
                     filename=None,
                     elev=30,
                     rotation_speed=30,
                     fps=10,
                     ax=None,
                     figsize=None,
                     dpi=None,
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
    dpi : number, optional (default: None)
        Controls the dots per inch for the movie frames. This combined with
        the figure's size in inches controls the size of the movie.
        If None, defaults to rcParams["savefig.dpi"]
    ipython_html : {'html5', 'jshtml'}
        which html writer to use if using a Jupyter Notebook
    **kwargs : keyword arguments
        See :~func:`phate.plot.scatter3d`.

    Returns
    -------
    ani : `matplotlib.animation.FuncAnimation`
        animation object

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
    warnings.warn("`phate.plot.rotate_scatter3d` is deprecated. "
                  "Use `scprep.plot.rotate_scatter3d` instead.",
                  FutureWarning)
    return scprep.plot.rotate_scatter3d(data,
                                        filename=filename,
                                        elev=elev,
                                        rotation_speed=rotation_speed,
                                        fps=fps,
                                        ax=ax,
                                        figsize=figsize,
                                        dpi=dpi,
                                        ipython_html=ipython_html,
                                        **kwargs)
