# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Plotting convenience functions
from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # NOQA: F401
import pandas as pd
from .phate import PHATE

try:
    import anndata
except ImportError:
    # anndata not installed
    pass


def _get_plot_data(data):
    if isinstance(data, PHATE):
        data = data.transform()
    else:
        try:
            if isinstance(data, anndata.AnnData):
                try:
                    data = data.obsm['X_phate']
                except KeyError:
                    raise RuntimeError(
                        "data.obsm['X_phate'] not found. "
                        "Please run `sc.tl.phate(adata)` before plotting.")
        except NameError:
            # anndata not installed
            pass
    return data


def _auto_params(data, c, discrete, cmap):
    if discrete is None:
        # guess
        discrete = len(np.unique(c)) <= 20
    if discrete:
        c, labels = pd.factorize(c)
        if cmap is None and len(np.unique(c)) <= 10:
            c = mpl.cm.tab10(np.linspace(0, 1, 10))[c]
            cmap = None
        else:
            cmap = 'tab20'
    else:
        labels = None
        if cmap is None:
            cmap = 'viridis'
    if len(data) == 3:
        subplot_kw = {'projection': '3d'}
    else:
        subplot_kw = {}
    return c, labels, discrete, cmap, subplot_kw


def _scatter(*data, c=None, cmap=None, s=1, discrete=None,
             ax=None, legend=True,
             **plot_kwargs):
    c, labels, discrete, cmap, subplot_kw = _auto_params(data, c, discrete,
                                                         cmap)
    plot_idx = np.random.permutation(data[0].shape[0])
    if ax is None:
        fig, ax = plt.subplots(subplot_kw=subplot_kw)
        show = True
    else:
        show = False
    if legend and not discrete:
        im = ax.imshow(np.arange(10).reshape(-1, 1),
                       vmin=np.min(c), vmax=np.max(c), cmap=cmap)
        ax.clear()
    sc = ax.scatter(*[d[plot_idx] for d in data],
                    c=c[plot_idx], cmap=cmap, s=s, **plot_kwargs)
    if legend:
        if discrete:
            def handle(c):
                return plt.Line2D([], [], color=c, ls="", marker="o")
            plt.legend(
                handles=[handle(sc.cmap(sc.norm(i)))
                         for i in range(len(labels))],
                labels=list(labels),
                ncol=max(1, len(labels) // 10))
        else:
            plt.colorbar(im)
    if show:
        plt.tight_layout()
        plt.show()


def scatter2d(data, c=None, cmap=None, s=1, discrete=None,
              ax=None, legend=True, **plot_kwargs):
    data = _get_plot_data(data)
    _scatter(data[:, 0], data[:, 1],
             c=c, cmap=cmap, s=s, discrete=discrete,
             ax=ax, legend=legend)


def scatter3d(data, c=None, cmap=None, s=1, discrete=None,
              ax=None, legend=True, **plot_kwargs):
    data = _get_plot_data(data)
    _scatter(data[:, 0], data[:, 1], data[:, 2],
             c=c, cmap=cmap, s=s, discrete=discrete,
             ax=ax, legend=legend)
