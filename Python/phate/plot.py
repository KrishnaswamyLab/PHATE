# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

# Plotting convenience functions
from __future__ import print_function, division
from .phate import PHATE

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
                    out = data.obsm["X_phate"]
                except KeyError:
                    raise RuntimeError(
                        "data.obsm['X_phate'] not found. "
                        "Please run `sc.tl.phate(adata)` before plotting."
                    )
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
                    ndim, out[0].shape[0]
                )
            )
    return out