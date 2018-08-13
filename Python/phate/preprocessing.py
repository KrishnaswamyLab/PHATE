# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
from sklearn.preprocessing import normalize
import numpy as np
from scipy import sparse
import warnings

try:
    import pandas as pd
except ImportError:
    pass


def library_size_normalize(data, verbose=False):
    """Performs L1 normalization on input data
    Performs L1 normalization on input data such that the sum of expression
    values for each cell sums to 1
    then returns normalized matrix to the metric space using median UMI count
    per cell effectively scaling all cells as if they were sampled evenly.

    Parameters
    ----------
    data : ndarray [n,p]
        2 dimensional input data array with n cells and p dimensions

    Returns
    -------
    data_norm : ndarray [n, p]
        2 dimensional array with normalized gene expression values
    """
    warnings.warn("phate.preprocessing is deprecated. "
                  "Please use scprep.normalize instead. "
                  "Read more at http://scprep.readthedocs.io",
                  FutureWarning)
    if verbose:
        print("Normalizing library sizes for %s cells" % (data.shape[0]))

    # pandas support
    columns, index = None, None
    try:
        if isinstance(data, pd.SparseDataFrame) or \
                pd.api.types.is_sparse(data):
            columns, index = data.columns, data.index
            data = data.to_coo()
        elif isinstance(data, pd.DataFrame):
            columns, index = data.columns, data.index
    except NameError as e:
        if not str(e) == "name 'pd' is not defined":
            raise
        else:
            pass
    except AttributeError as e:
        warnings.warn("{}: is pandas out of date? ({})".format(
            str(e), pd.__version__), ImportWarning)

    median_transcript_count = np.median(np.array(data.sum(axis=1)))
    if sparse.issparse(data) and data.nnz >= 2**31:
        # check we can access elements by index
        try:
            data[0, 0]
        except TypeError:
            data = sparse.csr_matrix(data)
        # normalize in chunks - sklearn doesn't does with more
        # than 2**31 non-zero elements
        #
        # determine maximum chunk size
        split = 2**30 // (data.nnz // data.shape[0])
        size_ok = False
        while not size_ok:
            for i in range(0, data.shape[0], split):
                if data[i:i + split, :].nnz >= 2**31:
                    split = split // 2
                    break
            size_ok = True
        # normalize
        data_norm = []
        for i in range(0, data.shape[0], split):
            data_norm.append(normalize(data[i:i + split, :], 'l1', axis=1))
        # combine chunks
        data_norm = sparse.vstack(data_norm)
    else:
        data_norm = normalize(data, norm='l1', axis=1)

    # norm = 'l1' computes the L1 norm which computes the
    # axis = 1 independently normalizes each sample

    data_norm = data_norm * median_transcript_count
    if columns is not None:
        # pandas dataframe
        if sparse.issparse(data_norm):
            data_norm = pd.SparseDataFrame(data_norm, default_fill_value=0)
        else:
            data_norm = pd.DataFrame(data_norm)
        data_norm.columns = columns
        data_norm.index = index
    return data_norm
