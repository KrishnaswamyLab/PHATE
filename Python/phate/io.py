# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import pandas as pd
import scipy.io as sio
import warnings
import numpy as np


def load_10X(data_dir, sparse=True, gene_labels='symbol'):
    """Basic IO for 10X data produced from the 10X Cellranger pipeline.

    A default run of the `cellranger count` command will generate gene-barcode
    matrices for secondary analysis. For both "raw" and "filtered" output,
    directories are created containing three files:
    'matrix.mtx', 'barcodes.tsv', 'genes.tsv'.
    Running `phate.io.load_10X(data_dir)` will return a Pandas DataFrame with
    genes as columns and cells as rows. The returned DataFrame will be ready to
    use with PHATE.

    Parameters
    ----------
    data_dir : string
        path to input data directory
        expects 'matrix.mtx', 'genes.tsv', 'barcodes.csv' to be present and
        will raise and error otherwise
    sparse : boolean
        If True, a sparse Pandas DataFrame is returned.
    gene_labels : string, 'id' or 'symbol', optional, default: 'id'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols

    Returns
    -------
    data : pandas.DataFrame shape=(n_cell, n_genes)
        imported data matrix
    """

    if gene_labels not in ['id', 'symbol']:
        raise ValueError("gene_labels not in ['id', 'symbol']")

    try:
        m = sio.mmread(data_dir + "/matrix.mtx")
        genes = pd.read_csv(data_dir + "/genes.tsv",
                            delimiter='\t', header=None)
        genes.columns = pd.Index(['id', 'symbol'])
        barcodes = pd.read_csv(data_dir + "/barcodes.tsv",
                               delimiter='\t', header=None)

    except (FileNotFoundError, OSError):
        raise FileNotFoundError(
            "'matrix.mtx', 'genes.tsv', and 'barcodes.tsv' must be present "
            "in data_dir")

    index = pd.Index(barcodes[0])
    columns = pd.Index(genes[gene_labels])
    if sparse and np.sum(columns.duplicated()) > 0:
        warnings.warn("Duplicate gene names detected! Forcing dense matrix. "
                      "Alternatively, try loading the matrix with "
                      "`gene_labels='id'`", RuntimeWarning)
        sparse = False

    if sparse:
        data = pd.SparseDataFrame(m.T, index=index,
                                  columns=columns,
                                  default_fill_value=0)
    else:
        data = pd.DataFrame(m.toarray().T, index=index, columns=columns)

    print("Imported data matrix with %s cells and %s genes." %
          (data.shape[0], data.shape[1]))
    return data
