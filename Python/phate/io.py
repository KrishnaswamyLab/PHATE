# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import pandas as pd
import scipy.io as sio
import warnings
import numpy as np
import os


def _combine_gene_id(genes):
    """Creates gene labels of the form SYMBOL (ID)

    Parameters
    ----------

    genes : pandas.DataFrame with columns ['symbol', 'id']

    Returns
    -------

    pandas.Index with combined gene symbols and ids
    """
    columns = np.core.defchararray.add(
        np.array(genes['symbol'], dtype=str), ' (')
    columns = np.core.defchararray.add(
        columns, np.array(genes['id'], dtype=str))
    columns = np.core.defchararray.add(columns, ')')
    return pd.Index(columns)


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
    gene_labels : string, {'id', 'symbol', 'both'} optional, default: 'symbol'
        Whether the columns of the dataframe should contain gene ids or gene
        symbols. If 'both', returns symbols followed by ids in parentheses.

    Returns
    -------
    data : pandas.DataFrame shape=(n_cell, n_genes)
        imported data matrix
    """

    if gene_labels not in ['id', 'symbol', 'both']:
        raise ValueError("gene_labels not in ['id', 'symbol', 'both']")

    try:
        m = sio.mmread(os.path.join(data_dir, "matrix.mtx"))
        genes = pd.read_csv(os.path.join(data_dir, "genes.tsv"),
                            delimiter='\t', header=None)
        genes.columns = pd.Index(['id', 'symbol'])
        barcodes = pd.read_csv(os.path.join(data_dir, "barcodes.tsv"),
                               delimiter='\t', header=None)

    except (FileNotFoundError, OSError):
        raise FileNotFoundError(
            "'matrix.mtx', 'genes.tsv', and 'barcodes.tsv' must be present "
            "in data_dir")

    index = pd.Index(barcodes[0])
    if gene_labels == 'both':
        columns = _combine_gene_id(genes)
    else:
        columns = pd.Index(genes[gene_labels])
        if sparse and np.sum(columns.duplicated()) > 0:
            warnings.warn(
                "Duplicate gene names detected! Forcing `gene_labels='id'`."
                "Alternatively, try loading the matrix with "
                "`sparse=False`", RuntimeWarning)
            columns = genes['id']

    if sparse:
        data = pd.SparseDataFrame(m.T, index=index,
                                  columns=columns,
                                  default_fill_value=0)
    else:
        data = pd.DataFrame(m.toarray().T, index=index, columns=columns)

    print("Imported data matrix with %s cells and %s genes." %
          (data.shape[0], data.shape[1]))
    return data
