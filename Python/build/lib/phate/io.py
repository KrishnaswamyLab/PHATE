import pandas as pd
import scipy.io as sio

def load_10X(data_dir, gene_labels='id'):
    """Basic IO for 10X data produced from the 10X Cellranger pipeline.

    A default run of the `cellranger count` command will generate gene-barcode matricies for secondary analysis. For both "raw" and "filtered" output, directories are created containing three files: 'matrix.mtx', 'barcodes.tsv', 'genes.tsv'. Running `phate.io.load_10X(data_dir)` will return a Pandas DataFrame will genes as columns and cells as rows. The returned DataFrame will be ready to use with PHATE.

    Parameters
    ----------
    data_dir : string
        path to input data directory
        expects 'matrix.mtx', 'genes.tsv', 'barcodes.csv' to be present and will raise and error otherwise
    gene_labels : string, 'id' or 'symbol', optional, default: 'id'
        Whether the columns of the dataframe should contain gene ids or gene symbols

    Returns
    -------
    data : pandas.DataFrame shape=(n_cell, n_genes)
        imported data matrix
    """

    if gene_labels not in ['id', 'symbol']:
        raise ValueError("gene_labels not in ['id', 'symbol']")

    try:
        m =  sio.mmread(data_dir + "/matrix.mtx")
        data = pd.DataFrame(m.toarray().T)
    except FileNotFoundError:
        raise FileNotFoundError("'matrix.mtx', 'genes.tsv', and 'barcodes.tsv' must be present in data_dir")
    try:
        genes = pd.read_csv(data_dir + "/genes.tsv", delimiter='\t', header=None)
        genes.columns = pd.Index(['id', 'symbol'])
        barcodes = pd.read_csv(data_dir + "/barcodes.tsv", delimiter='\t', header=None)
    except OSError:
        raise FileNotFoundError("'matrix.mtx', 'genes.tsv', and 'barcodes.tsv' must be present in data_dir")


    data.columns = pd.Index(genes[gene_labels])
    data.index = pd.Index(barcodes[0])



    print("Imported data matrix with %s cells and %s genes."%(data.shape[0], data.shape[1]))
    return data
