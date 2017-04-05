import scipy.io as sio
import numpy as np

def import_single_cell_file(data_file, file_type='csv'):
    """Basic IO

    Parameters
    ----------
    data_file : string
        path to input data file
    file_type : string, optional, default: 'csv'
        File type of input data
        Supported Values: ['mtx', 'csv', 'tsv']

    Returns
    -------
    M : ndarray
        imported data file
        for Phate, M must have shape [n_cells, n_dimensions]
    """



    file_type = file_type.lower()

    if file_type == 'mtx':
        M = sio.mmread(data_file)
    elif file_type == 'csv':
        M = np.loadtxt(data_file, delimiter=',')
    elif file_type == 'tsv':
        M = np.loadtxt(data_file, delimiter='\t')
    elif file_type == 'fcs':
        raise NotImplementedError("FCS files are not currently supported. Please post to the GitHub if you're interested in this function.")
    else:
        raise ValueError("Supported files types are ['mtx', 'csv', 'tsv']")

    print("Imported data matrix with %s cells and %s genes..."%(M.shape[0], M.shape[1]))
    return(M)
