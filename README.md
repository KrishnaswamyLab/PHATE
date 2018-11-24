PHATE - Potential of Heat-diffusion for Affinity-based Transition Embedding
---------------------------------------------------------------------------

[![Latest PyPI version](https://img.shields.io/pypi/v/phate.svg)](https://pypi.org/project/phate/)
[![Latest CRAN version](https://img.shields.io/cran/v/phateR.svg)](https://cran.r-project.org/package=phateR)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/phate.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/PHATE)
[![Read the Docs](https://img.shields.io/readthedocs/phate.svg)](https://phate.readthedocs.io/)
[![bioRxiv Preprint](https://zenodo.org/badge/DOI/10.1101/120378.svg)](https://www.biorxiv.org/content/early/2017/12/01/120378)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)

PHATE is a tool for visualizing high dimensional data. In particular, PHATE is well-suited for visualizing data with natural progressions or trajectories such as single-cell data. PHATE uses a novel conceptual framework for learning and visualizing the manifold inherent to the system in which smooth transitions mark the progressions of data points (e.g. cells) from one state to another. To see how PHATE can be applied to datasets such as facial images and single-cell data from human embryonic stem cells, check out our preprint on BioRxiv.

[Kevin R. Moon, David van Dijk, Zheng Wang, et al. **Visualizing Transitions and Structure for Biological Data Exploration**. 2018. *BioRxiv*](https://doi.org/10.1101/120378)

PHATE has been implemented in [Python](#python) (2.7 and >=3.5), [MATLAB](#matlab) and [R](#r).

### Table of Contents

* [System Requirements](#system-requirements)
* [Python](#python)
    * [Installation with pip](#installation-with-pip)
    * [Installation from source](#installation-from-source)
    * [Quick Start](#quick-start)
    * [Tutorial and Reference](#tutorial-and-reference)
* [MATLAB](#matlab)
    * [Installation](#installation)
    * [Tutorial and Reference](#tutorial-and-reference-1)
* [R](#r)
    * [Installation from CRAN and PyPi](#installation-from-cran-and-pypi)
    * [Installation with devtools and reticulate](#installation-with-devtools-and-reticulate)
    * [Installation from source](#installation-from-source-1)
    * [Quick Start](#quick-start-1)
    * [Tutorial and Reference](#tutorial-and-reference-2)
* [Help](#help)

### System Requirements

* Windows (>= 7), Mac OS X (>= 10.8) or Linux
* [Python](https://www.python.org/downloads/) (2.7 or >=3.5) or [MATLAB](https://www.mathworks.com/products/matlab.html) (>= 2015a)

All other software dependencies are installed automatically when installing PHATE.

### Python

#### Installation with `pip`

The Python version of PHATE can be installed by running the following from a terminal:

    pip install --user phate

Installation of PHATE and all dependencies should take no more than five minutes.

#### Installation from source

The Python version of PHATE can be installed from GitHub by running the following from a terminal:

    git clone --recursive git://github.com/KrishnaswamyLab/PHATE.git
    cd PHATE/Python
    python setup.py install --user

#### Quick Start

If you have loaded a data matrix `data` in Python (cells on rows, genes on columns) you can run PHATE as follows::

    import phate
    phate_op = phate.PHATE()
    data_phate = phate_op.fit_transform(data)

PHATE accepts the following data types: `numpy.array`, `scipy.spmatrix`, `pandas.DataFrame` and `anndata.AnnData`.

#### Tutorial and Reference

For more information, read the [documentation on ReadTheDocs](http://phate.readthedocs.io/) or view our tutorials on GitHub: [single-cell RNA-seq](http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb), [artificial tree](http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb). You can also access interactive versions of these tutorials on Google Colaboratory: [single-cell RNA-seq](https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb), [artificial tree](https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb).

### MATLAB

#### Installation

1. The MATLAB version of PHATE can be accessed by running the following from a terminal:

    git clone --recursive git://github.com/KrishnaswamyLab/PHATE.git
    cd PHATE/Matlab

2. Add the PHATE/Matlab directory to your MATLAB path.

Installation of PHATE should take no more than five minutes.

#### Tutorial and Reference

Run any of our `run_*` scripts to get a feel for PHATE. Documentation is available in the MATLAB help viewer.

### R

In order to use PHATE in R, you must also install the Python package.

#### Installation from CRAN and PyPi

Install `phate` in Python by running the following code from a terminal:

    pip install --user phate

Then, install `phateR` from CRAN by running the following code in R:

    install.packages("phateR")

If `python` or `pip` are not installed, you will need to install them. We recommend [Miniconda3](https://conda.io/miniconda.html) to install Python and `pip` together, or otherwise you can install `pip` from https://pip.pypa.io/en/stable/installing/.

Installation of PHATE and all dependencies should take no more than five minutes.

#### Installation with `devtools` and `reticulate`

The development version of PHATE can be installed directly from R with `devtools`:

    if (!suppressWarnings(require(devtools))) install.packages("devtools")
    devtools::install_github("KrishnaswamyLab/phateR")

If you have the development version of `reticulate`, you can also install `phate` in Python by running the following code in R:

    devtools::install_github("rstudio/reticulate")
    reticulate::py_install("phate", pip=TRUE)

#### Installation from source

The latest source version of PHATE can be accessed by running the following in a terminal:

    git clone --recursive git://github.com/SmitaKrishnaswamy/PHATE.git
    cd PHATE/phateR
    R CMD INSTALL
    cd ../Python
    python setup.py install --user

If the `phateR` folder is empty, you have may forgotten to use the `--recursive` option for `git clone`. You can rectify this by running the following in a terminal:

    cd PHATE
    git submodule init
    git submodule update
    cd phateR
    R CMD INSTALL
    cd ../Python
    python setup.py install --user

#### Quick Start

If you have loaded a data matrix `data` in R (cells on rows, genes on columns) you can run PHATE as follows::

    library(phateR)
    data_phate <- phate(data)

phateR accepts R matrices, `Matrix` sparse matrices, `data.frame`s, and any other data type that can be converted to a matrix with the function `as.matrix`.

#### Tutorial and Reference

For more information and a tutorial, read the [phateR README](https://github.com/KrishnaswamyLab/phateR). Documentation is available in the R help viewer with `help(phateR::phate)`. A tutorial notebook running PHATE on a single-cell RNA-seq dataset is available at <http://htmlpreview.github.io/?https://github.com/KrishnaswamyLab/phateR/blob/master/inst/examples/bonemarrow_tutorial.html> or in `phateR/inst/examples`.

### Help

If you have any questions or require assistance using PHATE, please contact us at <https://krishnaswamylab.org/get-help>.
