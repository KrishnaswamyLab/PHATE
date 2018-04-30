PHATE - Potential of Heat-diffusion for Affinity-based Transition Embedding
-------------------------------------------------------

[![Latest PyPI version](https://img.shields.io/pypi/v/phate.svg)](https://pypi.org/project/phate/)
[![Read the Docs](https://img.shields.io/readthedocs/phate.svg)](https://phate.readthedocs.io/)
[![bioRxiv Preprint](https://zenodo.org/badge/DOI/10.1101/120378.svg)](https://www.biorxiv.org/content/early/2017/12/01/120378)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)

PHATE is a tool for visualizing high dimensional single-cell data with natural progressions or trajectories. PHATE uses a novel conceptual framework for learning and visualizing the manifold inherent to biological systems in which smooth transitions mark the progressions of cells from one state to another. To see how PHATE can be applied to single-cell RNA-seq datasets from hematopoietic stem cells, human embryonic stem cells, and bone marrow samples, check out our preprint on BioRxiv.

[Kevin R. Moon, David van Dijk, Zheng Wang, et al. **PHATE: A Dimensionality Reduction Method for Visualizing Trajectory Structures in High-Dimensional Biological Data**. 2017. *BioRxiv*](http://biorxiv.org/content/early/2017/03/24/120378)


PHATE has been implemented in Python (2.7 and >=3.5), Matlab and R.

### Python installation

#### Installation with `pip`

The Python version of PHATE can be installed by running the following from a terminal:

        pip install --user phate

#### Installation from source

The Python version of PHATE can be installed from GitHub by running the following from a terminal:

        git clone --recursive git://github.com/SmitaKrishnaswamy/PHATE.git
        cd PHATE/Python
        python setup.py install --user

For more information, read the [documentation on ReadTheDocs](http://phate.readthedocs.io/en/latest/) or view our [tutorial on GitHub](https://github.com/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb)

### MATLAB installation
1. The MATLAB version of PHATE can be accessed by running the following from a terminal:

        git clone --recursive git://github.com/SmitaKrishnaswamy/PHATE.git
        cd PHATE/Matlab

2. Add the PHATE/Matlab directory to your MATLAB path and run any of our `run_*` scripts to get a feel for PHATE.

### R installation

#### Installation with `devtools`

The R version of PHATE can be installed directly from R with `devtools`:

        if (!suppressWarnings(require(devtools))) install.packages("devtools")
        devtools::install_github("KrishnaswamyLab/phater")

#### Installation from source

1. The R version of PHATE can be accessed [here](https://github.com/KrishnaswamyLab/phater), or by running the following from a terminal:

        git clone --recursive git://github.com/SmitaKrishnaswamy/PHATE.git
        cd PHATE/phater
        R CMD INSTALL

2. If the `phater` folder is empty, you have may forgotten to use the `--recursive` option for `git clone`. You can rectify this by running the following from a terminal:

        cd PHATE
        git submodule init
        git submodule update
        cd phater
        R CMD INSTALL

For more information, read the [phater README](https://github.com/KrishnaswamyLab/phater).
