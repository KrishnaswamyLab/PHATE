PHATE - Visualizing Transitions and Structure for Biological Data Exploration
---------------------------------------------------------------------------

[![Latest PyPI version](https://img.shields.io/pypi/v/phate.svg)](https://pypi.org/project/phate/)
[![Latest Conda version](https://anaconda.org/bioconda/phate/badges/version.svg)](https://anaconda.org/bioconda/phate/)
[![Latest CRAN version](https://img.shields.io/cran/v/phateR.svg)](https://cran.r-project.org/package=phateR)
[![Travis CI Build](https://api.travis-ci.com/KrishnaswamyLab/phate.svg?branch=master)](https://travis-ci.com/KrishnaswamyLab/PHATE)
[![Read the Docs](https://img.shields.io/readthedocs/phate.svg)](https://phate.readthedocs.io/)
[![Nature Biotechnology Publication](https://zenodo.org/badge/DOI/10.1038/s41587-019-0336-3.svg)](https://www.nature.com/articles/s41587-019-0336-3)
[![Twitter](https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow)](https://twitter.com/KrishnaswamyLab)

### Quick Start
If you would like to get started using PHATE, check out our [**guided tutorial in Python**](http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb).

If you have loaded a data matrix `data` in Python (cells on rows, genes on columns) you can run PHATE as follows:

    import phate
    phate_op = phate.PHATE()
    data_phate = phate_op.fit_transform(data)

PHATE accepts the following data types: `numpy.array`, `scipy.spmatrix`, `pandas.DataFrame` and `anndata.AnnData`.

### Introduction

PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding) is a tool for visualizing high dimensional data. PHATE uses a novel conceptual framework for learning and visualizing the manifold to preserve both local and global distances.

To see how PHATE can be applied to datasets such as facial images and single-cell data from human embryonic stem cells, check out our publication in Nature Biotechnology.

[Moon, van Dijk, Wang, Gigante et al. **Visualizing Transitions and Structure for Biological Data Exploration**. 2019. *Nature Biotechnology*.](https://doi.org/10.1038/s41587-019-0336-3)

PHATE has been implemented in [Python >=3.5](#python), [MATLAB](https://github.com/KrishnaswamyLab/PHATE/#matlab) and [R](https://github.com/KrishnaswamyLab/phateR/).

### Table of Contents

* [System Requirements](#system-requirements)
* [Installation with pip](#installation-with-pip)
* [Installation from source](#installation-from-source)
* [Quick Start](#quick-start)
* [Tutorial and Reference](#tutorial-and-reference)
* [Help](#help)

### System Requirements

* Windows (>= 7), Mac OS X (>= 10.8) or Linux
* [Python >= 3.5](https://www.python.org/downloads/)

All other software dependencies are installed automatically when installing PHATE.

### Installation with `pip`

The Python version of PHATE can be installed by running the following from a terminal:

    pip install --user phate

Installation of PHATE and all dependencies should take no more than five minutes.

### Installation from source

The Python version of PHATE can be installed from GitHub by running the following from a terminal:

    git clone --recursive git://github.com/KrishnaswamyLab/PHATE.git
    cd PHATE/Python
    python setup.py install --user

### Tutorial and Reference

For more information, read the [documentation on ReadTheDocs](http://phate.readthedocs.io/) or view our tutorials on GitHub: [single-cell RNA-seq](http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb), [artificial tree](http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb). You can also access interactive versions of these tutorials on Google Colaboratory: [single-cell RNA-seq](https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb), [artificial tree](https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb).

### Help

If you have any questions or require assistance using PHATE, please contact us at <https://krishnaswamylab.org/get-help>.
