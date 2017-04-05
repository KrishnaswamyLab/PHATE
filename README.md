PHATE  - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
-------------------------------------------------------

PHATE is a tool for visualizing high dimensional single-cell data with natural progressions or trajectories. PHATE uses a novel conceptual framework for learning and visualizing the manifold inherent to biological systems in which smooth transitions mark the progressions of cells from one state to another. To see how PHATE can be applied to single-cell RNA-seq datasets from hematopoietic stem cells, human embryonic stem cells, and bone marrow samples, check out our preprint on BioRxiv.

[Kevin R. Moon, David van Dijk, Zheng Wang, et al. **PHATE: A Dimensionality Reduction Method for Visualizing Trajectory Structures in High-Dimensional Biological Data**. 2017. *BioRxiv*](http://biorxiv.org/content/early/2017/03/24/120378)


PHATE has been implemented in Python3 and Matlab.


### Getting started

#### Python installation and dependencies
1. The Python3 version of PHATE can be installed using:

        $ git clone git://github.com/SmitaKrishnaswamy/PHATE.git
        $ cd Python
        $ python3 setup.py install --user

2. PHATE depends on a number of `python3` packages available on pypi and these dependencies are listed in `setup.py`
All the dependencies will be automatically installed using the above commands

#### MATLAB installation
1. The MATLAB version of PHATE can be accessed using:

        $ git clone git://github.com/SmitaKrishnaswamy/PHATE.git
        $ cd PHATE/Matlab

2. Add the PHATE/Matlab directory to your MATLAB path and run any of our `test` scripts to get a feel for PHATE.

#### Installation and dependencies for the R version

1. Check back soon, R code is coming!

### Python demo
A demo on PHATE usage and visualization for single cell RNA-seq data can be found in this notebook: [https://nbviewer.jupyter.org/github/SmitaKrishnaswamy/PHATE/blob/master/Python/test/phate_examples.ipynb](https://nbviewer.jupyter.org/github/SmitaKrishnaswamy/PHATE/blob/master/Python/test/phate_examples.ipynb?flush_cache=true)
