PHATE  - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
----------------------------------------------------------------------------

PHATE has been implemented in Python (2.7 and >=3.5), R and Matlab.

#### Python installation and dependencies
1. The Python version of PHATE can be installed using:

        $ git clone git://github.com/SmitaKrishnaswamy/PHATE.git
        $ cd Python
        $ python setup.py install --user

2. PHATE depends on a number of `python` packages available on pypi and these dependencies are listed in `setup.py`
All the dependencies will be automatically installed using the above commands

### Usage
PHATE has been implemented with an API that should be familiar to those with experience using scikit-learn. The core of the PHATE package is the `PHATE` class which is a subclass of `sklearn.base.BaseEstimator`.  To get started, `import phate` and instantiate a `phate.PHATE()` object. Just like most `sklearn` estimators, `PHATE()` objects have both `fit()` and `fit_transform()` methods. For more information, check out our notebook below.

If you want to try running our test script on a DLA fractal tree:
1. Make the test scripts executable

        $ cd PHATE/Python/test
        $ chmod +x phate_test_tree.py phate_test_mESC.py
        $ ./phate_test_tree.py #output saved in a png

### Jupyter Notebook

A demo on PHATE usage and visualization for single cell RNA-seq data can be found in this notebook: [https://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb](https://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb)
