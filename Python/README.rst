===========================================================================
PHATE - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
===========================================================================

.. image:: https://img.shields.io/pypi/v/phate.svg
    :target: https://pypi.org/project/phate/
    :alt: Latest PyPI version
.. image:: https://img.shields.io/readthedocs/phate.svg
    :target: https://phate.readthedocs.io/
    :alt: Read the Docs
.. image:: https://zenodo.org/badge/DOI/10.1101/120378.svg
    :target: https://www.biorxiv.org/content/early/2017/12/01/120378
    :alt: bioRxiv Preprint

PHATE has been implemented in Python (2.7 and >=3.5), R and Matlab.

Python installation and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Installation with `pip`
-----------------------

The Python version of PHATE can be installed using:

   ::

       pip install --user phate

Installation from source
------------------------

The Python version of PHATE can be installed from GitHub by running the following from a terminal:

   ::

       git clone --recursive git://github.com/SmitaKrishnaswamy/PHATE.git
       cd Python
       python setup.py install --user

Usage
~~~~~

PHATE has been implemented with an API that should be familiar to those
with experience using scikit-learn. The core of the PHATE package is the
``PHATE`` class which is a subclass of ``sklearn.base.BaseEstimator``.
To get started, ``import phate`` and instantiate a ``phate.PHATE()``
object. Just like most ``sklearn`` estimators, ``PHATE()`` objects have
both ``fit()`` and ``fit_transform()`` methods. For more information,
check out our notebook below.

If you want to try running our test script on a DLA fractal tree: 1.
Make the test scripts executable

::

        import phate
        import matplotlib.pyplot as plt
        tree_data, tree_clusters = phate.tree.gen_dla()
        phate_operator = phate.PHATE(k=15, t=100)
        tree_phate = phate_operator.fit_transform(tree_data)
        plt.scatter(tree_phate[:,0], tree_phate[:,1], c=tree_clusters)
        plt.show()

Jupyter Notebook
~~~~~~~~~~~~~~~~

A demo on PHATE usage and visualization for single cell RNA-seq data can
be found in this notebook:
https://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb
