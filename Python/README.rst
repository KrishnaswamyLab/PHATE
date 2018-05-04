===========================================================================
PHATE - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
===========================================================================

.. raw:: html

    <a href="https://pypi.org/project/phate/"><img src="https://img.shields.io/pypi/v/phate.svg" alt="Latest PyPi version"></img></a>

.. raw:: html

    <a href="https://phate.readthedocs.io/"><img src="https://img.shields.io/readthedocs/phate.svg" alt="Read the Docs"></img></a>

.. raw:: html

    <a href="https://www.biorxiv.org/content/early/2017/12/01/120378"><img src="https://zenodo.org/badge/DOI/10.1101/120378.svg" alt="bioRxiv Preprint"></img></a>

.. raw:: html

    <a href="https://twitter.com/KrishnaswamyLab"><img src="https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow" alt="Twitter"></img></a>

.. raw:: html

    <a href="https://github.com/KrishnaswamyLab/PHATE/"><img src="https://img.shields.io/github/stars/KrishnaswamyLab/PHATE.svg?style=social&label=Stars" alt="GitHub stars"></img></a>

PHATE is a tool for visualizing high dimensional single-cell data with natural progressions or trajectories. PHATE uses a novel conceptual framework for learning and visualizing the manifold inherent to biological systems in which smooth transitions mark the progressions of cells from one state to another. To see how PHATE can be applied to single-cell RNA-seq datasets from hematopoietic stem cells, human embryonic stem cells, and bone marrow samples, check out our preprint on BioRxiv.

PHATE has been implemented in Python (2.7 and >=3.5), R_ and MATLAB_.

.. _R: https://github.com/KrishnaswamyLab/phater
.. _MATLAB: https://github.com/KrishnaswamyLab/PHATE

Python installation and dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Installation with ``pip``
-------------------------

The Python version of PHATE can be installed using::

       pip install --user phate

Installation from source
------------------------

The Python version of PHATE can be installed from GitHub by running the following from a terminal::

       git clone --recursive git://github.com/KrishnaswamyLab/PHATE.git
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

If you want to try running our test script on a DLA fractal tree, run the following in a Python interpreter::

        import phate
        import matplotlib.pyplot as plt
        tree_data, tree_clusters = phate.tree.gen_dla()
        phate_operator = phate.PHATE(k=15, t=100)
        tree_phate = phate_operator.fit_transform(tree_data)
        plt.scatter(tree_phate[:,0], tree_phate[:,1], c=tree_clusters)
        plt.show()

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

A demo on PHATE usage and visualization for single cell RNA-seq data can be found in this notebook_: http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb

.. _notebook: http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb

A second tutorial is available here_ which works with the artificial tree shown above in more detail: http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb

.. _here: http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb
