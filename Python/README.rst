===========================================================================
PHATE - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
===========================================================================

.. image:: https://img.shields.io/pypi/v/phate.svg
    :target: https://pypi.org/project/phate/
    :alt: Latest PyPi version
.. image:: https://img.shields.io/cran/v/phateR.svg
    :target: https://cran.r-project.org/package=phateR
    :alt: Latest CRAN version
.. image:: https://api.travis-ci.com/KrishnaswamyLab/phate.svg?branch=master
    :target: https://travis-ci.com/KrishnaswamyLab/PHATE
    :alt: Travis CI Build
.. image:: https://img.shields.io/readthedocs/phate.svg
    :target: https://phate.readthedocs.io/
    :alt: Read the Docs
.. image:: https://zenodo.org/badge/DOI/10.1101/120378.svg
    :target: https://doi.org/10.1101/120378
    :alt: bioRxiv Preprint
.. image:: https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow
    :target: https://twitter.com/KrishnaswamyLab
    :alt: Twitter
.. image:: https://img.shields.io/github/stars/KrishnaswamyLab/PHATE.svg?style=social&label=Stars
    :target: https://github.com/KrishnaswamyLab/PHATE/
    :alt: GitHub stars

PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding) is a tool for visualizing high dimensional data. PHATE uses a novel conceptual framework for learning and visualizing the manifold to preserve both local and global distances.

To see how PHATE can be applied to datasets such as facial images and single-cell data from human embryonic stem cells, check out our `preprint on BioRxiv`_.

`Kevin R. Moon, David van Dijk, Zheng Wang, et al. Visualizing Transitions and Structure for Biological Data Exploration. 2018. BioRxiv.`__

.. _`preprint on BioRxiv`: https://www.biorxiv.org/content/early/2017/03/24/120378

__ `preprint on BioRxiv`_

PHATE has been implemented in Python >=3.5, R_ and MATLAB_.

.. _R: https://github.com/KrishnaswamyLab/phateR
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

Quick Start
~~~~~~~~~~~

If you have loaded a data matrix ``data`` in Python (cells on rows, genes on columns) you can run PHATE as follows::

    import phate
    phate_op = phate.PHATE()
    data_phate = phate_op.fit_transform(data)

PHATE accepts the following data types: ``numpy.array``, ``scipy.spmatrix``, ``pandas.DataFrame`` and ``anndata.AnnData``.

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
        tree_data, tree_clusters = phate.tree.gen_dla()
        phate_operator = phate.PHATE(k=15, t=100)
        tree_phate = phate_operator.fit_transform(tree_data)
        phate.plot.scatter2d(phate_operator, c=tree_clusters) # or phate.plot.scatter2d(tree_phate, c=tree_clusters)
        phate.plot.rotate_scatter3d(phate_operator, c=tree_clusters)

Jupyter Notebooks
~~~~~~~~~~~~~~~~~

A demo on PHATE usage and visualization for single cell RNA-seq data can be found in this `Jupyter notebook <http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb>`_. A second tutorial is available `here <http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb>`_ which works with the artificial tree shown above in more detail. You can also access interactive versions of these tutorials on Google Colaboratory: `single cell RNA seq <https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb>`_, `artificial tree <https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb>`_.

Help
^^^^

If you have any questions or require assistance using PHATE, please contact us at https://krishnaswamylab.org/get-help
