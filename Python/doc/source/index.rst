===========================================================================
PHATE - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
===========================================================================

.. raw:: html

    <a href="https://pypi.org/project/phate/"><img src="https://img.shields.io/pypi/v/phate.svg" alt="Latest PyPi version"></a>

.. raw:: html

    <a href="https://anaconda.org/bioconda/phate/"><img src="https://anaconda.org/bioconda/phate/badges/version.svg" alt="Latest Conda version"></a>

.. raw:: html

    <a href="https://cran.r-project.org/package=phateR"><img src="https://img.shields.io/cran/v/phateR.svg" alt="Latest CRAN version"></a>

.. raw:: html

    <a href="https://travis-ci.com/KrishnaswamyLab/PHATE"><img src="https://api.travis-ci.com/KrishnaswamyLab/phate.svg?branch=master" alt="Travis CI Build"></a>

.. raw:: html

    <a href="https://phate.readthedocs.io/"><img src="https://img.shields.io/readthedocs/phate.svg" alt="Read the Docs"></img></a>

.. raw:: html

    <a href="https://doi.org/10.1038/s41587-019-0336-3"><img src="https://zenodo.org/badge/DOI/10.1038/s41587-019-0336-3.svg" alt="Nature Biotechnology Publication"></a>

.. raw:: html

    <a href="https://twitter.com/KrishnaswamyLab"><img src="https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow" alt="Twitter"></a>

.. raw:: html

    <a href="https://github.com/KrishnaswamyLab/PHATE/"><img src="https://img.shields.io/github/stars/KrishnaswamyLab/PHATE.svg?style=social&label=Stars" alt="GitHub stars"></a>

PHATE (Potential of Heat-diffusion for Affinity-based Trajectory Embedding) is a tool for visualizing high dimensional data. PHATE uses a novel conceptual framework for learning and visualizing the manifold to preserve both local and global distances.

To see how PHATE can be applied to datasets such as facial images and single-cell data from human embryonic stem cells, check out our `Nature Biotechnology publication`_.

`Moon, van Dijk, Wang, Gigante et al. **Visualizing Transitions and Structure for Biological Data Exploration**. 2019. *Nature Biotechnology*.`__

.. _`Nature Biotechnology publication`: https://doi.org/10.1038/s41587-019-0336-3

__ `Nature Biotechnology publication`_

.. toctree::
    :maxdepth: 2

    installation
    tutorial
    api

Quick Start
===========

If you have loaded a data matrix ``data`` in Python (cells on rows, genes on columns) you can run PHATE as follows::

    import phate
    phate_op = phate.PHATE()
    data_phate = phate_op.fit_transform(data)

PHATE accepts the following data types: ``numpy.array``, ``scipy.spmatrix``, ``pandas.DataFrame`` and ``anndata.AnnData``.

Usage
=====

To run PHATE on your dataset, create a PHATE operator and run `fit_transform`. Here we show an example with an artificial tree::

    import phate
    tree_data, tree_clusters = phate.tree.gen_dla()
    phate_operator = phate.PHATE(k=15, t=100)
    tree_phate = phate_operator.fit_transform(tree_data)
    phate.plot.scatter2d(phate_operator, c=tree_clusters) 
    # or phate.plot.scatter2d(tree_phate, c=tree_clusters)
    phate.plot.rotate_scatter3d(phate_operator, c=tree_clusters)

Help
====

If you have any questions or require assistance using PHATE, please contact us at https://krishnaswamylab.org/get-help

.. autoclass:: phate.PHATE
    :members:
    :noindex:
