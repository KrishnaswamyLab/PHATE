===========================================================================
PHATE - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
===========================================================================

.. raw:: html

    <a href="https://pypi.org/project/phate/"><img src="https://img.shields.io/pypi/v/phate.svg" alt="Latest PyPi version"></a>

.. raw:: html

    <a href="https://cran.r-project.org/package=phateR"><img src="https://img.shields.io/cran/v/phateR.svg" alt="Latest CRAN version"></a>

.. raw:: html

    <a href="https://phate.readthedocs.io/"><img src="https://img.shields.io/readthedocs/phate.svg" alt="Read the Docs"></img></a>

.. raw:: html

    <a href="https://www.biorxiv.org/content/early/2017/12/01/120378"><img src="https://zenodo.org/badge/DOI/10.1101/120378.svg" alt="bioRxiv Preprint"></a>

.. raw:: html

    <a href="https://twitter.com/KrishnaswamyLab"><img src="https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow" alt="Twitter"></a>

.. raw:: html

    <a href="https://github.com/KrishnaswamyLab/PHATE/"><img src="https://img.shields.io/github/stars/KrishnaswamyLab/PHATE.svg?style=social&label=Stars" alt="GitHub stars"></a>

PHATE is a tool for visualizing high dimensional single-cell data with natural progressions or trajectories. PHATE uses a novel conceptual framework for learning and visualizing the manifold inherent to biological systems in which smooth transitions mark the progressions of cells from one state to another. To see how PHATE can be applied to single-cell RNA-seq datasets from hematopoietic stem cells, human embryonic stem cells, and bone marrow samples, check out our `preprint on BioRxiv`_.

`Kevin R. Moon, David van Dijk, Zheng Wang, et al. PHATE: A Dimensionality Reduction Method for Visualizing Trajectory Structures in High-Dimensional Biological Data. 2017. BioRxiv.`__

.. _`preprint on BioRxiv`: https://www.biorxiv.org/content/early/2017/03/24/120378

__ `preprint on BioRxiv`_

.. toctree::
    :maxdepth: 2

    installation
    tutorial
    api

Quick Start
===========

To run PHATE on your dataset, create a PHATE operator and run `fit_transform`. Here we show an example with an artificial tree::

    import phate
    import matplotlib.pyplot as plt
    tree_data, tree_clusters = phate.tree.gen_dla()
    phate_operator = phate.PHATE(k=15, t=100)
    tree_phate = phate_operator.fit_transform(tree_data)
    plt.scatter(tree_phate[:,0], tree_phate[:,1], c=tree_clusters)
    plt.show()

.. autoclass:: phate.PHATE
    :members:
    :noindex:
