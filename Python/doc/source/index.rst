===========================================================================
PHATE - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
===========================================================================

.. raw:: html

    <a href="https://pypi.org/project/phate/"><object data="https://img.shields.io/pypi/v/phate.svg" type="image/svg+xml">Latest PyPi version</object></a>

.. raw:: html

    <a href="https://phate.readthedocs.io/"><object data="https://img.shields.io/readthedocs/phate.svg" type="image/svg+xml">Read the Docs</object></a>

.. raw:: html

    <a href="https://www.biorxiv.org/content/early/2017/12/01/120378"><object data="https://zenodo.org/badge/DOI/10.1101/120378.svg" type="image/svg+xml">bioRxiv Preprint</object></a>

.. raw:: html

    <a href="https://twitter.com/KrishnaswamyLab"><object data="https://img.shields.io/twitter/follow/KrishnaswamyLab.svg?style=social&label=Follow" type="image/svg+xml">Twitter</object></a>

.. raw:: html

    <a href="https://github.com/KrishnaswamyLab/PHATE/"><object data="https://img.shields.io/github/stars/KrishnaswamyLab/PHATE.svg?style=social&label=Stars" type="image/svg+xml">GitHub stars</object></a>

PHATE is a tool for visualizing high dimensional single-cell data with natural progressions or trajectories. PHATE uses a novel conceptual framework for learning and visualizing the manifold inherent to biological systems in which smooth transitions mark the progressions of cells from one state to another. To see how PHATE can be applied to single-cell RNA-seq datasets from hematopoietic stem cells, human embryonic stem cells, and bone marrow samples, check out our preprint on BioRxiv_.

`Kevin R. Moon, David van Dijk, Zheng Wang, et al. PHATE: A Dimensionality Reduction Method for Visualizing Trajectory Structures in High-Dimensional Biological Data. 2017. BioRxiv.`__

.. _BioRxiv: https://www.biorxiv.org/content/early/2017/03/24/120378

__ BioRxiv_

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
