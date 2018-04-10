===========================================================================
PHATE - Potential of Heat-diffusion for Affinity-based Trajectory Embedding
===========================================================================

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
    tree_data, tree_clusters = phate.tree.gen_dla()
    phate_operator = phate.PHATE()
    tree_phate = phate_operator.fit_transform(tree_data)

.. autoclass:: phate.PHATE
    :noindex:
