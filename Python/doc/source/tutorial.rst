Tutorial
--------

To run PHATE on your dataset, create a PHATE operator and run `fit_transform`. Here we show an example with an artificial tree::

        import phate
        tree_data, tree_clusters = phate.tree.gen_dla()
        phate_operator = phate.PHATE(k=15, t=100)
        tree_phate = phate_operator.fit_transform(tree_data)
        phate.plot.scatter2d(phate_operator, c=tree_clusters) # or phate.plot.scatter2d(tree_phate, c=tree_clusters)
        phate.plot.rotate_scatter3d(phate_operator, c=tree_clusters)

A demo on PHATE usage and visualization for single cell RNA-seq data can be found in this `Jupyter notebook <http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb>`_. A second tutorial is available `here <http://nbviewer.jupyter.org/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb>`_ which works with the artificial tree shown above in more detail. You can also access interactive versions of these tutorials on Google Colaboratory: `single cell RNA seq <https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/EmbryoidBody.ipynb>`_, `artificial tree <https://colab.research.google.com/github/KrishnaswamyLab/PHATE/blob/master/Python/tutorial/PHATE_tree.ipynb>`_.
