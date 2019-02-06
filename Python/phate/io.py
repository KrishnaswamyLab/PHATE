# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import warnings
import scprep


def load_10X(data_dir, sparse=True, gene_labels='symbol'):
    warnings.warn("phate.io is deprecated. Please use scprep.io instead. "
                  "Read more at http://scprep.readthedocs.io",
                  FutureWarning)
    return scprep.io.load_10X(data_dir, sparse=sparse, gene_labels=gene_labels)
