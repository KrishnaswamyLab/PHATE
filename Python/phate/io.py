# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import warnings
import scprep
from deprecated import deprecated

@deprecated("1.5.0", reason="Use scprep.io.load_10X instead")
def load_10X(data_dir, sparse=True, gene_labels="symbol"):
    return scprep.io.load_10X(data_dir, sparse=sparse, gene_labels=gene_labels)
