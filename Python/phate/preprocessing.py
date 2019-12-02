# author: Daniel Burkhardt <daniel.burkhardt@yale.edu>
# (C) 2017 Krishnaswamy Lab GPLv2

from __future__ import print_function, division
import warnings
import scprep
from deprecated import deprecated

@deprecated("1.5.0", reason="Use scprep.normalize.library_size_normalize instead")
def library_size_normalize(data, verbose=False):
    return scprep.normalize.library_size_normalize(data)
