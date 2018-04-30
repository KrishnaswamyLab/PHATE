Installation
============

Python installation
-------------------

Installation with `pip`
~~~~~~~~~~~~~~~~~~~~~~~

The Python version of PHATE can be installed using::

       pip install --user phate

Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~

The Python version of PHATE can be installed from GitHub by running the following from a terminal::

       git clone --recursive git://github.com/SmitaKrishnaswamy/PHATE.git
       cd Python
       python setup.py install --user

MATLAB installation
-------------------

1. The MATLAB version of PHATE can be accessed using::

    git clone git://github.com/SmitaKrishnaswamy/PHATE.git
    cd PHATE/Matlab

2. Add the PHATE/Matlab directory to your MATLAB path and run any of our `test` scripts to get a feel for PHATE.

R installation
--------------

Installation with `devtools`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The R version of PHATE can be installed directly from R with `devtools`::

        if (!suppressWarnings(require(devtools))) install.packages("devtools")
        devtools::install_github("KrishnaswamyLab/phater")

Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~

The R version of PHATE can be accessed GitHub_ at `https://github.com/KrishnaswamyLab/phater`__, or by running the following from a terminal::

        git clone --recursive git://github.com/SmitaKrishnaswamy/PHATE.git
        cd PHATE/phater
        R CMD INSTALL

22. If the `phater` folder is empty, you have may forgotten to use the `--recursive` option for `git clone`. You can rectify this by running the following from a terminal::

        cd PHATE
        git submodule init
        git submodule update
        cd phater
        R CMD INSTALL

.. _GitHub: https://github.com/KrishnaswamyLab/phater

__ GitHub_