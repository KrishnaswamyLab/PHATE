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

       git clone --recursive git://github.com/KrishnaswamyLab/PHATE.git
       cd Python
       python setup.py install --user

MATLAB installation
-------------------

1. The MATLAB version of PHATE can be accessed using::

    git clone git://github.com/KrishnaswamyLab/PHATE.git
    cd PHATE/Matlab

2. Add the PHATE/Matlab directory to your MATLAB path and run any of our `test` scripts to get a feel for PHATE.

R installation
--------------

In order to use PHATE in R, you must also install the Python package.

Installation from CRAN and PyPi
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install `phater` from CRAN by running the following code in R::

    install.packages("phater")

Install `phate` in Python by running the following code from a terminal::

    pip install --user phate

Installation with `devtools` and `reticulate`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The development version of PHATE can be installed directly from R with `devtools`::

    if (!suppressWarnings(require(devtools))) install.packages("devtools")
    devtools::install_github("KrishnaswamyLab/phater")

If you have the development version of `reticulate`, you can also install `phate` in Python by running the following code in R::

    devtools::install_github("rstudio/reticulate")
    reticulate::py_install("phate")

Installation from source
~~~~~~~~~~~~~~~~~~~~~~~~

The latest source version of PHATE can be accessed by running the following in a terminal::

    git clone --recursive git://github.com/SmitaKrishnaswamy/PHATE.git
    cd PHATE/phater
    R CMD INSTALL
    cd ../Python
    python setup.py install --user

If the `phater` folder is empty, you have may forgotten to use the `--recursive` option for `git clone`. You can rectify this by running the following in a terminal::

    cd PHATE
    git submodule init
    git submodule update
    cd phater
    R CMD INSTALL
    cd ../Python
    python setup.py install --user
