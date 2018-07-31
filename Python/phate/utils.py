import numbers
import numpy as np


def check_positive(**params):
    """Check that parameters are positive as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Number) or params[p] <= 0:
            raise ValueError(
                "Expected {} > 0, got {}".format(p, params[p]))


def check_int(**params):
    """Check that parameters are integers as expected

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if not isinstance(params[p], numbers.Integral):
            raise ValueError(
                "Expected {} integer, got {}".format(p, params[p]))


def check_if_not(x, *checks, **params):
    """Run checks only if parameters are not equal to a specified value

    Parameters
    ----------

    x : excepted value
        Checks not run if parameters equal x

    checks : function
        Unnamed arguments, check functions to be run

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] is not x and params[p] != x:
            [check(**{p: params[p]}) for check in checks]


def check_in(choices, **params):
    """Checks parameters are in a list of allowed parameters

    Parameters
    ----------

    choices : array-like, accepted values

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] not in choices:
            raise ValueError(
                "{} value {} not recognized. Choose from {}".format(
                    p, params[p], choices))


def check_between(v_min, v_max, **params):
    """Checks parameters are in a specified range

    Parameters
    ----------

    v_min : float, minimum allowed value (inclusive)

    v_max : float, maximum allowed value (inclusive)

    params : object
        Named arguments, parameters to be checked

    Raises
    ------
    ValueError : unacceptable choice of parameters
    """
    for p in params:
        if params[p] < v_min or params[p] > v_max:
            raise ValueError("Expected {} between {} and {}, "
                             "got {}".format(p, v_min, v_max, params[p]))


def matrix_is_equivalent(X, Y):
    """
    Checks matrix equivalence with numpy, scipy and pandas
    """
    return X is Y or (isinstance(X, Y.__class__) and X.shape == Y.shape and
                      np.sum((X != Y).sum()) == 0)


def in_ipynb():
    """Check if we are running in a Jupyter Notebook

    Credit to https://stackoverflow.com/a/24937408/3996580
    """
    __VALID_NOTEBOOKS = ["<class 'google.colab._shell.Shell'>",
                         "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>"]
    try:
        return str(type(get_ipython())) in __VALID_NOTEBOOKS
    except NameError:
        return False
