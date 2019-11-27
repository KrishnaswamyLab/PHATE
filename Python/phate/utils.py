import graphtools
from deprecated import deprecated


@deprecated(version="0.5.0", reason="Use instead graphtools.utils.check_positive")
def check_positive(*args, **kwargs):
    return graphtools.utils.check_positive(*args, **kwargs)


@deprecated(version="0.5.0", reason="Use instead graphtools.utils.check_int")
def check_int(*args, **kwargs):
    return graphtools.utils.check_int(*args, **kwargs)


@deprecated(version="0.5.0", reason="Use instead graphtools.utils.check_if_not")
def check_if_not(*args, **kwargs):
    return graphtools.utils.check_if_not(*args, **kwargs)


@deprecated(version="0.5.0", reason="Use instead graphtools.utils.check_in")
def check_in(*args, **kwargs):
    return graphtools.utils.check_in(*args, **kwargs)


@deprecated(version="0.5.0", reason="Use instead graphtools.utils.check_between")
def check_between(*args, **kwargs):
    return graphtools.utils.check_between(*args, **kwargs)


@deprecated(version="0.5.0", reason="Use instead graphtools.utils.matrix_is_equivalent")
def matrix_is_equivalent(*args, **kwargs):
    return graphtools.utils.matrix_is_equivalent(*args, **kwargs)


def in_ipynb():
    """Check if we are running in a Jupyter Notebook

    Credit to https://stackoverflow.com/a/24937408/3996580
    """
    __VALID_NOTEBOOKS = [
        "<class 'google.colab._shell.Shell'>",
        "<class 'ipykernel.zmqshell.ZMQInteractiveShell'>",
    ]
    try:
        get_ipython
    except NameError:
        return False
    else:
        return str(type(get_ipython())) in __VALID_NOTEBOOKS
