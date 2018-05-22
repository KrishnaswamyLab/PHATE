from __future__ import absolute_import
from builtins import super, bytes
import os
import logging
import time


class RSafeStdErr(object):
    """
    R's reticulate package inadvertently captures stderr and stdout
    This class writes directly to stderr to avoid this.
    """

    def write(self, msg):
        os.write(2, bytes(msg, 'utf8'))

    def flush(self):
        pass


class TaskLogger(object):
    """
    Class which deals with timing and logging tasks
    """

    def __init__(self, logger, *args, **kwargs):
        self.tasks = {}
        self.logger = logger
        super().__init__(*args, **kwargs)

    def start_task(self, name):
        self.tasks[name] = time.time()
        self.logger.info("Calculating {}...".format(name))

    def complete_task(self, name):
        try:
            self.logger.info("Calculated {} in {:.2f} seconds.".format(
                name, time.time() - self.tasks[name]))
            del self.tasks[name]
        except KeyError:
            self.logger.info("Calculated {}.".format(name))


def set_logging(level=1):
    """Set up logging

    Parameters
    ----------

    level : `int` or `bool` (optional, default: 1)
        If False or 0, prints WARNING and higher messages.
        If True or 1, prints INFO and higher messages.
        If 2 or higher, prints all messages.
    """
    if level is True or level == 1:
        level = logging.INFO
    elif level is False or level <= 0:
        level = logging.WARNING
    elif level >= 2:
        level = logging.DEBUG

    logger = get_logger()
    logger.task_logger = TaskLogger(logger)
    logger.setLevel(level)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(stream=RSafeStdErr())
        handler.setFormatter(logging.Formatter(fmt='%(message)s'))
        logger.addHandler(handler)


def get_logger():
    return logging.getLogger("PHATE")


def get_task_logger():
    return get_logger().task_logger


def log_start(name):
    """
    Convenience function to log a task in the default
    TaskLogger
    """
    try:
        get_task_logger().start_task(name)
    except AttributeError:
        if not hasattr(logging.getLogger, "task_logger"):
            set_logging(logging.INFO)
            log_start(name)
        else:
            raise


def log_complete(name):
    """
    Convenience function to log a task in the default
    TaskLogger
    """
    try:
        get_task_logger().complete_task(name)
    except AttributeError:
        if not hasattr(logging.getLogger, "task_logger"):
            set_logging(logging.INFO)
            log_complete(name)
        else:
            raise


def log_debug(msg):
    """
    Convenience function to log a message to the default Logger
    """
    get_logger().debug(msg)


def log_info(msg):
    """
    Convenience function to log a message to the default Logger
    """
    get_logger().info(msg)


def log_warning(msg):
    """
    Convenience function to log a message to the default Logger
    """
    get_logger().warning(msg)


def log_error(msg):
    """
    Convenience function to log a message to the default Logger
    """
    get_logger().error(msg)


def log_critical(msg):
    """
    Convenience function to log a message to the default Logger
    """
    get_logger().critical(msg)
