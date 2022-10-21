"""Module for defining a logger."""

import os
import logging
import sys

def setup_logger(name = __name__, log_filepath = None):
    """Sets up a Python logging instance.

    Args:
        name: String of logger name
        log_filepath: String of path to log file.

    Returns:
        Python logger instance.
    """

    log_dir = os.path.dirname(log_filepath)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if log_filepath is None:
        raise ValueError('Please specify a log filepath.')

    # Create a custom logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    if not logger.hasHandlers():

        # Create handlers
        c_handler = logging.StreamHandler(sys.stdout)
        f_handler = logging.FileHandler(log_filepath)

        # Add formatters
        c_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger