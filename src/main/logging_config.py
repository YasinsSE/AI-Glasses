"""Centralised logging configuration for the ALAS main loop."""

import logging

_LOG_FORMAT = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level=logging.INFO):
    """Install a single root handler shared by every ALAS subsystem."""
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)
    return logging.getLogger("ALAS")
