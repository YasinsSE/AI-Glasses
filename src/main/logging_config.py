"""Centralised logging configuration for the ALAS main loop.

Every ALAS subsystem logs through ``logging.getLogger("ALAS.<area>")`` so each
line carries its source (e.g. ``ALAS.perception``). This module installs the
single shared formatter and also folds two other noise sources into the SAME
format:

  * **Python warnings** (e.g. Jetson.GPIO's "ignores pull_up_down") are routed
    through logging via ``captureWarnings`` so they appear as ``py.warnings``
    lines instead of raw ``UserWarning`` text.
  * **Native GStreamer/Argus chatter** (GST_ARGUS / OpenCV cap_gstreamer) is
    silenced at the source by ``GST_DEBUG=0`` (set in ``alas_main`` before the
    camera stack initialises) and by a brief stdout/stderr mute around the
    camera open in ``perception_service``.

Tip: ``journalctl -u alas-launcher -o cat`` drops systemd's own
``Haz 02 .. alas python[..]:`` prefix, leaving just the ALAS format.
"""

import logging

_LOG_FORMAT = "[%(asctime)s] %(name)-22s %(levelname)-7s %(message)s"
_DATE_FORMAT = "%H:%M:%S"


def configure_logging(level=logging.INFO):
    """Install a single root handler shared by every ALAS subsystem."""
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Route Python warnings (warnings.warn) through the same handler/format so a
    # stray UserWarning does not break the uniform log layout.
    logging.captureWarnings(True)
    logging.getLogger("py.warnings").setLevel(logging.WARNING)

    return logging.getLogger("ALAS")
