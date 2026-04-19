# navigation/sensors package — GPS readers and the build_gps() factory.

from .gps_reader import GPSReader, GPSStatus, GPSHealth
from .mock_gps import MockGPSReader


def build_gps(config):
    """Return a started GPS reader matching the runtime mode.

    --mock returns a deterministic MockGPSReader so the rest of the system
    can run on a desktop without a UART NMEA device attached.
    """
    if config.mock:
        gps = MockGPSReader()
    else:
        gps = GPSReader(
            port=config.gps_port,
            baudrate=config.gps_baudrate,
            warmup_sec=config.gps_warmup_sec,
        )
    gps.start()
    return gps
