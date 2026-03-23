"""
Helper methods for the archive API routes.
"""

import logging
import os
from pathlib import Path

import numpy as np
from obspy import UTCDateTime, read_inventory
from obspy.clients.filesystem.sds import Client as SDSClient
import yaml

from app.exc.archive import ArchiveNotFound, InventoryNotFound, InvalidTimeFormat

logger = logging.getLogger(__name__)


RPI_SEISM_PATH = Path(os.getenv("RPI_SEISM_PATH", "/usr_data"))


def _load_config():
    config_path = RPI_SEISM_PATH / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

_cfg = _load_config()


class ArchiveHelper:
    """Helper methods for the archive API routes."""
    SDS_ROOT  = RPI_SEISM_PATH / "archive"
    NETWORK  = _cfg.get("station", {}).get("network",  "XX")
    STATION  = _cfg.get("station", {}).get("station",  "RPI3")
    LOCATION = _cfg.get("station", {}).get("location", "00")

    STATION_XML  = SDS_ROOT.parent / "station.xml"

    # Hard cap — prevents OOM on absurdly large requests
    MAX_SAMPLES  = 200_000
    # Maximum requestable window
    MAX_WINDOW_H = 6

    # Pre-filter corners (Hz) applied before deconvolution to suppress
    # low-frequency blow-up below the GD-4.5 natural frequency and
    # high-frequency noise above Nyquist.
    # (f1, f2, f3, f4) — cosine taper: zero below f1, full pass f2–f3, zero above f4
    PRE_FILT = (0.5, 1.0, 45.0, 48.0)

    # Stabilises division near the zeros of the response (higher = more stable,
    # slightly less accurate at the edges of the pass band)
    WATER_LEVEL = 60  # dB

    # Output unit labels sent back to the frontend
    UNIT_LABELS: dict[str, str] = {
        "COUNTS": "counts",
        "VEL":    "nm/s",
        "DISP":   "nm",
        "ACC":    "nm/s²",
    }

    # ── Cached inventory ──────────────────────────────────────────────────────────
    # Loaded once on first deconvolution request and kept in memory.
    # Re-read from disk if station.xml is newer than the cached version
    # (handles epoch updates without restarting the API).

    _inventory_cache: dict = {"inv": None, "mtime": 0.0}

    @classmethod
    def get_inventory(cls):
        """Load and return the station inventory from station.xml, with caching."""
        if not cls.STATION_XML.exists():
            raise InventoryNotFound(
                f"station.xml not found at {cls.STATION_XML}. "
                "Start the rpi-seism daemon at least once to generate it, "
                "or use units=COUNTS for raw data."
            )
        mtime = cls.STATION_XML.stat().st_mtime
        if cls._inventory_cache["inv"] is None or mtime > cls._inventory_cache["mtime"]:
            logger.info("Loading inventory from %s", cls.STATION_XML)
            cls._inventory_cache["inv"]   = read_inventory(str(cls.STATION_XML))
            cls._inventory_cache["mtime"] = mtime
        return cls._inventory_cache["inv"]

    @classmethod
    def sds_client(cls) -> SDSClient:
        """Create and return an SDSClient for reading from the archive."""
        if not cls.SDS_ROOT.exists():
            raise ArchiveNotFound(
                f"SDS archive root not found: {cls.SDS_ROOT}",
            )
        return SDSClient(str(cls.SDS_ROOT))

    @classmethod
    def parse_time(cls, value: str, label: str) -> UTCDateTime:
        """Parse an ISO-8601 time string and return a UTCDateTime."""
        try:
            return UTCDateTime(value)
        except Exception as ex:
            raise InvalidTimeFormat(
                f"Invalid {label} — expected ISO-8601, got: {value!r}",
            ) from ex

    @classmethod
    def peak_decimate(cls, data: np.ndarray, target: int) -> tuple[np.ndarray, int]:
        """
        Downsample *data* to at most *target* points using peak-preserving
        decimation: keep the sample with the largest absolute value in each chunk.

        This is intentionally NOT a low-pass filter decimate — for display
        purposes we want earthquake peaks to survive, not be smoothed away.

        Returns (decimated_data, decimation_factor).
        """
        if len(data) <= target:
            return data, 1

        factor = int(len(data) / target)
        trim   = (len(data) // factor) * factor
        chunks = data[:trim].reshape(-1, factor)
        idx    = np.argmax(np.abs(chunks), axis=1)
        return chunks[np.arange(len(chunks)), idx], factor

    @classmethod
    def deconvolve(cls, trace, units: str):
        """
        Remove instrument response from *trace* in-place.

        Steps:
        1. Demean  — removes DC offset that would blow up after deconvolution
        2. Taper   — 5 % cosine taper suppresses Gibbs ringing at the edges
        3. remove_response — divides by the PAZ + ADC transfer function in the
            frequency domain, applying a cosine pre-filter to suppress frequencies
            outside the flat region of the GD-4.5 response

        After removal, data is converted from SI units (m, m/s, m/s²) to
        nanometres for display — avoids tiny floats on the Y axis.
        """
        inv = cls.get_inventory()

        trace.detrend("demean")
        trace.taper(max_percentage=0.05, type="cosine")
        trace.remove_response(
            inventory=inv,
            output=units,
            pre_filt=cls.PRE_FILT,
            water_level=cls.WATER_LEVEL,
        )

        # SI → nm  (1 m = 1e9 nm)
        trace.data = trace.data * 1e9

    @classmethod
    def read_channel(
        cls,
        channel: str,
        start: str,
        end: str,
        units: str,
        max_pts: int,
    ) -> dict:
        """
        Read and process a single channel from SDS. Creates a fresh SDS client
        per call so this function is safe to call from multiple threads or
        back-to-back without shared state issues.
        """
        t_start = ArchiveHelper.parse_time(start, "start")
        t_end   = ArchiveHelper.parse_time(end,   "end")

        # Each call gets its own client instance — no shared ObsPy state
        client = ArchiveHelper.sds_client()

        try:
            st = client.get_waveforms(
                ArchiveHelper.NETWORK, ArchiveHelper.STATION,
                ArchiveHelper.LOCATION, channel,
                t_start, t_end,
            )
        except Exception as exc:
            logger.warning("SDS read failed for %s: %s", channel, exc)
            raise Exception(f"Archive read error for {channel}: {exc}") from exc

        if not st:
            raise Exception(f"No data for {channel} between {start} and {end}")

        st.merge(fill_value=0)
        tr = st[0]

        if tr.stats.npts > ArchiveHelper.MAX_SAMPLES:
            raise Exception(
                 f"Trace contains {tr.stats.npts:,} samples which exceeds the "
                 f"{ArchiveHelper.MAX_SAMPLES:,} sample limit. Narrow the time window."
            )

        if units != "COUNTS":
            try:
                ArchiveHelper.deconvolve(tr, units)
            except Exception as exc:
                logger.exception("Deconvolution failed for %s", channel)
                raise Exception(f"Deconvolution failed for {channel}: {exc}",) from exc

        data, factor = ArchiveHelper.peak_decimate(tr.data, max_pts)
        display_fs   = tr.stats.sampling_rate / factor

        return {
            "channel":      channel,
            "network":      ArchiveHelper.NETWORK,
            "station":      ArchiveHelper.STATION,
            "units":        ArchiveHelper.UNIT_LABELS[units],
            "fs":           display_fs,
            "starttime":    tr.stats.starttime.isoformat(),
            "endtime":      tr.stats.endtime.isoformat(),
            "npts_raw":     tr.stats.npts,
            "npts_display": int(len(data)),
            "data":         data.tolist(),
        }
