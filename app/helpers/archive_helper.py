"""
Helper methods for the archive API routes.
"""

import logging
import os
from pathlib import Path
import tempfile
import io
import json

import numpy as np
from obspy import UTCDateTime, read_inventory
from obspy.clients.filesystem.sds import Client as SDSClient
from rpi_seism_common.settings import Settings

from app.exc.archive import ArchiveNotFound, InventoryNotFound, InvalidTimeFormat

logger = logging.getLogger(__name__)


RPI_SEISM_PATH = Path(os.getenv("RPI_SEISM_PATH", "/usr_data"))


def _load_config():
    config_path = RPI_SEISM_PATH / "config.yml"
    if not config_path.exists():
        logger.warning("Settings not found, using defaults")
        return Settings.get_default_settings()

    return Settings.load_settings(config_path)

_cfg = _load_config()


class ArchiveHelper:
    """Helper methods for the archive API routes."""
    SDS_ROOT  = RPI_SEISM_PATH / "archive"
    NETWORK  = _cfg.station.network
    STATION  = _cfg.station.station
    LOCATION = _cfg.station.location_code

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

    EXPORT_FORMATS: dict[str, tuple[str, str]] = {
        "mseed": ("application/octet-stream", "mseed"),
        "sac":   ("application/octet-stream", "sac"),
        "csv":   ("text/csv",                 "csv"),
        "json":  ("application/json",         "json"),
    }

    #  Cached inventory 
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
            "starttime":    tr.stats.starttime.isoformat() + "Z",
            "endtime":      tr.stats.endtime.isoformat() + "Z",
            "npts_raw":     tr.stats.npts,
            "npts_display": int(len(data)),
            "data":         data.tolist(),
        }
    
    @classmethod
    def _to_mseed(cls, st, **kwargs) -> bytes:
        buf = io.BytesIO()
        st.write(buf, format="MSEED", reclen=512)
        return buf.getvalue()

    @classmethod
    def _to_sac(cls, st, **kwargs) -> bytes:
        # Note: SAC is single-trace. st[0] is assumed here.
        with tempfile.NamedTemporaryFile(suffix=".sac", delete=True) as tmp:
            st[0].write(tmp.name, format="SAC")
            with open(tmp.name, "rb") as f:
                return f.read()

    @classmethod
    def _to_csv(cls, st, channel="", unit_label="", **kwargs) -> bytes:
        tr = st[0]
        t0, dt = tr.stats.starttime, tr.stats.delta
        buf = io.StringIO()
        buf.write(f"time_utc,{channel}_{unit_label}\n")
        # Vectorized string formatting is faster for large CSVs
        for i, v in enumerate(tr.data):
            buf.write(f"{(t0 + i * dt).isoformat()},{v}\n")
        return buf.getvalue().encode()

    @classmethod
    def _to_json(cls, st, channel="", unit_label="", **kwargs) -> bytes:
        data, factor = cls.peak_decimate(st[0].data, 4000)
        payload = {
            "channel": channel,
            "units": unit_label,
            "fs": st[0].stats.sampling_rate / factor,
            "data": data.tolist(),
            # ... add other metadata ...
        }
        return json.dumps(payload).encode()

    @classmethod
    def export_channel(cls, channel: str, start: str, end: str, units: str, fmt: str):
        t_start = cls.parse_time(start, "start")
        t_end   = cls.parse_time(end,   "end")

        client = cls.sds_client()
        try:
            st = client.get_waveforms(
                cls.NETWORK, cls.STATION, cls.LOCATION, channel,
                t_start, t_end,
            )
        except ValueError as exc:
            if "mmap" in str(exc).lower():
                raise Exception(f"No data for {channel} between {start} and {end} (empty archive file)") from exc
            raise Exception(f"Archive read error for {channel}: {exc}") from exc
        except Exception as exc:
            raise Exception(f"Archive read error for {channel}: {exc}") from exc

        if not st:
            raise Exception(f"No data for {channel} between {start} and {end}")

        st.merge(fill_value=0)

        if st[0].stats.npts > cls.MAX_SAMPLES:
            raise Exception(
                f"Trace contains {st[0].stats.npts:,} samples which exceeds the "
                f"{cls.MAX_SAMPLES:,} sample limit. Narrow the time window."
            )

        # 1. Fetch data
        st = client.get_waveforms(
                cls.NETWORK, cls.STATION, cls.LOCATION, channel,
                t_start, t_end,
            )
        
        # 2. Process data
        if units != "COUNTS":
            for tr in st: cls.deconvolve(tr, units)

        # 3. Format mapping
        formatters = {
            "mseed": cls._to_mseed,
            "sac":   cls._to_sac,
            "csv":   cls._to_csv,
            "json":  cls._to_json,
        }
        
        unit_label = cls.UNIT_LABELS[units].replace("/", "-")
        file_bytes = formatters[fmt](st, channel=channel, unit_label=unit_label)
        
        # Generate filename
        safe_start = start.replace(":", "-")[:16]
        filename = f"{cls.NETWORK}.{cls.STATION}.{channel}.{safe_start}.{fmt}"
        
        return file_bytes, filename

    # @classmethod
    # def export_channel(
    #     cls,
    #     channel: str,
    #     start: str,
    #     end: str,
    #     units: str,
    #     fmt: str,
    # ) -> tuple[bytes, str, str]:
    #     """
    #     Read and optionally deconvolve a channel, then serialise to the
    #     requested format.

    #     Returns (raw_bytes, mimetype, suggested_filename).
    #     """
    #     t_start = cls.parse_time(start, "start")
    #     t_end   = cls.parse_time(end,   "end")

    #     client = cls.sds_client()
    #     try:
    #         st = client.get_waveforms(
    #             cls.NETWORK, cls.STATION, cls.LOCATION, channel,
    #             t_start, t_end,
    #         )
    #     except ValueError as exc:
    #         if "mmap" in str(exc).lower():
    #             raise Exception(f"No data for {channel} between {start} and {end} (empty archive file)") from exc
    #         raise Exception(f"Archive read error for {channel}: {exc}") from exc
    #     except Exception as exc:
    #         raise Exception(f"Archive read error for {channel}: {exc}") from exc

    #     if not st:
    #         raise Exception(f"No data for {channel} between {start} and {end}")

    #     st.merge(fill_value=0)

    #     if st[0].stats.npts > cls.MAX_SAMPLES:
    #         raise Exception(
    #             f"Trace contains {st[0].stats.npts:,} samples which exceeds the "
    #             f"{cls.MAX_SAMPLES:,} sample limit. Narrow the time window."
    #         )

    #     if units != "COUNTS":
    #         for tr in st:
    #             cls.deconvolve(tr, units)

    #     unit_label = cls.UNIT_LABELS[units].replace("/", "-")
    #     safe_start = start.replace(":", "-").replace("T", "_")[:19]
    #     base_name  = f"{cls.NETWORK}.{cls.STATION}.{channel}.{safe_start}.{unit_label}"

    #     mimetype, ext = cls.EXPORT_FORMATS[fmt]
    #     filename = f"{base_name}.{ext}"

    #     if fmt == "mseed":
    #         buf = io.BytesIO()
    #         st.write(buf, format="MSEED", reclen=512)
    #         return buf.getvalue(), mimetype, filename

    #     if fmt == "sac":
    #         # ObsPy SAC writer requires a real filesystem path
    #         with tempfile.NamedTemporaryFile(suffix=".sac", delete=False) as tmp:
    #             tmp_path = tmp.name
    #         try:
    #             # One SAC file per trace — zip them if multi-trace after merge
    #             st[0].write(tmp_path, format="SAC")
    #             return Path(tmp_path).read_bytes(), mimetype, filename
    #         finally:
    #             Path(tmp_path).unlink(missing_ok=True)

    #     if fmt == "csv":
    #         tr = st[0]
    #         t0 = tr.stats.starttime
    #         dt = tr.stats.delta   # seconds between samples

    #         buf = io.StringIO()
    #         buf.write(f"time_utc,{channel}_{unit_label}\n")
    #         for i, v in enumerate(tr.data):
    #             ts = (t0 + i * dt).isoformat()
    #             buf.write(f"{ts},{v}\n")
    #         return buf.getvalue().encode(), mimetype, filename

    #     if fmt == "json":
    #         data, factor = cls.peak_decimate(st[0].data, 4000)
    #         display_fs   = st[0].stats.sampling_rate / factor
    #         payload = {
    #             "channel":      channel,
    #             "network":      cls.NETWORK,
    #             "station":      cls.STATION,
    #             "units":        unit_label,
    #             "fs":           display_fs,
    #             "starttime":    st[0].stats.starttime.isoformat(),
    #             "endtime":      st[0].stats.endtime.isoformat(),
    #             "npts_raw":     st[0].stats.npts,
    #             "npts_display": int(len(data)),
    #             "data":         data.tolist(),
    #         }
    #         return json.dumps(payload).encode(), mimetype, filename

    #     raise ValueError(f"Unsupported format: {fmt!r}")
