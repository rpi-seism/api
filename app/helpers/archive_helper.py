"""
Helper methods for the archive API routes.
"""

import logging
import os
from pathlib import Path
from io import BytesIO, StringIO
import json

import numpy as np
from obspy import UTCDateTime, read_inventory
from obspy.clients.filesystem.sds import Client as SDSClient
from obspy.io.sac.sactrace import SACTrace
from obspy.signal.trigger import ar_pick

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
    SDS_ROOT  = RPI_SEISM_PATH / "archive" / "sds"
    NETWORK  = _cfg.station.network
    STATION  = _cfg.station.station
    LOCATION = _cfg.station.location_code

    STATION_XML  = SDS_ROOT.parent.parent / "station.xml"

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

    COMPONENT_GEOMETRY: dict[str, tuple[float, float]] = {
        #  channel  : (cmpaz, cmpinc)
        #  cmpaz  — azimuth clockwise from North (deg)
        #  cmpinc — incidence from vertical (deg): 0 = up, 90 = horizontal
        "EHZ": (  0.0,   0.0),   # vertical
        "EHN": (  0.0,  90.0),   # horizontal, North
        "EHE": ( 90.0,  90.0),   # horizontal, East
        # aggiungi altri canali se necessario (HHZ, HHN, HHE, ...)
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
        buf = BytesIO()
        st.write(buf, format="MSEED", reclen=512)
        return buf.getvalue()

    @classmethod
    def _to_sac(cls, st, **kwargs) -> bytes:
        """
        Export a single-trace Stream to SAC binary format.
        SAC is inherently single-trace; only st[0] is processed.
        """
        tr = st[0].copy()

        sac = SACTrace.from_obspy_trace(tr)

        # Station metadata
        sac.stla   = _cfg.station.latitude
        sac.stlo   = _cfg.station.longitude
        sac.stel   = _cfg.station.elevation
        sac.stdp   = 0.0
        sac.kstnm  = cls.STATION[:8]
        sac.knetwk = cls.NETWORK[:8]
        sac.khole  = cls.LOCATION         # ← location code (era -12345)
        sac.kcmpnm = tr.stats.channel[:8] # ← channel name  (esplicito)
        sac.kevnm  = f"{cls.NETWORK}.{cls.STATION}"[:16]  # ← "Name" nel viewer

        # Component geometry
        az, inc = cls.COMPONENT_GEOMETRY.get(tr.stats.channel, (None, None))
        if az is not None:
            sac.cmpaz  = az
            sac.cmpinc = inc
        else:
            logger.warning(
                "No component geometry defined for channel %s — "
                "cmpaz/cmpinc will be -12345 in the SAC header.",
                tr.stats.channel,
            )

        # Event metadata (optional, passed via kwargs)
        if "origin" in kwargs:
            o = kwargs["origin"]
            sac.evla = o.latitude
            sac.evlo = o.longitude
            sac.evdp = o.depth / 1000.0          # SAC expects km
            sac.o    = o.time - tr.stats.starttime  # offset from trace start (s)
            sac.ko   = "OT"

        buf = BytesIO()
        sac.write(buf)
        return buf.getvalue()

    @classmethod
    def _to_csv(cls, st, channel="", unit_label="", **kwargs) -> bytes:
        tr = st[0]
        t0, dt = tr.stats.starttime, tr.stats.delta
        buf = StringIO()
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
        
        # Process data
        if units != "COUNTS":
            for tr in st: cls.deconvolve(tr, units)

        # Format mapping
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

    @classmethod
    def pick_p_onset(
        cls,
        start: str,
        end: str,
        f1: float = 1.0,
        f2: float = 20.0,
        lta_p: float = 1.0,
        sta_p: float = 0.1,
        lta_s: float = 4.0,
        sta_s: float = 1.0,
    ) -> dict:
        """
        Run AR-AIC picker on a channel to detect P-wave onset.
        
        Parameters
        ----------
        start : str
            ISO-8601 start time of search window
        end : str
            ISO-8601 end time of search window
        f1 : float
            Lower corner frequency for bandpass filter (Hz)
        f2 : float
            Upper corner frequency for bandpass filter (Hz)
        lta_p : float
            Long-term average window for P-wave (seconds)
        sta_p : float
            Short-term average window for P-wave (seconds)
        lta_s : float
            Long-term average window for S-wave (seconds)
        sta_s : float
            Short-term average window for S-wave (seconds)
            
        Returns
        -------
        dict
            Pick results containing:
            - p_pick: ISO-8601 timestamp of P-wave onset (or None)
            - s_pick: ISO-8601 timestamp of S-wave onset (or None)
            - channel: Channel codes
            - distance_estimation: Estimated distance in kilometers
            - search_window: Dictionary with start/end times
        """
        channel = "EH?"
        t_start = cls.parse_time(start, "start")
        t_end = cls.parse_time(end, "end")
        
        # Read raw counts for picking (don't deconvolve)
        client = cls.sds_client()
        
        try:
            st = client.get_waveforms(
                cls.NETWORK, cls.STATION, cls.LOCATION, channel,
                t_start, t_end,
            )
        except Exception as exc:
            logger.warning("SDS read failed for %s: %s", channel, exc)
            raise Exception(f"Archive read error for {channel}: {exc}") from exc
        
        if not st:
            raise Exception(f"No data for {channel} between {start} and {end}")
        
        st.merge(fill_value=0)
        
        # Preprocess: demean, taper, bandpass filter
        st.detrend("demean")
        st.taper(max_percentage=0.05)
        st.filter("bandpass", freqmin=f1, freqmax=f2)
    
        # Extract components
        try:
            # Map Z, N, E to a, b, c respectively
            tr_z = st.select(component="Z")[0]
            tr_n = st.select(component="N")[0]
            tr_e = st.select(component="E")[0]
            
            data_z = tr_z.data
            data_n = tr_n.data
            data_e = tr_e.data
            samp_rate = tr_z.stats.sampling_rate
        except (IndexError, KeyError):
            raise Exception("Stream must contain Z, N, and E components for AR picking.")
        
        # Run AR-AIC picker
        # Returns: (p_pick_sample, s_pick_sample, snr, slope)
        try:
            p_pick, s_pick = ar_pick(
                a=data_z,
                b=data_n,
                c=data_e,
                samp_rate=samp_rate,
                f1=f1,
                f2=f2,
                lta_p=lta_p,
                sta_p=sta_p,
                lta_s=lta_s,
                sta_s=sta_s,
                m_p=2,      # Number of AR coefficients for P
                m_s=8,      # Number of AR coefficients for S
                l_p=0.1,    # Length of P-coda window (s)
                l_s=0.2,    # Length of S-coda window (s)
            )
        except Exception as exc:
            logger.exception("AR-AIC picker failed for %s", channel)
            raise Exception(f"Picker failed for {channel}: {exc}") from exc
        
        # Convert relative seconds to absolute times
        p_time = None
        s_time = None
        dist_km = None
        
        # ar_pick returns seconds from tr.stats.starttime
        if p_pick > 0:
            p_dt = tr_z.stats.starttime + p_pick
            p_time = p_dt.isoformat() + "Z"
        
        if s_pick > 0:
            s_dt = tr_z.stats.starttime + s_pick
            s_time = s_dt.isoformat() + "Z"
            
        # Calculate distance only if both picks exist
        if p_time and s_time:
            s_p_diff = s_dt - p_dt  # ObsPy UTCDateTime subtraction returns seconds
            dist_km = s_p_diff * 8.0
        
        return {
            "network": cls.NETWORK,
            "station": cls.STATION,
            "channels_used": [tr.stats.channel for tr in st],
            "p_pick": p_time,
            "s_pick": s_time,
            "distance_estimation": dist_km,
            "search_window": {"start": start, "end": end}
        }
