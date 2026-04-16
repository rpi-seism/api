import logging
import re
from io import BytesIO
import asyncio
from datetime import datetime
from typing import Annotated

import zipfile

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from app.helpers.archive_helper import ArchiveHelper


logger = logging.getLogger(__name__)

router = APIRouter()


def generate_zip(outcomes):
    """
    Generate a ZIP archive incrementally.
    Yields compressed data as each file is added.
    """
    # Use BytesIO as the ZIP destination
    buffer = BytesIO()

    # Create the ZIP entirely within the buffer
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for _, result, _ in outcomes:
            if result is not None:
                value, filename = result
                zf.writestr(filename, value)

    # Seek to start so we can read it out
    buffer.seek(0)

    # Yield the completed buffer
    yield buffer.getvalue()


@router.get("/health")
def health():
    """Health check endpoint to verify archive access and configuration."""
    return {
        "status":       "ok",
        "sds_root":     str(ArchiveHelper.SDS_ROOT),
        "sds_exists":   ArchiveHelper.SDS_ROOT.exists(),
        "station_xml":  str(ArchiveHelper.STATION_XML),
        "xml_exists":   ArchiveHelper.STATION_XML.exists(),
        "network":      ArchiveHelper.NETWORK,
        "station":      ArchiveHelper.STATION,
    }


@router.get("/channels", response_model=list[str])
def get_channels():
    """
    Scan the SDS tree and return all unique channel codes present
    for the configured network/station, e.g. ["EHE", "EHN", "EHZ"].
    """
    if not ArchiveHelper.SDS_ROOT.exists():
        raise HTTPException(status_code=503, detail=f"SDS root not found: {ArchiveHelper.SDS_ROOT}")

    channels: set[str] = set()
    ch_dir_pattern = re.compile(r"^[A-Z]{3}\.D$")

    for year_dir in ArchiveHelper.SDS_ROOT.iterdir():
        if not year_dir.is_dir():
            continue
        sta_dir = year_dir / ArchiveHelper.NETWORK / ArchiveHelper.STATION
        if not sta_dir.is_dir():
            continue
        for ch_dir in sta_dir.iterdir():
            if ch_dir.is_dir() and ch_dir_pattern.match(ch_dir.name):
                channels.add(ch_dir.name.split(".")[0])

    if not channels:
        raise HTTPException(status_code=404, detail="No channels found in archive")

    return sorted(channels)


@router.get("/days", response_model=list[str])
def get_days(
    channel: str = Query(..., description="Channel code, e.g. EHZ"),
):
    """
    Return sorted ISO dates (YYYY-MM-DD) for which at least one SDS day
    file exists for the requested channel.
    """
    days: set[str] = set()
    file_glob = f"{ArchiveHelper.NETWORK}.{ArchiveHelper.STATION}.{ArchiveHelper.LOCATION}.{channel}.D.*"

    for year_dir in ArchiveHelper.SDS_ROOT.iterdir():
        if not year_dir.is_dir():
            continue
        ch_dir = year_dir / ArchiveHelper.NETWORK / ArchiveHelper.STATION / f"{channel}.D"
        if not ch_dir.is_dir():
            continue
        for f in ch_dir.glob(file_glob):
            parts = f.name.split(".")
            if len(parts) < 7:
                continue
            try:
                dt = datetime.strptime(f"{parts[5]} {parts[6]}", "%Y %j")
                days.add(dt.date().isoformat())
            except ValueError:
                continue

    if not days:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for channel {channel!r}",
        )

    return sorted(days)


@router.get("/waveform")
def get_waveform(
    channel: str = Query(...,        description="Channel code, e.g. EHZ"),
    start:   str = Query(...,        description="ISO-8601 start time"),
    end:     str = Query(...,        description="ISO-8601 end time"),
    units:   str = Query("COUNTS",   description="COUNTS | VEL | DISP | ACC"),
    max_pts: int = Query(4000,       description="Max display points", ge=100, le=20000),
):
    """Single-channel waveform. Kept for backwards compatibility."""
    if units not in ArchiveHelper.UNIT_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid units {units!r}. Must be one of: {list(ArchiveHelper.UNIT_LABELS)}",
        )

    t_start = ArchiveHelper.parse_time(start, "start")
    t_end   = ArchiveHelper.parse_time(end,   "end")

    if t_end <= t_start:
        raise HTTPException(status_code=400, detail="end must be after start")

    window_h = (t_end - t_start) / 3600
    if window_h > ArchiveHelper.MAX_WINDOW_H:
        raise HTTPException(
            status_code=400,
            detail=f"Window too large ({window_h:.1f} h). Maximum is {ArchiveHelper.MAX_WINDOW_H} h.",
        )

    logger.info("Serving %s %s [%s - %s] units=%s", ArchiveHelper.STATION, channel, start, end, units)

    try:
        return ArchiveHelper.read_channel(channel, start, end, units, max_pts)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/waveforms")
async def get_waveforms(
    channels: Annotated[list[str], Query(description="Channel codes, repeat param: ?channels=EHZ&channels=EHN")],
    start:    str = Query(...,       description="ISO-8601 start time"),
    end:      str = Query(...,       description="ISO-8601 end time"),
    units:    str = Query("COUNTS",  description="COUNTS | VEL | DISP | ACC"),
    max_pts:  int = Query(4000,      description="Max display points per channel", ge=100, le=20000),
):
    """
    Multi-channel waveform fetch. All channels are read concurrently —
    latency is bounded by the slowest channel, not the sum of all channels.
    Channels with no data are reported in 'errors' without aborting the request.
    """
    if not channels:
        raise HTTPException(status_code=400, detail="At least one channel is required")

    if units not in ArchiveHelper.UNIT_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid units {units!r}. Must be one of: {list(ArchiveHelper.UNIT_LABELS)}",
        )

    t_start = ArchiveHelper.parse_time(start, "start")
    t_end   = ArchiveHelper.parse_time(end,   "end")

    if t_end <= t_start:
        raise HTTPException(status_code=400, detail="end must be after start")

    window_h = (t_end - t_start) / 3600
    if window_h > ArchiveHelper.MAX_WINDOW_H:
        raise HTTPException(
            status_code=400,
            detail=f"Window too large ({window_h:.1f} h). Maximum is {ArchiveHelper.MAX_WINDOW_H} h.",
        )

    async def _fetch(ch: str) -> tuple[str, dict | None, str | None]:
        """
        Run one blocking SDS read in a thread-pool worker.
        Returns (channel, result_or_None, error_detail_or_None).
        Each call gets its own SDSClient instance inside read_channel,
        so there is no shared ObsPy state between concurrent tasks.
        """
        logger.info("Serving %s %s [%s - %s] units=%s", ArchiveHelper.STATION, ch, start, end, units)
        try:
            result = await asyncio.to_thread(
                ArchiveHelper.read_channel, ch, start, end, units, max_pts
            )
            return ch, result, None
        except Exception as exc:
            return ch, None, str(exc)

    outcomes = await asyncio.gather(*(_fetch(ch) for ch in channels))

    results, errors = [], []
    for ch, result, detail in outcomes:
        if result is not None:
            results.append(result)
        else:
            errors.append({"channel": ch, "detail": detail})

    return {"results": results, "errors": errors}

@router.get("/export_waveforms")
async def export_waveforms(
    channels: Annotated[list[str], Query(description="Channel codes, repeat param: ?channels=EHZ&channels=EHN")],
    start:      str = Query(...,       description="ISO-8601 start time"),
    end:        str = Query(...,       description="ISO-8601 end time"),
    units:      str = Query("COUNTS",  description="COUNTS | VEL | DISP | ACC"),
    fmt:        str = Query("mseed",   description="mseed | sac | csv | json")
):
    """
    Export a waveform segment in the requested file format.
    The same instrument response deconvolution available on /waveform
    is applied here, so you can export calibrated velocity, displacement,
    or acceleration traces directly.

    Supported formats:
    - mseed  — MiniSEED (trimmed to the requested window, not the full day file)
    - sac    — SAC binary (first trace only after merge)
    - csv    — Plain CSV: time_utc, value; one row per sample, no decimation
    - json   — Same payload shape as /waveform, with peak-preserving decimation
    """
    if not channels:
        raise HTTPException(status_code=400, detail="At least one channel is required")

    if units not in ArchiveHelper.UNIT_LABELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid units {units!r}. Must be one of: {list(ArchiveHelper.UNIT_LABELS)}",
        )

    t_start = ArchiveHelper.parse_time(start, "start")
    t_end   = ArchiveHelper.parse_time(end,   "end")

    if t_end <= t_start:
        raise HTTPException(status_code=400, detail="end must be after start")

    window_h = (t_end - t_start) / 3600
    if window_h > ArchiveHelper.MAX_WINDOW_H:
        raise HTTPException(
            status_code=400,
            detail=f"Window too large ({window_h:.1f} h). Maximum is {ArchiveHelper.MAX_WINDOW_H} h.",
        )

    fmt = fmt.lower()
    if fmt not in ArchiveHelper.EXPORT_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format {fmt!r}. Must be one of: {list(ArchiveHelper.EXPORT_FORMATS)}",
        )

    async def _fetch(ch: str) -> tuple[str, dict | None, str | None]:
        """
        Run one blocking SDS read in a thread-pool worker.
        Returns (channel, result_or_None, error_detail_or_None).
        Each call gets its own SDSClient instance inside export_channel,
        so there is no shared ObsPy state between concurrent tasks.
        """
        logger.info("Serving %s %s [%s - %s] units=%s", ArchiveHelper.STATION, ch, start, end, units)
        try:
            result = await asyncio.to_thread(
                ArchiveHelper.export_channel, ch, start, end, units, fmt
            )
            return ch, result, None
        except Exception as exc:
            return ch, None, str(exc)

    outcomes = await asyncio.gather(*(_fetch(ch) for ch in channels))

    file_name = f"{start}-{end}.{"-".join(channels)}.{units}.{fmt}"

    return StreamingResponse(
        generate_zip([i for i in outcomes]),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={file_name}.zip"},
    )

@router.get("/events", response_model=list[dict])
def get_events(
    channel: str = Query(...,  description="Channel code, e.g. EHZ"),
    date:    str = Query(...,  description="ISO date, e.g. 2025-03-23"),
    limit:   int = Query(100,  description="Maximum records to return", ge=1, le=1000),
):
    """Return SDS day file records for a specific channel and date."""
    if not ArchiveHelper.SDS_ROOT.exists():
        raise HTTPException(status_code=503, detail=f"SDS root not found: {ArchiveHelper.SDS_ROOT}")

    try:
        dt  = datetime.strptime(date, "%Y-%m-%d")
        doy = dt.timetuple().tm_yday
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {date!r}. Expected YYYY-MM-DD") from exc

    file_re = re.compile(
        rf"^{re.escape(ArchiveHelper.NETWORK)}\."
        rf"{re.escape(ArchiveHelper.STATION)}\."
        rf"{re.escape(ArchiveHelper.LOCATION)}\."
        rf"{re.escape(channel)}\.D\."
        rf"{dt.year}\.{doy:03d}$"
    )

    records: list[dict] = []

    year_dir = ArchiveHelper.SDS_ROOT / str(dt.year)
    ch_dir   = year_dir / ArchiveHelper.NETWORK / ArchiveHelper.STATION / f"{channel}.D"

    if ch_dir.is_dir():
        for f in ch_dir.iterdir():
            if file_re.match(f.name):
                records.append({
                    "date":     date,
                    "channel":  channel,
                    "filename": f.name,
                    "size_kb":  round(f.stat().st_size / 1024, 1),
                })

    records.sort(key=lambda r: r["filename"])
    return records[:limit]


@router.get("/download")
def download_mseed(
    channel: str = Query(..., description="Channel code, e.g. EHZ"),
    date:    str = Query(..., description="ISO date, e.g. 2025-03-23"),
):
    """Serve the raw MiniSEED day file for download."""
    try:
        dt  = datetime.strptime(date, "%Y-%m-%d")
        day = dt.timetuple().tm_yday
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid date: {date!r}") from exc

    filename = f"{ArchiveHelper.NETWORK}.{ArchiveHelper.STATION}.{ArchiveHelper.LOCATION}.{channel}.D.{dt.year}.{day:03d}"
    path = (
        ArchiveHelper.SDS_ROOT
        / str(dt.year)
        / ArchiveHelper.NETWORK
        / ArchiveHelper.STATION
        / f"{channel}.D"
        / filename
    )

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")

    return FileResponse(
        path=str(path),
        media_type="application/octet-stream",
        filename=filename,
    )

@router.get("/pick")
async def pick_p_onset(
    start:   str = Query(...,   description="ISO-8601 start time of search window"),
    end:     str = Query(...,   description="ISO-8601 end time of search window"),
    f1:      float = Query(1.0,   description="Bandpass lower frequency (Hz)", ge=0.1, le=50.0),
    f2:      float = Query(20.0,  description="Bandpass upper frequency (Hz)", ge=0.1, le=50.0),
    lta_p:   float = Query(1.0,   description="P-wave long-term average window (s)", ge=0.1, le=10.0),
    sta_p:   float = Query(0.1,   description="P-wave short-term average window (s)", ge=0.01, le=5.0),
    lta_s:   float = Query(4.0,   description="S-wave long-term average window (s)", ge=0.1, le=10.0),
    sta_s:   float = Query(1.0,   description="S-wave short-term average window (s)", ge=0.01, le=5.0),
):
    """
    Run AR-AIC picker to detect P-wave and S-wave onset times.
    
    The picker uses an autoregressive (AR) model combined with the Akaike
    Information Criterion (AIC) to identify phase arrivals in seismic data.
    
    Typical usage:
    - Set a search window around the expected arrival time
    - Use EHZ channel for P-wave detection
    - Adjust frequency band to match expected signal content
    - Tune STA/LTA windows based on expected onset sharpness
    
    Returns:
    - p_pick: ISO timestamp of P-wave onset (None if not detected)
    - s_pick: ISO timestamp of S-wave onset (None if not detected)
    - snr: Signal-to-noise ratio
    - picker_params: Parameters used for picking
    """
    if f2 <= f1:
        raise HTTPException(
            status_code=400,
            detail=f"Upper frequency f2 ({f2}) must be greater than lower frequency f1 ({f1})"
        )
    
    t_start = ArchiveHelper.parse_time(start, "start")
    t_end   = ArchiveHelper.parse_time(end, "end")
    
    if t_end <= t_start:
        raise HTTPException(status_code=400, detail="end must be after start")
    
    window_h = (t_end - t_start) / 3600
    if window_h > ArchiveHelper.MAX_WINDOW_H:
        raise HTTPException(
            status_code=400,
            detail=f"Window too large ({window_h:.1f} h). Maximum is {ArchiveHelper.MAX_WINDOW_H} h.",
        )
    
    logger.info(
        "Running picker on %s [%s - %s] f1=%.1f f2=%.1f",
        ArchiveHelper.STATION, start, end, f1, f2
    )
    
    try:
        result = await asyncio.to_thread(
            ArchiveHelper.pick_p_onset,
            start, end, f1, f2, lta_p, sta_p, lta_s, sta_s
        )
        return result
    except Exception as exc:
        logger.exception("Picker failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

# Optional: Simpler AIC-only endpoint (no AR model)
@router.get("/pick_simple")
async def pick_simple_aic(
    channel: str = Query("EHZ", description="Channel code for picking"),
    start:   str = Query(...,   description="ISO-8601 start time"),
    end:     str = Query(...,   description="ISO-8601 end time"),
):
    """
    Simple AIC picker (alternative to AR-AIC).
    
    Uses only the Akaike Information Criterion on the raw waveform
    without AR modeling. Faster but potentially less accurate than AR-AIC.
    """
    from obspy.signal.trigger import aic_simple
    import numpy as np
    
    t_start = ArchiveHelper.parse_time(start, "start")
    t_end   = ArchiveHelper.parse_time(end, "end")
    
    if t_end <= t_start:
        raise HTTPException(status_code=400, detail="end must be after start")
    
    # Read and preprocess data
    client = ArchiveHelper.sds_client()
    st = client.get_waveforms(
        ArchiveHelper.NETWORK, ArchiveHelper.STATION,
        ArchiveHelper.LOCATION, channel,
        t_start, t_end,
    )
    
    if not st:
        raise HTTPException(
            status_code=404,
            detail=f"No data for {channel} between {start} and {end}"
        )
    
    st.merge(fill_value=0)
    tr = st[0]
    
    # Preprocess
    tr.detrend("demean")
    tr.taper(max_percentage=0.05)
    tr.filter("bandpass", freqmin=1.0, freqmax=20.0, corners=2, zerophase=True)
    
    # Run simple AIC
    aic = aic_simple(tr.data)
    pick_sample = np.argmin(aic)
    pick_time = (tr.stats.starttime + pick_sample / tr.stats.sampling_rate).isoformat() + "Z"
    
    return {
        "channel": channel,
        "network": ArchiveHelper.NETWORK,
        "station": ArchiveHelper.STATION,
        "p_pick": pick_time,
        "search_window": {
            "start": start,
            "end": end,
        },
        "method": "aic_simple"
    }