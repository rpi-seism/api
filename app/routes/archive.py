from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

import logging
import re
from datetime import datetime
from typing import Annotated

from app.helpers.archive_helper import ArchiveHelper


logger = logging.getLogger(__name__)

router = APIRouter()


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
def get_waveforms(
    channels: Annotated[list[str], Query(description="Channel codes, repeat param: ?channels=EHZ&channels=EHN")],
    start:    str = Query(...,       description="ISO-8601 start time"),
    end:      str = Query(...,       description="ISO-8601 end time"),
    units:    str = Query("COUNTS",  description="COUNTS | VEL | DISP | ACC"),
    max_pts:  int = Query(4000,      description="Max display points per channel", ge=100, le=20000),
):
    """
    Multi-channel waveform fetch. Returns one result object per channel.
    Channels that have no data for the requested window are reported in
    the 'errors' field instead of aborting the whole request.

    Response shape:
    {
        "results": [ { channel, network, station, units, fs, starttime,
                       endtime, npts_raw, npts_display, data }, ... ],
        "errors":  [ { channel, detail }, ... ]
    }
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

    results = []
    errors  = []

    for ch in channels:
        try:
            logger.info("Serving %s %s [%s - %s] units=%s", ArchiveHelper.STATION, ch, start, end, units)
            results.append(ArchiveHelper.read_channel(ch, start, end, units, max_pts))
        except HTTPException as exc:
            errors.append({"channel": ch, "detail": exc.detail})
        except Exception as exc:
            errors.append({"channel": ch, "detail": str(exc)})

    return {"results": results, "errors": errors}


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
