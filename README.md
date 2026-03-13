# rpi-seism-api

HTTP archive browser for the [rpi-seism](https://github.com/rpi-seism/daemon) seismic monitoring system.

Serves historical waveform data from the SDS MiniSEED archive written by the acquisition daemon, with optional instrument response deconvolution using the station's `station.xml`.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service status and path diagnostics |
| `GET` | `/archive/channels` | List available channel codes |
| `GET` | `/archive/days` | List dates with data for a channel |
| `GET` | `/archive/waveform` | Fetch waveform data for a time window |
| `GET` | `/archive/events` | List day files for a channel and date |

Interactive API docs are available at `/docs` when the server is running.

---

## Configuration

The API reads station metadata directly from the `config.yml` written by the daemon. No manual configuration is needed if both services share the same data volume.


---

## Waveform endpoint

```
GET /archive/waveform?channel=EHZ&start=2025-03-23T10:00:00&end=2025-03-23T10:05:00
```

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `channel` | yes | — | Channel code, e.g. `EHZ` |
| `start` | yes | — | ISO-8601 start time |
| `end` | yes | — | ISO-8601 end time |
| `units` | no | `COUNTS` | `COUNTS` · `VEL` · `DISP` · `ACC` |
| `max_pts` | no | `4000` | Maximum display points (peak-preserving decimation) |

**Units**

| Value | Output | Requires |
|-------|--------|----------|
| `COUNTS` | Raw ADC integers | — |
| `VEL` | Velocity in nm/s | `station.xml` |
| `DISP` | Displacement in nm | `station.xml` |
| `ACC` | Acceleration in nm/s² | `station.xml` |

When using `VEL`, `DISP`, or `ACC`, the API deconvolves the instrument response using the PAZ + ADC gain stages from `station.xml`. A cosine pre-filter `(0.5, 1.0, 45.0, 48.0 Hz)` is applied to suppress amplification below the GD-4.5 natural frequency and above Nyquist.

**Response**

```json
{
  "channel":      "EHZ",
  "network":      "XX",
  "station":      "RPI3",
  "units":        "nm/s",
  "fs":           25.0,
  "starttime":    "2025-03-23T10:00:00.000000Z",
  "endtime":      "2025-03-23T10:05:00.000000Z",
  "npts_raw":     30000,
  "npts_display": 4000,
  "data":         [...]
}
```

`npts_display` may be less than `npts_raw` when peak-preserving decimation is applied. The displayed sample rate `fs` reflects the decimated rate.

---

## Running

**Development**

```bash
uv sync
uv run fastapi dev
```

**Docker** (standalone)

```bash
docker build -t rpi-seism-api .
docker run -p 8000:8000 \
  -v /path/to/data:/app/data:ro \
  rpi-seism-api
```

**Docker Compose** (as part of the full stack)

See [rpi-seism/stack](https://github.com/rpi-seism/stack) for the full deployment setup.

---

## Limits

| Constraint | Value |
|------------|-------|
| Maximum time window | 6 hours |
| Maximum raw samples | 200,000 |
| Maximum display points | 20,000 |

---

## Compatibility

| api | daemon | 
|-----|--------|
| v1.x | v1.x |

---

## License

[GNU General Public License v3.0](LICENSE)