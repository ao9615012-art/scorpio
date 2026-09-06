"""
SCORPIO – OLR from ECMWF IFS (European model) – Open Data
=========================================================
Source : ECMWF Open Data (CC-BY-4.0), IFS HRES 0.25°, 4 cycles/day
         primary  : https://data.ecmwf.int/forecasts/ (last ~4 days)
         archive  : https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/ (2023 →)
Field  : ttr  = top net thermal radiation, accumulated J/m² since forecast start.
         OLR (W/m²) = -ttr(6h) / 21600 → 0–6 h mean from each cycle (00/06/12/18 UTC).
         Daily mean OLR = mean of the 4 cycles → a full 24 h coverage.
Only the ttr message (~1.3 MB) is fetched via HTTP byte-range using the .index file.
Fields are regridded (box-average) to 1°×1°, ±30°, cached as data/ecmwf_olr/YYYYMMDD.npy
"""
from __future__ import annotations
import os, json, time, datetime as dt, tempfile, warnings
import numpy as np


def _ensure_cfgrib():
    """cfgrib/eccodes may be missing in ephemeral environments – install on demand."""
    try:
        import cfgrib  # noqa
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cfgrib", "eccodes"], check=False)
        import importlib; importlib.invalidate_caches()
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "data", "ecmwf_olr")
os.makedirs(CACHE, exist_ok=True)

ROOTS = ("https://data.ecmwf.int/forecasts",
         "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com")
PATH = "{root}/{ymd}/{hh}z/ifs/0p25/oper/{ymd}{hh}0000-{step}h-oper-fc"
CYCLES = ("00", "06", "12", "18")
STEP = 6
LAT = np.arange(-30, 31, 1.0)
LON = np.arange(0, 360, 1.0)
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "SCORPIO-tropical-waves/1.0"


def _get(url, tries=8, **kw):
    """GET with exponential back-off (S3 returns 503 SlowDown under load)."""
    for i in range(tries):
        try:
            r = SESSION.get(url, timeout=120, **kw)
            if r.status_code in (200, 206):
                return r
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(min(60, 2.0 * 2 ** i))
    return None


def _grib_to_1deg(raw: bytes) -> np.ndarray:
    import xarray as xr
    _ensure_cfgrib()
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(raw); fn = f.name
    try:
        ds = xr.open_dataset(fn, engine="cfgrib", backend_kwargs={"indexpath": ""})
        da = ds["ttr"]
        olr = -da.values / (STEP * 3600.0)                    # J/m² → W/m² (positive up)
        latg = da.latitude.values; long = da.longitude.values  # lat 90..-90, lon -180..179.75 or 0..359.75
        # --- longitude to 0..359.75 ascending
        long = np.round(long % 360, 3)
        order = np.argsort(long); long = long[order]; olr = olr[:, order]
        # --- latitude descending → ascending
        if latg[0] > latg[-1]:
            latg = latg[::-1]; olr = olr[::-1]
        # rows centred on integer lat: la-0.375..la+0.375 (4 rows of 0.25°)
        i0 = int(np.argmin(np.abs(latg - (LAT[0] - 0.375))))
        sub = olr[i0:i0 + 4 * len(LAT)]
        sub = sub.reshape(len(LAT), 4, -1).mean(1)                              # (61, 1440)
        # columns centred on integer lon: lo-0.375..lo+0.375 → roll by +1 cell so block starts at lo-0.375
        sub = np.roll(sub, 1, axis=1)
        out = sub.reshape(len(LAT), len(LON), 4).mean(2).astype(np.float32)
        return out
    finally:
        try: os.remove(fn)
        except OSError: pass


def fetch_cycle(day: dt.date, hh: str) -> np.ndarray | None:
    ymd = day.strftime("%Y%m%d")
    for root in ROOTS:
        base = PATH.format(root=root, ymd=ymd, hh=hh, step=STEP)
        r = _get(base + ".index")
        if r is None:
            continue
        entry = None
        for line in r.text.splitlines():
            if '"ttr"' in line:
                try:
                    entry = json.loads(line); break
                except json.JSONDecodeError:
                    pass
        if entry is None:
            continue
        s, n = entry["_offset"], entry["_length"]
        g = _get(base + ".grib2", headers={"Range": f"bytes={s}-{s + n - 1}"})
        if g is None:
            continue
        try:
            return _grib_to_1deg(g.content)
        except Exception:
            continue
    return None


def fetch_day(day: dt.date, min_cycles=3) -> np.ndarray | None:
    f = os.path.join(CACHE, day.strftime("%Y%m%d") + ".npy")
    if os.path.exists(f):
        return np.load(f)
    fields = [x for x in (fetch_cycle(day, hh) for hh in CYCLES) if x is not None]
    if len(fields) < min_cycles:
        return None
    arr = np.mean(fields, 0).astype(np.float32)
    np.save(f, arr)
    return arr


def available_days():
    return sorted(dt.datetime.strptime(x[:8], "%Y%m%d").date()
                  for x in os.listdir(CACHE) if x.endswith(".npy"))


def update(start: dt.date, end: dt.date | None = None, log=None, workers=2):
    from concurrent.futures import ThreadPoolExecutor
    end = end or (dt.date.today() - dt.timedelta(days=1))
    have = set(available_days())
    todo = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in todo if d not in have]
    if not todo:
        return []
    if log: log(f"ECMWF OLR: downloading {len(todo)} missing days …")
    new = []
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_day, d): d for d in todo}
        for fu in as_completed(futs):
            d = futs[fu]
            try:
                if fu.result() is not None:
                    new.append(d)
                    if log: log(f"  ✓ {d}", flush=True) if log is print else log(f"  ✓ {d}")
            except Exception:
                pass
    return sorted(new)


def load_series(start: dt.date, end: dt.date | None = None):
    """(time, LAT, LON, olr) from cache; calendar gaps linearly interpolated in time."""
    end = end or dt.date.today()
    days = [d for d in available_days() if start <= d <= end]
    if not days:
        return None
    # keep only the most recent block without gaps > 5 days
    blk = [days[-1]]
    for d in reversed(days[:-1]):
        if (blk[-1] - d).days > 5:
            break
        blk.append(d)
    days = sorted(blk)
    olr = np.stack([np.load(os.path.join(CACHE, d.strftime("%Y%m%d") + ".npy")) for d in days])
    t = np.array([np.datetime64(d) for d in days])
    full = np.arange(t[0], t[-1] + np.timedelta64(1, "D"), dtype="datetime64[D]")
    if len(full) != len(t):
        idx = (t - t[0]).astype(int); allidx = np.arange(len(full))
        out = np.empty((len(full),) + olr.shape[1:], np.float32)
        for j in range(olr.shape[1]):
            for i in range(olr.shape[2]):
                out[:, j, i] = np.interp(allidx, idx, olr[:, j, i])
        olr, t = out, full
    return t, LAT.copy(), LON.copy(), olr


def load_olr_ecmwf(ndays=240, log=None, quick_days=10):
    """Load the last `ndays` (all ECMWF). Only the most recent `quick_days` are fetched here
    (fast); the deep backfill of older days is done by update_data.py in the background."""
    today = dt.date.today()
    start = today - dt.timedelta(days=ndays)
    try:
        update(today - dt.timedelta(days=quick_days), today - dt.timedelta(days=1), log=log)
    except Exception:
        pass
    g = load_series(start)
    if g is None:
        raise RuntimeError("No ECMWF OLR data available")
    t, lat, lon, olr = g
    src = f"ECMWF IFS HRES 0.25° (Open Data) – TOA OLR from 'ttr', {len(t)} days, auto-updated to {t[-1]}"
    return t, lat, lon, olr, src, dict(ecmwf_days=len(t))
