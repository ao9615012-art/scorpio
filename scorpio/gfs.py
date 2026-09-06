"""
SCORPIO – Near-real-time OLR from GFS analyses (auto-updating)
==============================================================
Source: NOAA GFS 1.0° on the AWS Open Data bucket (noaa-gfs-bdp-pds), updated 4×/day.
Field : ULWRF top-of-atmosphere, 0–6 h average of the f006 forecast from each cycle
        (00/06/12/18 UTC) → the 4 cycles cover a full 24 h → daily mean OLR.
Only the ULWRF GRIB message (~70 KB) is downloaded via HTTP byte-range using the .idx file,
so a full day costs < 300 KB. Fallback: NOMADS grib-filter CGI.
Daily fields are cached in data/gfs_olr/YYYYMMDD.npy (±30°, 1°×1°).
"""
from __future__ import annotations
import os, io, datetime as dt, tempfile
import warnings
import numpy as np
import requests
warnings.filterwarnings("ignore", category=FutureWarning)

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "data", "gfs_olr")
os.makedirs(CACHE, exist_ok=True)

S3 = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/gfs.{ymd}/{hh}/atmos/gfs.t{hh}z.pgrb2.1p00.f006"
NOMADS = ("https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_1p00.pl?dir=%2Fgfs.{ymd}%2F{hh}%2Fatmos"
          "&file=gfs.t{hh}z.pgrb2.1p00.f006&var_ULWRF=on&lev_top_of_atmosphere=on")

LAT = np.arange(-30, 31, 1.0)          # south → north
LON = np.arange(0, 360, 1.0)
CYCLES = ("00", "06", "12", "18")


def _grib_to_array(raw: bytes) -> np.ndarray:
    import xarray as xr
    with tempfile.NamedTemporaryFile(suffix=".grb2", delete=False) as f:
        f.write(raw); fn = f.name
    try:
        ds = xr.open_dataset(fn, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = [v for v in ds.data_vars][0]
        da = ds[var].sel(latitude=slice(30, -30))       # GRIB lat is north→south
        arr = da.values[::-1, :].astype(np.float32)     # flip to south→north
        return arr
    finally:
        try:
            os.remove(fn)
        except OSError:
            pass


def fetch_cycle(day: dt.date, hh: str, timeout=60) -> np.ndarray | None:
    ymd = day.strftime("%Y%m%d")
    url = S3.format(ymd=ymd, hh=hh)
    try:
        idx = requests.get(url + ".idx", timeout=timeout)
        if idx.status_code == 200:
            lines = idx.text.splitlines()
            for i, l in enumerate(lines):
                if "ULWRF:top of atmosphere" in l:
                    s = int(l.split(":")[1])
                    e = int(lines[i + 1].split(":")[1]) - 1 if i + 1 < len(lines) else ""
                    r = requests.get(url, headers={"Range": f"bytes={s}-{e}"}, timeout=timeout)
                    if r.status_code in (200, 206):
                        return _grib_to_array(r.content)
                    break
    except Exception:
        pass
    try:  # fallback NOMADS
        r = requests.get(NOMADS.format(ymd=ymd, hh=hh), timeout=timeout)
        if r.status_code == 200 and len(r.content) > 5000:
            return _grib_to_array(r.content)
    except Exception:
        pass
    return None


def fetch_day(day: dt.date, min_cycles=3) -> np.ndarray | None:
    """Daily-mean TOA OLR for `day` (requires ≥ min_cycles of the 4 cycles). Cached."""
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


def update(start: dt.date, end: dt.date | None = None, log=None, workers=8):
    """Download any missing days in [start, end] in parallel. Returns list of newly added dates."""
    from concurrent.futures import ThreadPoolExecutor
    end = end or (dt.date.today() - dt.timedelta(days=1))
    have = set(available_days())
    todo = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    todo = [d for d in todo if d not in have]
    if not todo:
        return []
    if log: log(f"GFS OLR: downloading {len(todo)} missing days …")
    new = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d, arr in zip(todo, ex.map(fetch_day, todo)):
            if arr is not None:
                new.append(d)
    return sorted(new)


def load_series(start: dt.date, end: dt.date | None = None):
    """Return (time[datetime64[D]], LAT, LON, olr[t,lat,lon]) from cache for the range."""
    end = end or dt.date.today()
    days = [d for d in available_days() if start <= d <= end]
    if not days:
        return None
    olr = np.stack([np.load(os.path.join(CACHE, d.strftime("%Y%m%d") + ".npy")) for d in days])
    t = np.array([np.datetime64(d) for d in days])
    return t, LAT.copy(), LON.copy(), olr
