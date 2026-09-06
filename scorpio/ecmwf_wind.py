"""
SCORPIO – 850 hPa wind (u, v) from ECMWF IFS HRES Open Data, analysis + forecast to 360 h
Daily mean = average of the 4 six-hourly steps that fall inside the calendar day.
Cached per (cycle, target-day) in data/ecmwf_wind/ as npz on a 2.5° grid (±30°).
"""
from __future__ import annotations
import os, json, datetime as dt, tempfile, warnings
import numpy as np


def _ensure_cfgrib():
    """cfgrib/eccodes may be missing in ephemeral environments – install on demand."""
    try:
        import cfgrib  # noqa
    except ImportError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cfgrib", "eccodes"], check=False)
        import importlib; importlib.invalidate_caches()
from .ecmwf import _get, ROOTS, PATH

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "data", "ecmwf_wind")
os.makedirs(CACHE, exist_ok=True)

LAT = np.arange(-30, 31, 2.5)
LON = np.arange(0, 360, 2.5)


def _grib_uv(raw: bytes):
    import xarray as xr
    _ensure_cfgrib()
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as f:
        f.write(raw); fn = f.name
    try:
        ds = xr.open_dataset(fn, engine="cfgrib", backend_kwargs={"indexpath": ""})
        var = list(ds.data_vars)[0]
        da = ds[var]
        latg = da.latitude.values; long = np.round(da.longitude.values % 360, 3)
        v = da.values
        o = np.argsort(long); long = long[o]; v = v[:, o]
        if latg[0] > latg[-1]:
            latg = latg[::-1]; v = v[::-1]
        ii = np.searchsorted(latg, LAT); jj = np.searchsorted(long, LON)
        return v[np.ix_(ii, jj)].astype(np.float32)
    finally:
        try: os.remove(fn)
        except OSError: pass


def fetch_step(cycle: dt.datetime, step: int):
    """u, v at 850 hPa for one forecast step (2.5° subset)."""
    ymd = cycle.strftime("%Y%m%d"); hh = cycle.strftime("%H")
    for root in ROOTS:
        base = PATH.format(root=root, ymd=ymd, hh=hh, step=step)
        r = _get(base + ".index")
        if r is None:
            continue
        ent = {}
        for line in r.text.splitlines():
            if '"levelist": "850"' in line and ('"param": "u"' in line or '"param": "v"' in line):
                e = json.loads(line); ent[e["param"]] = e
        if len(ent) < 2:
            continue
        out = {}
        for p in ("u", "v"):
            e = ent[p]
            g = _get(base + ".grib2", headers={"Range": f"bytes={e['_offset']}-{e['_offset'] + e['_length'] - 1}"})
            if g is None:
                break
            out[p] = _grib_uv(g.content)
        if len(out) == 2:
            return out["u"], out["v"]
    return None


def latest_cycle(max_back=8) -> dt.datetime | None:
    """Most recent 00/12 UTC HRES cycle whose 360 h file is published."""
    now = dt.datetime.utcnow()
    c = now.replace(minute=0, second=0, microsecond=0, hour=(now.hour // 12) * 12)
    for _ in range(max_back):
        ymd = c.strftime("%Y%m%d"); hh = c.strftime("%H")
        for root in ROOTS:
            r = _get(PATH.format(root=root, ymd=ymd, hh=hh, step=360) + ".index", tries=2)
            if r is not None:
                return c
        c -= dt.timedelta(hours=12)
    return None


def daily_wind(cycle: dt.datetime, day: dt.date, log=None):
    """Daily-mean 850 hPa (u, v) for `day` from forecast `cycle` (cached)."""
    f = os.path.join(CACHE, f"{cycle:%Y%m%d%H}_{day:%Y%m%d}.npz")
    if os.path.exists(f):
        z = np.load(f); return z["u"], z["v"]
    steps = []
    for h in (0, 6, 12, 18):
        valid = dt.datetime.combine(day, dt.time(h))
        s = int((valid - cycle).total_seconds() // 3600)
        if 0 <= s <= 360 and (s <= 144 or s % 6 == 0):
            steps.append(s)
    if not steps:
        return None
    us, vs = [], []
    for s in steps:
        r = fetch_step(cycle, s)
        if r is not None:
            us.append(r[0]); vs.append(r[1])
    if not us:
        return None
    u = np.mean(us, 0); v = np.mean(vs, 0)
    np.savez_compressed(f, u=u, v=v)
    if log: log(f"  wind {day} ({len(us)} steps)")
    return u, v


def analysis_day(day: dt.date, log=None):
    """Daily-mean 850 hPa wind for a past day from the 4 analyses (step 0 of 00/06/12/18 cycles)."""
    f = os.path.join(CACHE, f"an_{day:%Y%m%d}.npz")
    if os.path.exists(f):
        z = np.load(f); return z["u"], z["v"]
    us, vs = [], []
    for h in (0, 6, 12, 18):
        r = fetch_step(dt.datetime.combine(day, dt.time(h)), 0)
        if r is not None:
            us.append(r[0]); vs.append(r[1])
    if len(us) < 2:
        return None
    u = np.mean(us, 0); v = np.mean(vs, 0)
    np.savez_compressed(f, u=u, v=v)
    if log: log(f"  wind analysis {day} ({len(us)} cycles)")
    return u, v


def wind_for_days(days, log=None, workers=3):
    """dict day -> (u, v). Days before the latest cycle come from analyses, later days from the
    latest HRES forecast cycle (out to 360 h)."""
    from concurrent.futures import ThreadPoolExecutor
    cyc = latest_cycle()
    if cyc is None:
        return {}, None

    def get(d):
        if d < cyc.date():
            return analysis_day(d, log)
        return daily_wind(cyc, d, log)

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d, r in zip(days, ex.map(get, days)):
            if r is not None:
                out[d] = r
    return out, cyc


# --------------------------------------------------------------------------- generic pressure-level fields
def fetch_fields(cycle: dt.datetime, step: int, wanted):
    """wanted: list of (param, level) e.g. [("d","200"),("gh","850"),("u","200"),("v","200")].
    Returns dict (param,level) -> 2.5° array (±30°)."""
    ymd = cycle.strftime("%Y%m%d"); hh = cycle.strftime("%H")
    for root in ROOTS:
        base = PATH.format(root=root, ymd=ymd, hh=hh, step=step)
        r = _get(base + ".index")
        if r is None:
            continue
        ent = {}
        for line in r.text.splitlines():
            for p, lv in wanted:
                if f'"levelist": "{lv}"' in line and f'"param": "{p}"' in line:
                    ent[(p, lv)] = json.loads(line)
        if len(ent) < len(wanted):
            continue
        out = {}
        for key, e in ent.items():
            g = _get(base + ".grib2", headers={"Range": f"bytes={e['_offset']}-{e['_offset'] + e['_length'] - 1}"})
            if g is None:
                break
            out[key] = _grib_uv(g.content)
        if len(out) == len(wanted):
            return out
    return None


ADV_FIELDS = [("d", "200"), ("u", "200"), ("v", "200"), ("gh", "850"), ("u", "850")]


def daily_advanced(cycle: dt.datetime, day: dt.date, log=None):
    """Daily mean of ADV_FIELDS for `day` from forecast `cycle` (cached npz)."""
    f = os.path.join(CACHE, f"adv_{cycle:%Y%m%d%H}_{day:%Y%m%d}.npz")
    if os.path.exists(f):
        z = np.load(f); return {tuple(k.split("_")): z[k] for k in z.files}
    steps = []
    for h in (0, 6, 12, 18):
        valid = dt.datetime.combine(day, dt.time(h))
        st = int((valid - cycle).total_seconds() // 3600)
        if 0 <= st <= 360 and (st <= 144 or st % 6 == 0):
            steps.append(st)
    acc = {}; n = 0
    for st in steps:
        r = fetch_fields(cycle, st, ADV_FIELDS)
        if r is None:
            continue
        for k, v in r.items():
            acc[k] = acc.get(k, 0) + v
        n += 1
    if n == 0:
        return None
    out = {k: (v / n).astype(np.float32) for k, v in acc.items()}
    np.savez_compressed(f, **{f"{k[0]}_{k[1]}": v for k, v in out.items()})
    if log: log(f"  adv {day} ({n} steps)")
    return out


def advanced_analysis_day(day: dt.date, log=None):
    f = os.path.join(CACHE, f"advan_{day:%Y%m%d}.npz")
    if os.path.exists(f):
        z = np.load(f); return {tuple(k.split("_")): z[k] for k in z.files}
    acc = {}; n = 0
    for h in (0, 12):
        r = fetch_fields(dt.datetime.combine(day, dt.time(h)), 0, ADV_FIELDS)
        if r is None:
            continue
        for k, v in r.items():
            acc[k] = acc.get(k, 0) + v
        n += 1
    if n == 0:
        return None
    out = {k: (v / n).astype(np.float32) for k, v in acc.items()}
    np.savez_compressed(f, **{f"{k[0]}_{k[1]}": v for k, v in out.items()})
    if log: log(f"  adv analysis {day}")
    return out


def advanced_for_days(days, log=None, workers=3):
    from concurrent.futures import ThreadPoolExecutor
    cyc = latest_cycle()
    if cyc is None:
        return {}, None

    def get(d):
        try:
            return advanced_analysis_day(d, log) if d < cyc.date() else daily_advanced(cyc, d, log)
        except Exception:
            return None

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for d, r in zip(days, ex.map(get, days)):
            if r is not None:
                out[d] = r
    return out, cyc
