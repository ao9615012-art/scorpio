"""
SCORPIO – Data access layer
===========================
Sources (all real, public):
  * OLR daily  : NOAA PSL OpenDAP – NCEP/NCAR Reanalysis 'ulwrf.ntat.gauss' (updated to near real-time)
                 Fallback: NOAA Interpolated OLR (1974-2022) if reanalysis unreachable
  * MJO index  : NOAA PSL OMI (OLR MJO Index) – daily, near real-time
  * BSISO index: IPRC / Kikuchi Bimodal ISO index (25-90 day filtered PCs)
A local cache (data/*.npz / *.csv) makes the app work offline after the first run.
"""
from __future__ import annotations
import os, io, datetime as dt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "data")
os.makedirs(CACHE, exist_ok=True)

REANALYSIS_URL = ("https://psl.noaa.gov/thredds/dodsC/Datasets/ncep.reanalysis/"
                  "Dailies/other_gauss/ulwrf.ntat.gauss.{year}.nc")
INTERP_OLR_URL = "https://psl.noaa.gov/thredds/dodsC/Datasets/interp_OLR/olr.day.mean.nc"
OMI_URL = "https://psl.noaa.gov/mjo/mjoindex/omi.1x.txt"
BSISO_URL = "https://iprc.soest.hawaii.edu/users/kazuyosh/ISO_index/data/BSISO_25-90bpfil_pc.txt"

LAT_LIM = 30.0


# --------------------------------------------------------------------------- OLR
def _open(url):
    import netCDF4
    return netCDF4.Dataset(url)


def load_olr(ndays: int = 240, force: bool = False):
    """Return (time[np.datetime64], lat, lon, olr[t,lat,lon]) for the tropics ±30°.

    ndays : number of most-recent days to fetch (≥ 200 recommended for the 96-day filter).
    """
    cache_file = os.path.join(CACHE, "olr_cache.npz")
    today = np.datetime64(dt.date.today())
    if not force and os.path.exists(cache_file):
        z = np.load(cache_file, allow_pickle=True)
        age = (today - z["time"][-1]).astype("timedelta64[D]").astype(int)
        if age <= 2 and len(z["time"]) >= ndays:
            return z["time"], z["lat"], z["lon"], z["olr"], str(z["source"])
    try:
        out = _load_reanalysis(ndays)
        src = "NCEP/NCAR Reanalysis – OLR top of atmosphere (NOAA PSL OpenDAP)"
    except Exception as e:  # noqa
        try:
            out = _load_interp(ndays)
            src = "NOAA Interpolated OLR (NOAA PSL OpenDAP) – archive"
        except Exception as e2:
            if os.path.exists(cache_file):
                z = np.load(cache_file, allow_pickle=True)
                return z["time"], z["lat"], z["lon"], z["olr"], str(z["source"]) + " [cached]"
            raise RuntimeError(f"OLR download failed: {e} / {e2}")
    t, lat, lon, olr = out
    lat = np.ascontiguousarray(np.ma.filled(lat, np.nan), dtype=np.float64)
    lon = np.ascontiguousarray(np.ma.filled(lon, np.nan), dtype=np.float64)
    olr = np.ascontiguousarray(np.ma.filled(olr, np.nan), dtype=np.float32)
    # fill any missing values by linear interpolation in time
    if np.isnan(olr).any():
        for j in range(olr.shape[1]):
            for i in range(olr.shape[2]):
                col = olr[:, j, i]; bad = np.isnan(col)
                if bad.any() and (~bad).any():
                    col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
        olr = np.nan_to_num(olr, nan=float(np.nanmean(olr)))
    np.savez_compressed(cache_file, time=t, lat=lat, lon=lon, olr=olr, source=src)
    return t, lat, lon, olr, src


def _load_reanalysis(ndays):
    import netCDF4
    year = dt.date.today().year
    chunks, times = [], []
    need = ndays
    lat = lon = None
    for y in range(year, year - 3, -1):
        try:
            ds = _open(REANALYSIS_URL.format(year=y))
        except Exception:
            continue
        tv = ds.variables["time"]
        tt = netCDF4.num2date(tv[:], tv.units, only_use_cftime_datetimes=False)
        tt = np.array([np.datetime64(dt.date(x.year, x.month, x.day)) for x in tt])
        if lat is None:
            lat_all = ds.variables["lat"][:].astype(float)
            sel = np.where(np.abs(lat_all) <= LAT_LIM + 0.6)[0]
            lat = lat_all[sel]
            lon = ds.variables["lon"][:].astype(float)
        n = min(need, len(tt))
        v = np.asarray(ds.variables["ulwrf"][-n:, sel[0]:sel[-1] + 1, :], dtype=np.float32)
        chunks.insert(0, v)
        times.insert(0, tt[-n:])
        need -= n
        ds.close()
        if need <= 0:
            break
    if not chunks:
        raise RuntimeError("no reanalysis data")
    olr = np.concatenate(chunks, 0)
    t = np.concatenate(times)
    # lat ascending south→north
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        olr = olr[:, ::-1, :]
    return t, lat, lon, olr


def _load_interp(ndays):
    import netCDF4
    ds = _open(INTERP_OLR_URL)
    tv = ds.variables["time"]
    tt = netCDF4.num2date(tv[-ndays:], tv.units, only_use_cftime_datetimes=False)
    t = np.array([np.datetime64(dt.date(x.year, x.month, x.day)) for x in tt])
    lat_all = ds.variables["lat"][:].astype(float)
    sel = np.where(np.abs(lat_all) <= LAT_LIM + 0.1)[0]
    lat = lat_all[sel]
    lon = ds.variables["lon"][:].astype(float)
    olr = np.asarray(ds.variables["olr"][-ndays:, sel[0]:sel[-1] + 1, :], dtype=np.float32)
    ds.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        olr = olr[:, ::-1, :]
    return t, lat, lon, olr


# --------------------------------------------------------------------------- indices
def _fetch_text(url, cache_name, max_age_days=1):
    import requests
    f = os.path.join(CACHE, cache_name)
    if os.path.exists(f):
        age = (dt.datetime.now() - dt.datetime.fromtimestamp(os.path.getmtime(f))).days
        if age < max_age_days:
            return open(f, encoding="utf-8", errors="ignore").read()
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        open(f, "w", encoding="utf-8").write(r.text)
        return r.text
    except Exception:
        if os.path.exists(f):
            return open(f, encoding="utf-8", errors="ignore").read()
        raise


def load_omi() -> pd.DataFrame:
    """OMI (Kiladis et al. 2014): year mon day hour PC1 PC2 amplitude.
    Note: OMI PC1 ≈ RMM2, OMI -PC2 ≈ RMM1 → we convert to RMM-like phase space."""
    txt = _fetch_text(OMI_URL, "omi.txt")
    rows = []
    for line in txt.splitlines():
        p = line.split()
        if len(p) < 6:
            continue
        try:
            y, m, d = int(p[0]), int(p[1]), int(p[2])
            vals = [float(x) for x in p[3:]]
        except ValueError:
            continue
        if len(vals) == 4:   # with hour column
            vals = vals[1:]
        pc1, pc2, amp = vals[:3]
        rows.append((dt.date(y, m, d), pc1, pc2, amp))
    df = pd.DataFrame(rows, columns=["date", "PC1", "PC2", "amp"]).set_index("date")
    # RMM-equivalent coordinates
    df["RMM1"] = -df["PC2"]
    df["RMM2"] = df["PC1"]
    df["phase"] = rmm_phase(df["RMM1"].values, df["RMM2"].values)
    return df


def load_bsiso() -> pd.DataFrame:
    txt = _fetch_text(BSISO_URL, "bsiso.txt", max_age_days=7)
    rows = []
    for line in txt.splitlines():
        p = line.split()
        if len(p) < 7 or not p[0].isdigit():
            continue
        y, m, d = int(p[0]), int(p[1]), int(p[2])
        rows.append((dt.date(y, m, d), float(p[3]), float(p[4]), int(p[5]), float(p[6])))
    return pd.DataFrame(rows, columns=["date", "PCx", "PCy", "phase", "amp"]).set_index("date")


def rmm_phase(x, y):
    ang = np.degrees(np.arctan2(y, x)) % 360  # 0° = +RMM1 axis
    # WH04 phases: phase 5 spans angle 0–45 (RMM1>0, RMM2>0 small) ... standard mapping:
    # phase = floor((ang+180)/45)+1 with ang measured from -RMM1 axis counter-clockwise
    ph = (np.floor(((ang + 180) % 360) / 45).astype(int) + 1)
    return ph


# --------------------------------------------------------------------------- merged near-real-time record
def _regrid_to_1deg(lat, lon, olr):
    """Bilinear regrid of (t,lat,lon) from any grid to the GFS 1° grid (±30°)."""
    from scipy.interpolate import RegularGridInterpolator
    from . import gfs
    lon_ext = np.concatenate([lon, [lon[0] + 360]])
    olr_ext = np.concatenate([olr, olr[:, :, :1]], axis=2)
    out = np.empty((olr.shape[0], len(gfs.LAT), len(gfs.LON)), np.float32)
    LA, LO = np.meshgrid(gfs.LAT, gfs.LON, indexing="ij")
    pts = np.stack([LA.ravel(), LO.ravel()], -1)
    for i in range(olr.shape[0]):
        f = RegularGridInterpolator((lat, lon_ext), olr_ext[i], bounds_error=False, fill_value=None)
        out[i] = f(pts).reshape(LA.shape)
    return out


def load_olr_realtime(ndays=240, gfs_days=200, log=None):
    """Long record = reanalysis (regridded to 1°) + GFS-analysis daily OLR for the recent period.
    The two are blended with a per-grid-point bias correction estimated on the overlap.
    Returns time, lat, lon, olr, source_string, meta(dict)."""
    from . import gfs
    today = dt.date.today()
    yesterday = today - dt.timedelta(days=1)

    # 1) GFS recent days (auto-download whatever is missing)
    start = today - dt.timedelta(days=gfs_days)
    try:
        gfs.update(start, yesterday, log=log)
    except Exception:
        pass
    g = gfs.load_series(start, yesterday)

    # 2) reanalysis history
    try:
        t_r, lat_r, lon_r, olr_r, src_r = load_olr(ndays)
        olr_r1 = _regrid_to_1deg(lat_r, lon_r, olr_r)
    except Exception:
        t_r = None

    if g is None and t_r is None:
        raise RuntimeError("No OLR data available (GFS and reanalysis both failed).")
    if g is None:
        return t_r, gfs.LAT.copy(), gfs.LON.copy(), olr_r1, src_r, dict(gfs_days=0, rean_end=str(t_r[-1]))

    t_g, lat, lon, olr_g = g
    if t_r is None:
        return t_g, lat, lon, olr_g, "GFS analysis TOA OLR (NOAA/AWS, auto-updated)", dict(gfs_days=len(t_g))

    # 3) blend: bias-correct GFS toward reanalysis on overlapping dates
    common = np.intersect1d(t_r, t_g)
    if len(common) >= 10:
        ir = np.isin(t_r, common); ig = np.isin(t_g, common)
        bias = (olr_g[ig].mean(0) - olr_r1[ir].mean(0)).astype(np.float32)
        olr_g = olr_g - bias[None]
        nover = len(common)
    else:
        nover = 0
    keep = t_r < t_g[0]
    t = np.concatenate([t_r[keep], t_g])
    olr = np.concatenate([olr_r1[keep], olr_g], 0)
    # fill any calendar gaps by linear interpolation in time
    full = np.arange(t[0], t[-1] + np.timedelta64(1, "D"), dtype="datetime64[D]")
    if len(full) != len(t):
        idx = (t - t[0]).astype(int)
        out = np.empty((len(full),) + olr.shape[1:], np.float32)
        allidx = np.arange(len(full))
        for j in range(olr.shape[1]):
            for i in range(olr.shape[2]):
                out[:, j, i] = np.interp(allidx, idx, olr[:, j, i])
        olr, t = out, full
    olr = olr[-max(ndays, 200):]
    t = t[-len(olr):]
    src = (f"NCEP/NCAR Reanalysis (حتى {t_r[-1]}) + GFS analysis TOA OLR "
           f"(NOAA/AWS، {len(t_g)} يوم، محدَّث تلقائياً حتى {t_g[-1]})")
    return t, lat, lon, olr, src, dict(gfs_days=len(t_g), rean_end=str(t_r[-1]), overlap=nover)
