"""
SCORPIO – Advanced diagnostics ("Scorpion" tool-kit)
=====================================================
1. Velocity potential at 200 hPa (VP200) from ECMWF divergence  → upper-level MJO/Kelvin outflow
   (Ventrice et al. 2013 style). Poisson equation ∇²χ = D solved spectrally on a lat-lon grid.
2. Kelvin-mode projection on (u, Φ) at 850 hPa using the theoretical Matsuno/Hough meridional
   structure exp(-y²/2) – the "local Kelvin wave identification" idea (Žagar et al.; QJRMS 2024).
3. Wheeler–Kiladis (1999) wavenumber–frequency power spectrum (symmetric part) with the
   theoretical dispersion curves → verification of which waves are actually active.
4. Hindcast skill of the statistical wave forecast (correlation vs lead) for the current record.
"""
from __future__ import annotations
import numpy as np
from . import waves as W

R = 6.371e6
OMEGA = 7.292e-5
BETA = 2.28e-11
G = 9.81


# --------------------------------------------------------------------------- 1. velocity potential
def velocity_potential(div, lat, lon):
    """Solve ∇²χ = D on the sphere with a spectral (FFT in lon) + finite-difference (lat) method.
    div: (nlat, nlon) 1/s on a regular grid (lat ascending, lon 0..360).  Returns χ in m²/s."""
    nlat, nlon = div.shape
    phi = np.deg2rad(lat)
    dphi = phi[1] - phi[0]
    cos = np.cos(phi)
    Dk = np.fft.rfft(div, axis=1)                       # (nlat, nk)
    ks = np.arange(Dk.shape[1])
    chi_k = np.zeros_like(Dk)
    # Laplacian: (1/R²)[ (1/cos) d/dphi(cos dχ/dphi) - k²/cos² χ ]
    for ik, k in enumerate(ks):
        A = np.zeros((nlat, nlat))
        for j in range(nlat):
            cp = np.cos(phi[j] + dphi / 2) if j < nlat - 1 else 0.0
            cm = np.cos(phi[j] - dphi / 2) if j > 0 else 0.0
            A[j, j] = -(cp + cm) / (cos[j] * dphi ** 2) - k ** 2 / cos[j] ** 2
            if j < nlat - 1: A[j, j + 1] = cp / (cos[j] * dphi ** 2)
            if j > 0: A[j, j - 1] = cm / (cos[j] * dphi ** 2)
        A /= R ** 2
        rhs = Dk[:, ik].copy()
        if k == 0:
            A[0, :] = 0; A[0, 0] = 1; rhs[0] = 0          # gauge: χ=0 at southern boundary for k=0
        chi_k[:, ik] = np.linalg.solve(A, rhs)
    chi = np.fft.irfft(chi_k, n=nlon, axis=1)
    return chi - chi.mean()


# --------------------------------------------------------------------------- 2. Kelvin projection
def kelvin_projection(u, z, lat, lon, h_eq=25.0):
    """Project (u, Φ=g z) at one level onto the Kelvin Hough structure  ψ(y)=exp(-y²/2),
    y = lat·(R)/L,  L = (c/β)^½,  c = (g h_eq)^½ .
    Returns the longitude series of the Kelvin coefficient (dimensionless, standardized units)
    and the reconstructed Kelvin u-field (m/s)."""
    c = np.sqrt(G * h_eq)
    L = np.sqrt(c / BETA)
    y = np.deg2rad(lat) * R / L
    psi = np.exp(-y ** 2 / 2)
    wts = np.cos(np.deg2rad(lat)) * psi
    # non-dimensionalize: u/c and Φ/c²  (Kelvin: u' = Φ'/c  → both normalized equal)
    un = u / c
    phin = (G * z) / c ** 2
    proj = ((un + phin) * wts[:, None]).sum(0) / (2 * (psi * wts).sum())
    u_k = c * proj[None, :] * psi[:, None]
    return proj, u_k


# --------------------------------------------------------------------------- 3. WK spectrum
def wk_spectrum(anom, lat, lon, seg_len=96, overlap=60, lat_max=15):
    """Symmetric wavenumber–frequency power (log10) following Wheeler & Kiladis (1999),
    normalized by a smoothed red background → 'signal strength' ratio.
    anom: (t, lat, lon) anomalies. Returns k (signed, east>0), freq (cpd>0), ratio, raw_log_power."""
    sel = np.abs(lat) <= lat_max
    a = anom[:, sel, :]
    la = lat[sel]
    sym = 0.5 * (a + a[:, ::-1, :])           # symmetric about equator (grid assumed symmetric)
    nt, nl, nx = sym.shape
    if nt < seg_len:
        seg_len = nt; overlap = 0
    step = max(1, seg_len - overlap)
    win = np.hanning(seg_len)
    P = None; n = 0
    for s0 in range(0, nt - seg_len + 1, step):
        seg = sym[s0:s0 + seg_len]
        seg = seg - seg.mean(0)
        seg = seg * win[:, None, None]
        F = np.fft.fft2(seg, axes=(0, 2))
        p = (np.abs(F) ** 2).mean(1)          # average over latitudes
        P = p if P is None else P + p
        n += 1
    P /= max(n, 1)
    freq = np.fft.fftfreq(seg_len)
    k = np.fft.fftfreq(nx, d=1.0 / nx)
    # physical convention: eastward ⇔ sign(ω) ≠ sign(k) in numpy  → reorder to (freq>0, k signed east+)
    pos = freq > 0
    fr = freq[pos]
    Pp = P[pos]                                # (nf, nx)
    kk = -k                                     # east positive for ω>0
    order = np.argsort(kk)
    kk = kk[order]; Pp = Pp[:, order]
    keep = np.abs(kk) <= 15
    kk = kk[keep]; Pp = Pp[:, keep]
    keep_f = fr <= 0.5
    fr = fr[keep_f]; Pp = Pp[keep_f]
    def smooth121(a, nk, nf):
        a = a.copy()
        for _ in range(nk):
            a[:, 1:-1] = 0.25 * a[:, :-2] + 0.5 * a[:, 1:-1] + 0.25 * a[:, 2:]
        for _ in range(nf):
            a[1:-1] = 0.25 * a[:-2] + 0.5 * a[1:-1] + 0.25 * a[2:]
        return a
    logP = np.log10(Pp + 1e-12)
    # light smoothing of the raw spectrum (short record) – WK99 style
    logP_s = smooth121(logP, 1, 2)
    # background: heavy smoothing (more passes at higher frequency, as in WK99)
    bg = smooth121(logP_s, 40, 20)
    ratio = 10 ** (logP_s - bg)
    return kk, fr, ratio, logP_s


def dispersion_curves(h_list=(8, 25, 90)):
    """Theoretical curves for plotting on WK diagram: dict name -> list of (k, freq) arrays."""
    ke = np.linspace(0.1, 15, 60); kw = -np.linspace(0.1, 15, 60)
    out = {"Kelvin": [], "ER": [], "MRG": []}
    for h in h_list:
        out["Kelvin"].append((ke, W.kelvin_freq(ke, h)))
        out["ER"].append((kw, W.er_freq(kw, h)))
        out["MRG"].append((kw, W.mrg_freq(kw, h)))
    return out


# --------------------------------------------------------------------------- 4. hindcast skill
def hindcast_skill(anom, lat, lon, wave_list, nfcst=15, n_cases=12, min_hist=120, box=None):
    """Re-run the statistical forecast from earlier initial times and correlate the forecast
    filtered field with the 'analysis' filtered field (full record filter). Returns
    dict wave -> corr[lead] (pattern correlation over the tropical belt or `box`)."""
    nt = anom.shape[0]
    full = W.filter_forecast(anom, lon, wave_list, nfcst=0)
    if box is not None:
        lon1, lon2, lat1, lat2 = box
        s1 = (lat >= lat1) & (lat <= lat2); s2 = (lon >= lon1) & (lon <= lon2)
    else:
        s1 = np.abs(lat) <= 20; s2 = np.ones(len(lon), bool)
    min_hist = min(min_hist, max(60, nt // 2))
    last_init = nt - nfcst                     # forecast valid times i0 .. i0+nfcst-1 must be < nt
    if last_init <= min_hist:
        min_hist = max(40, last_init - 10)
    if last_init < 30:
        return {w: np.full(nfcst, np.nan) for w in wave_list}, 0
    min_hist = min(min_hist, last_init)
    inits = np.unique(np.clip(np.linspace(min_hist, last_init, n_cases).astype(int), 30, last_init))
    skill = {w: np.zeros((len(inits), nfcst)) for w in wave_list}
    for ci, i0 in enumerate(inits):
        fc = W.filter_forecast(anom[:i0], lon, wave_list, nfcst=nfcst)
        for w in wave_list:
            for ld in range(nfcst):
                f = fc[w][i0 + ld][s1][:, s2].ravel()
                a = full[w][i0 + ld][s1][:, s2].ravel()
                if f.std() > 0 and a.std() > 0:
                    skill[w][ci, ld] = np.corrcoef(f, a)[0, 1]
    return {w: skill[w].mean(0) for w in wave_list}, len(inits)


# --------------------------------------------------------------------------- 5. Scorpion phase plot
def rmm_like_index(olr_field, u850_field, lat, lon, ref=None):
    """RMM-like two-component index (Wheeler & Hendon 2004 spirit) from 15S–15N means of
    OLR anomaly and 850 hPa zonal wind anomaly:
       PC1 ~ projection on cos/sin of planetary wavenumber-1 of the combined field.
    olr_field: (t, nlat, nlon) on any grid; u850_field: (t, nlatw, nlonw) on the 2.5° grid or None.
    ref: optional dict with normalisation (so forecast tracks use the same scaling as obs).
    Returns x, y, amp, phase, ref."""
    def band_mean(f, la):
        s = np.abs(la) <= 15
        w = np.cos(np.deg2rad(la[s]))
        return (f[:, s, :] * w[None, :, None]).sum(1) / w.sum()
    o = band_mean(olr_field, lat)                              # (t, nlon)
    lam = np.deg2rad(lon)
    # combined complex k=1 projection: OLR (sign flipped: convection positive) and U850
    c_o = (-o * np.exp(-1j * lam)[None, :]).mean(1)
    c = c_o / (ref["s_o"] if ref else (np.abs(c_o).std() or 1))
    if u850_field is not None:
        lam_u = np.deg2rad(np.arange(0, 360, 360 / u850_field.shape[-1]))
        c_u = (u850_field * np.exp(-1j * lam_u)[None, :]).mean(1)
        c = c + 0.7 * c_u / (ref["s_u"] if ref else (np.abs(c_u).std() or 1))
    if ref is None:
        ref = {"s_o": np.abs(c_o).std() or 1, "s_u": 1.0}
    amp = np.abs(c) / (np.abs(c).mean() * 0.8 + 1e-9)
    lon_c = (-np.degrees(np.angle(c))) % 360                  # longitude of convective centre (c built from -OLR)
    ph_lon = np.array([20, 65, 90, 115, 140, 160, 185, 260, 380]); ph_val = np.arange(1, 10, dtype=float)
    lc = np.where(lon_c < 20, lon_c + 360, lon_c)
    p_cont = np.interp(lc, ph_lon, ph_val)
    ang = np.deg2rad(180 + 45 * (p_cont - 1) + 22.5)
    x, y = amp * np.cos(ang), amp * np.sin(ang)
    phase = (np.floor(p_cont - 1).astype(int) % 8) + 1
    return x, y, amp, phase, ref


def scorpion_tracks(anom, lat, lon, nfcst, wave_sets, ecmwf_olr_fcst=None):
    """Build the 'Scorpion' multi-track set:
       body  = observed track (unfiltered anomaly, 11-day smoothed) for last 40 days
       tails = forecast tracks branching from today:
               * each wave_sets entry  (e.g. ["MJO"], ["MJO","Kelvin","ER"], ["MJO","ER"], ...)
               * optional ECMWF dynamical OLR forecast (t_f, lat, lon) if provided
    Returns dict name -> (x, y) arrays covering [today .. today+nfcst]."""
    from . import waves as W
    nt = anom.shape[0]
    allw = sorted({w for ws in wave_sets for w in ws})
    filt = W.filter_forecast(anom, lon, allw, nfcst=nfcst)
    # observed body: 11-day running mean of raw anomaly
    sm = anom.copy()
    k = 11
    cs = np.cumsum(np.concatenate([np.zeros((1,) + anom.shape[1:]), anom], 0), 0)
    for i in range(nt):
        a, b = max(0, i - k // 2), min(nt, i + k // 2 + 1)
        sm[i] = (cs[b] - cs[a]) / (b - a)
    xo, yo, ao, po, ref = rmm_like_index(sm, None, lat, lon)
    out = {"observed": (xo[nt - 40:], yo[nt - 40:], po[nt - 40:])}
    for ws in wave_sets:
        f = sum(filt[w] for w in ws)
        x, y, a, p, _ = rmm_like_index(f, None, lat, lon, ref=ref)
        # blend start to observed point for continuity
        x[nt - 1:] += (xo[nt - 1] - x[nt - 1]) * np.linspace(1, 0, nfcst + 1) ** 2
        y[nt - 1:] += (yo[nt - 1] - y[nt - 1]) * np.linspace(1, 0, nfcst + 1) ** 2
        out["+".join(ws)] = (x[nt - 1:], y[nt - 1:], p[nt - 1:])
    if ecmwf_olr_fcst is not None:
        fa = ecmwf_olr_fcst - anom.mean(0)[None]   # anomaly vs record mean (same base as anom)
        seq = np.concatenate([sm[nt - 5:], fa], 0)
        x, y, a, p, _ = rmm_like_index(seq, None, lat, lon, ref=ref)
        out["ECMWF"] = (x[4:], y[4:], p[4:])
    return out


# --------------------------------------------------------------------------- 6. Scorpion tracks on the map
def track_wave_centres(field, lat, lon, nt, thresh=None, lat_band=(-20, 25), min_sep=25,
                       max_jump=12, direction="east", min_len=4, recent=5):
    """Track centres of enhanced convection (negative filtered-OLR anomaly) through time.
    field : (t_total, lat, lon) filtered anomaly (obs 0..nt-1, forecast nt..)
    thresh: None → adaptive (−0.8 σ of the meridional-mean field)
    Returns list of tracks; each track = list of (t_index, lon, lat, value)."""
    T = field.shape[0]
    s = (lat >= lat_band[0]) & (lat <= lat_band[1])
    la = lat[s]
    mm_all = field[:, s].mean(1)                       # (T, nlon)
    if thresh is None:
        thresh = -0.8 * mm_all.std()
    cands = []
    for ti in range(T):
        f = field[ti][s]; mm = mm_all[ti]; c = []
        for j in range(len(lon)):
            jm, jp = (j - 1) % len(lon), (j + 1) % len(lon)
            if mm[j] < thresh and mm[j] <= mm[jm] and mm[j] <= mm[jp]:
                # latitude = centroid of negative anomaly in a ±10° lon window
                jj = [(j + k) % len(lon) for k in range(-10, 11)]
                blk = np.minimum(f[:, jj], 0)
                wts = -blk.sum(1)
                clat = float((wts * la).sum() / wts.sum()) if wts.sum() > 0 else 0.0
                c.append((lon[j], clat, mm[j]))
        c.sort(key=lambda x: x[2]); keep = []
        for x in c:
            if all(min(abs(x[0] - k[0]), 360 - abs(x[0] - k[0])) >= min_sep for k in keep):
                keep.append(x)
        cands.append(keep)

    def ok_step(d):
        if direction == "east":
            return -3 <= d <= max_jump
        if direction == "west":
            return -max_jump <= d <= 3
        return abs(d) <= max_jump

    tracks, active = [], []
    for ti in range(T):
        used, new_active = set(), []
        for tr in active:
            plon = tr[-1][1]; best, bd = None, 1e9
            for ci, (clon, clat, cv) in enumerate(cands[ti]):
                d = (clon - plon + 180) % 360 - 180
                if ci not in used and ok_step(d) and abs(d) < bd:
                    best, bd = ci, abs(d)
            if best is not None:
                clon, clat, cv = cands[ti][best]
                tr.append((ti, clon, clat, cv)); used.add(best); new_active.append(tr)
            elif len(tr) >= min_len:
                tracks.append(tr)
        for ci, (clon, clat, cv) in enumerate(cands[ti]):
            if ci not in used:
                new_active.append([(ti, clon, clat, cv)])
        active = new_active
    tracks += [tr for tr in active if len(tr) >= min_len]
    tracks = [tr for tr in tracks if tr[-1][0] >= nt - recent]
    # drop weak tracks: peak strength must reach 1.5× threshold
    tracks = [tr for tr in tracks if min(p[3] for p in tr) <= 1.5 * thresh]
    # smooth latitude (5-point running mean) & drop tracks with unphysical lat jumps
    out = []
    for tr in tracks:
        ti = [p[0] for p in tr]; lx = [p[1] for p in tr]; ly = np.array([p[2] for p in tr]); v = [p[3] for p in tr]
        k = 5; lys = ly.copy()
        for i in range(len(ly)):
            a_, b_ = max(0, i - k // 2), min(len(ly), i + k // 2 + 1); lys[i] = ly[a_:b_].mean()
        out.append(list(zip(ti, lx, lys, v)))
    return out


TRACK_CFG = {
    "MJO":    dict(direction="east", max_jump=10, min_sep=40, min_len=5),
    "Kelvin": dict(direction="east", max_jump=18, min_sep=20, min_len=3),
    "ER":     dict(direction="west", max_jump=10, min_sep=20, min_len=4),
    "BSISO":  dict(direction="both", max_jump=8,  min_sep=30, min_len=5),
}
