"""
SCORPIO – Space-time spectral filtering & statistical wave forecast
===================================================================
Method
------
1. OLR anomalies (time-mean + linear trend removed, per grid point).
2. For every latitude row: 2-D FFT in (time, longitude)  →  wavenumber-frequency space
   (Wheeler & Kiladis 1999).
3. Keep only the (k, ω) box / dispersion-curve region belonging to each wave type
   (Kiladis et al. 2009, Wheeler & Weickmann 2001), inverse-FFT back to (time, lon).
4. FORECAST  (Wheeler & Weickmann 2001): the series is zero-padded AFTER the last
   observation before the FFT. Because the retained modes are narrow-band and
   propagating, the filtered signal in the padded window is a statistical extrapolation
   of the wave – a skilful ~1-3 week outlook for MJO/BSISO, ~5-10 days for Kelvin/ER.

Conventions: k > 0 eastward, k < 0 westward, ω in cycles per day (cpd).
"""
from __future__ import annotations
import numpy as np

G = 9.81
BETA = 2.28e-11          # m^-1 s^-1
R_EARTH = 6.371e6
DAY = 86400.0

# --------------------------------------------------------------------------- wave definitions
WAVES = {
    "MJO": dict(name_ar="مادن–جوليان MJO", kmin=1, kmax=5, pmin=30, pmax=96,
                h=None, color="#d62728", direction="east"),
    "Kelvin": dict(name_ar="موجة كلفن Kelvin", kmin=1, kmax=14, pmin=2.5, pmax=20,
                   h=(8, 90), color="#1f77b4", direction="east"),
    "ER": dict(name_ar="روسبي الاستوائية ER", kmin=-10, kmax=-1, pmin=9.7, pmax=48,
               h=(8, 90), color="#2ca02c", direction="west"),
    "BSISO": dict(name_ar="تذبذب موسم الصيف BSISO", kmin=-6, kmax=6, pmin=25, pmax=90,
                  h=None, color="#9467bd", direction="both"),
    "MRG": dict(name_ar="روسبي–جاذبية مختلطة MRG", kmin=-10, kmax=-1, pmin=3, pmax=10,
                h=(8, 90), color="#ff7f0e", direction="west"),
    "TD": dict(name_ar="اضطراب مداري TD-type", kmin=-20, kmax=-6, pmin=2.5, pmax=5,
               h=None, color="#8c564b", direction="west"),
}


# --------------------------------------------------------------------------- dispersion curves
def kelvin_freq(k, h):
    """ω (cpd) of Kelvin wave for planetary wavenumber k and equivalent depth h."""
    c = np.sqrt(G * h)
    kk = k / R_EARTH
    return kk * c * DAY / (2 * np.pi)


def er_freq(k, h, n=1):
    """ω (cpd) of n=1 equatorial Rossby wave (negative k westward); returns |ω|."""
    c = np.sqrt(G * h)
    kk = np.abs(k) / R_EARTH
    w = BETA * kk / (kk ** 2 + (2 * n + 1) * BETA / c)
    return w * DAY / (2 * np.pi)


def mrg_freq(k, h):
    """|ω| (cpd) of mixed Rossby-gravity wave for westward k (<0)."""
    c = np.sqrt(G * h)
    kk = k / R_EARTH  # negative
    w = 0.5 * c * (kk + np.sqrt(kk ** 2 + 4 * BETA / c))
    return np.abs(w) * DAY / (2 * np.pi)


def wave_mask(K, W, wave):
    """Boolean mask in (k, ω) space for wave definition dict (K, W broadcastable arrays;
    W ≥ 0 is absolute frequency, K signed)."""
    d = WAVES[wave]
    m = (K >= d["kmin"]) & (K <= d["kmax"]) & (W >= 1.0 / d["pmax"]) & (W <= 1.0 / d["pmin"])
    if d["h"] is not None:
        hmin, hmax = d["h"]
        if wave == "Kelvin":
            m &= (W >= kelvin_freq(K, hmin)) & (W <= kelvin_freq(K, hmax))
        elif wave == "ER":
            m &= (W >= er_freq(K, hmin)) & (W <= er_freq(K, hmax))
        elif wave == "MRG":
            m &= (W >= mrg_freq(K, hmin)) & (W <= mrg_freq(K, hmax))
    return m


# --------------------------------------------------------------------------- anomalies
def anomalies(olr):
    """Remove time mean and linear trend at every grid point (t, lat, lon)."""
    nt = olr.shape[0]
    t = np.arange(nt, dtype=float)
    t -= t.mean()
    mean = olr.mean(0)
    a = olr - mean
    slope = (t[:, None, None] * a).sum(0) / (t ** 2).sum()
    return a - slope[None] * t[:, None, None]


def _taper(nt, frac=0.1):
    """Split-cosine-bell taper applied to the START of the record only
    (the end is left intact because the forecast is appended there)."""
    w = np.ones(nt)
    n = int(nt * frac)
    x = np.linspace(0, np.pi, n)
    w[:n] = 0.5 * (1 - np.cos(x))
    return w


# --------------------------------------------------------------------------- core filter
def filter_forecast(anom, lon, waves, nfcst=15, taper=True):
    """Space-time filter + zero-padding forecast.

    anom : (t, lat, lon) OLR anomalies (observed only)
    waves: list of wave names
    nfcst: forecast days appended
    Returns dict wave -> (t+nfcst, lat, lon) filtered fields.
    """
    nt, nlat, nlon = anom.shape
    npad = nfcst + max(60, nt // 2)           # extra zeros reduce wrap-around leakage
    N = nt + npad
    x = np.zeros((N, nlat, nlon), dtype=np.float64)
    x[:nt] = anom
    if taper:
        x[:nt] *= _taper(nt)[:, None, None]

    # FFT: time axis 0, lon axis 2.  numpy convention exp(-i(ωt + kx))
    F = np.fft.fft2(x, axes=(0, 2))
    freq = np.fft.fftfreq(N, d=1.0)           # cpd, signed
    kk = np.fft.fftfreq(nlon, d=1.0 / nlon)   # planetary wavenumber, signed
    Wf, Kk = np.meshgrid(freq, kk, indexing="ij")   # (N, nlon)

    # Eastward propagation ⇔ ω and k opposite sign in exp(-i(ωt+kx)) convention.
    # Define signed physical wavenumber: k_phys = -sign(ω) * k  (positive = eastward)
    K_phys = -np.sign(Wf) * Kk
    Wabs = np.abs(Wf)

    out = {}
    for w in waves:
        m = wave_mask(K_phys, Wabs, w)
        Ff = F * m[:, None, :]
        y = np.real(np.fft.ifft2(Ff, axes=(0, 2)))
        out[w] = y[: nt + nfcst].astype(np.float32)
    return out


# --------------------------------------------------------------------------- diagnostics
def meridional_mean(field, lat, lat1, lat2):
    sel = (lat >= lat1) & (lat <= lat2)
    wts = np.cos(np.deg2rad(lat[sel]))
    return (field[:, sel, :] * wts[None, :, None]).sum(1) / wts.sum()


def zonal_mean(field, lon, lon1, lon2):
    sel = (lon >= lon1) & (lon <= lon2)
    return field[:, :, sel].mean(2)


def wave_amplitude(field, lat, lon, box):
    """RMS amplitude of a filtered field inside a box (lon1, lon2, lat1, lat2)."""
    lon1, lon2, lat1, lat2 = box
    s1 = (lat >= lat1) & (lat <= lat2)
    s2 = (lon >= lon1) & (lon <= lon2)
    sub = field[:, s1][:, :, s2]
    return np.sqrt((sub ** 2).mean((1, 2)))


def bsiso_proxy_index(bsiso_field, lat, lon):
    """Simple two-component BSISO phase proxy (Kikuchi-like) from 25–90 day filtered OLR:
    PC-x  ~ –OLR over equatorial Indian Ocean (70–100E, 5S–5N)
    PC-y  ~ –OLR over Bay of Bengal / India (70–100E, 12–22N)
    Northward propagation traces an anticlockwise loop in this space."""
    def box(lat1, lat2):
        s1 = (lat >= lat1) & (lat <= lat2)
        s2 = (lon >= 70) & (lon <= 100)
        return -bsiso_field[:, s1][:, :, s2].mean((1, 2))
    x = box(-5, 5)
    y = box(12, 22)
    sd = np.sqrt(0.5 * (x.var() + y.var())) or 1.0
    x, y = x / sd, y / sd
    amp = np.hypot(x, y)
    ang = np.degrees(np.arctan2(y, x)) % 360
    phase = (np.floor(ang / 45).astype(int) % 8) + 1
    return x, y, amp, phase


def mjo_proxy_index(mjo_field, lat, lon):
    """RMM-like MJO index from the 30–96 day, k=1–5 eastward-filtered OLR (includes forecast days).
    1. 15S–15N mean  2. zonal wavenumber-1 complex projection → convection-centre longitude & amplitude
    3. map longitude to the WH04 RMM angle via the observed phase/longitude relation
    Returns x (RMM1-like), y (RMM2-like), amp, phase(1-8)."""
    m = meridional_mean(mjo_field, lat, -15, 15)                # (t, lon)
    lam = np.deg2rad(lon)
    c = (m * np.exp(-1j * lam)[None, :]).mean(1)                # k=1 projection
    amp = 2 * np.abs(c)
    lon_c = (-np.degrees(np.angle(-c))) % 360                   # longitude of OLR MINIMUM (convection)
    # WH04 phase-centre longitudes (approx.): phase 1..8 then wrap
    ph_lon = np.array([20, 65, 90, 115, 140, 160, 185, 260, 380])
    ph_val = np.arange(1, 10, dtype=float)                       # 1..9 (9 ≡ 1)
    lc = np.where(lon_c < 20, lon_c + 360, lon_c)
    p_cont = np.interp(lc, ph_lon, ph_val)                      # continuous phase 1..9
    ang = np.deg2rad(180 + 45 * (p_cont - 1) + 22.5)            # RMM angle (0 = +RMM1)
    sd = amp.std() or 1.0
    r = amp / (0.6 * amp.mean() + 1e-9)                         # amp≈1 ⇒ moderate activity (RMM-like scaling)
    x, y = r * np.cos(ang), r * np.sin(ang)
    phase = (np.floor(p_cont - 1).astype(int) % 8) + 1
    return x, y, r, phase, lon_c
