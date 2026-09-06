"""Build a fully self-contained HTML dashboard (inline base64 PNGs, inline CSS) from the
latest SCORPIO data – renders anywhere, no server / no network needed."""
import sys, os, io, base64, datetime as dt
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorpio import data, waves, ecmwf, ecmwf_wind, advanced

NF = 15
t, lat, lon, olr, src, meta = ecmwf.load_olr_ecmwf(240)
if len(t) < 100:
    raise SystemExit(f"need >= 100 days of ECMWF OLR, have {len(t)}")
anom = waves.anomalies(olr)
filt = waves.filter_forecast(anom, lon, list(waves.WAVES), nfcst=NF)
nt = len(t)
t_all = np.concatenate([t, t[-1] + np.arange(1, NF + 1).astype("timedelta64[D]")])
coast = np.load(os.path.join(os.path.dirname(__file__), "data", "coast110.npz"))

# ---- 850 hPa wind (ECMWF HRES, latest cycle, analysis + forecast) for the map days
MAP_DAYS = [nt - 1, nt + 2, nt + 5, nt + 8, nt + 11, nt + 14]
_wind_days = [pd.Timestamp(t_all[i]).date() for i in MAP_DAYS]
WIND, WIND_CYCLE = ecmwf_wind.wind_for_days(_wind_days, log=print)
WLAT, WLON = ecmwf_wind.LAT, ecmwf_wind.LON
print("wind days:", len(WIND), "cycle:", WIND_CYCLE)

HOV_BACK = 45
_hov_days = [pd.Timestamp(t_all[i]).date() for i in range(nt - HOV_BACK, nt + NF)]
WIND_HOV, _ = ecmwf_wind.wind_for_days(_hov_days, log=None, workers=4)
print("hovmöller wind days:", len(WIND_HOV), "/", len(_hov_days))

ADV_BACK = 30
_adv_days = [pd.Timestamp(t_all[i]).date() for i in range(nt - ADV_BACK, nt + NF)]
ADV, _ = ecmwf_wind.advanced_for_days(_adv_days, log=print, workers=3)
print("advanced days:", len(ADV), "/", len(_adv_days))
SHOW = ["MJO", "Kelvin", "ER", "BSISO"]
COL = {w: waves.WAVES[w]["color"] for w in SHOW}
plt.rcParams["font.size"] = 9
EN = {"MJO": "MJO", "Kelvin": "Kelvin wave", "ER": "Equatorial Rossby (ER)", "BSISO": "BSISO"}
DESIGNER = "Designed by Ahmed Omar Dhafer"


def stamp(fig, title):
    """Main title + designer credit next to it, on every figure."""
    fig.suptitle(f"SCORPIO – {title}     |     {DESIGNER}", fontsize=12, fontweight="bold")



def b64(fig):
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=95, bbox_inches="tight"); plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def draw_wind(ax, i, regional=False):
    d = pd.Timestamp(t_all[i]).date()
    if d not in WIND:
        return
    u, v = WIND[d]
    st = 2 if regional else 3                      # 5° spacing regional, 7.5° global
    spd = np.hypot(u, v)
    q = ax.quiver(WLON[::st], WLAT[::st], u[::st, ::st], v[::st, ::st], color="#111",
                  scale=(180 if regional else 450), width=0.0035 if regional else 0.0014,
                  headwidth=3.5, headlength=4.5, pivot="mid", zorder=5, edgecolor="white", linewidth=0.4)
    ax.quiverkey(q, 0.93, 1.03, 10, "10 m/s", labelpos="W", fontproperties={"size": 7}, coordinates="axes")


def draw_map(ax, i, x1=0, x2=360, y1=-30, y2=30, title=""):
    tot = sum(filt[w][i] for w in SHOW)
    cf = ax.contourf(lon, lat, tot, levels=np.arange(-40, 41, 5), cmap="RdBu", extend="both")
    for w in SHOW:
        ax.contour(lon, lat, filt[w][i], levels=[-40, -30, -20, -10], colors=COL[w], linewidths=1.4)
        ax.contour(lon, lat, filt[w][i], levels=[10, 20, 30, 40], colors=COL[w], linewidths=0.8, linestyles="dotted")
    ax.plot(coast["x"], coast["y"], "k", lw=0.5)
    ax.plot(48.8, 15.9, marker="*", color="gold", ms=13, mec="k", zorder=6)
    draw_wind(ax, i, regional=(x2 - x1) < 200)
    draw_scorpion_tracks(ax, i, x1, x2)
    ax.set_xlim(x1, x2); ax.set_ylim(y1, y2); ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_facecolor("#f4f4f4")
    return cf


imgs = {}
TRACKS = {w: advanced.track_wave_centres(filt[w], lat, lon, nt, **cfg) for w, cfg in advanced.TRACK_CFG.items() if w in SHOW}


def draw_scorpion_tracks(ax, i, x1=0, x2=360, body_days=30):
    """Scorpion technique: for each wave, plot the observed path of its convective centre up to
    the map date (body: solid) and its forecast path from the map date onward (tail: dashed +
    arrow), with the position on the map date as a big dot."""
    def wrap(x):
        return x - 360 if (x1 < 0 and x > 180) else x
    for w, trs in TRACKS.items():
        col = COL[w]
        for tr in trs:
            ti = np.array([p[0] for p in tr]); lx = np.array([wrap(p[1]) for p in tr], float); ly = np.array([p[2] for p in tr], float)
            if ti[0] > i or ti[-1] < i - 2:          # track not relevant for this date
                continue
            # split segments at any longitude wrap (> 90° jump) so no line crosses the whole map
            br = np.where(np.abs(np.diff(lx)) > 90)[0]
            lx = np.insert(lx, br + 1, np.nan); ly = np.insert(ly, br + 1, np.nan); tif = np.insert(ti.astype(float), br + 1, np.nan)
            body = (tif <= i) & (tif >= i - body_days)
            tail = tif >= i
            if body.sum() >= 2:
                ax.plot(lx[body], ly[body], "-", color=col, lw=2.6, solid_capstyle="round", zorder=7,
                        path_effects=[pe.Stroke(linewidth=4.2, foreground="white"), pe.Normal()])
            if tail.sum() >= 2:
                ax.plot(lx[tail], ly[tail], "--", color=col, lw=2.0, zorder=7,
                        path_effects=[pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])
                sel = tail & (np.mod(tif - i, 5) == 0) & (tif > i)
                ax.plot(lx[sel], ly[sel], "s", color=col, ms=4, mec="white", zorder=9)
                for xx, yy, tt in zip(lx[sel], ly[sel], tif[sel]):
                    if np.isfinite(xx):
                        ax.annotate(f"+{int(tt - i)}", (xx, yy), fontsize=6.5, color=col, xytext=(3, 3), textcoords="offset points", fontweight="bold", zorder=10)
                fx = lx[tail]; fy = ly[tail]
                good = np.isfinite(fx)
                # arrow on the last finite segment
                idx = np.where(good)[0]
                if len(idx) >= 2 and idx[-1] - idx[-2] == 1:
                    ax.annotate("", xy=(fx[idx[-1]], fy[idx[-1]]), xytext=(fx[idx[-2]], fy[idx[-2]]),
                                arrowprops=dict(arrowstyle="-|>", color=col, lw=2.2, mutation_scale=16), zorder=10)
            here = np.where(ti == i)[0]
            if len(here):
                ax.plot(wrap(tr[here[0]][1]), tr[here[0]][2], "o", color=col, ms=9, mec="k", mew=1.2, zorder=11)

# --- 1. forecast panel: global
days = MAP_DAYS
fig, axs = plt.subplots(len(days), 1, figsize=(13, 3.2 * len(days)), constrained_layout=True)
for ax, i in zip(axs, days):
    k = i - nt + 1
    ttl = f"{pd.Timestamp(t_all[i]).date()}  –  " + ("Observed (latest analysis)" if k <= 0 else f"Forecast +{k} days")
    cf = draw_map(ax, i, title=ttl)
fig.colorbar(cf, ax=axs, orientation="horizontal", fraction=0.02, pad=0.02, label="OLR anomaly (W/m²) – blue = enhanced convection   |   arrows: ECMWF 850 hPa wind (trade-wind / monsoon flow)")
axs[0].text(0.005, 1.10, "Scorpion tracks: solid = path of each wave's convective centre over the 30 d before the map date, dashed + arrow = its path after the map date (+5/+10 markers), big dot = position on the map date",
            transform=axs[0].transAxes, fontsize=8, va="bottom", style="italic")
stamp(fig, "Global Tropical Wave Forecast (30S–30N) with Scorpion tracks – ECMWF IFS data")
imgs["global"] = b64(fig)

# --- 2. regional panel: Arabia / Indian Ocean
fig, axs = plt.subplots(2, 3, figsize=(15, 8.2))
for ax, i in zip(axs.ravel(), days):
    k = i - nt + 1
    draw_map(ax, i, 25, 110, -20, 30, f"{pd.Timestamp(t_all[i]).date()} ({'Observed' if k <= 0 else f'Forecast +{k}d'})")
stamp(fig, "Arabian Peninsula & Indian Ocean – ECMWF IFS data")
imgs["arabia"] = b64(fig)

# --- 3. Hovmöller : (a) OLR anomaly + wave forecast   (b) 850 hPa zonal wind U (analysis + ECMWF forecast)
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
i0 = nt - HOV_BACK
raw = waves.meridional_mean(anom, lat, -10, 10)
pm = ax.pcolormesh(lon, np.arange(i0, nt), raw[i0:], cmap="RdBu", vmin=-50, vmax=50, shading="auto")
# forecast window: shade with the sum of the filtered waves (statistical extrapolation)
tot_h = sum(waves.meridional_mean(filt[w], lat, -10, 10) for w in SHOW)
ax.pcolormesh(lon, np.arange(nt - 1, nt + NF), tot_h[nt - 1:], cmap="RdBu", vmin=-50, vmax=50, shading="auto")
for w in SHOW:
    m = waves.meridional_mean(filt[w], lat, -10, 10)
    ax.contour(lon, np.arange(i0, nt + NF), m[i0:], levels=[-40, -30, -20, -10, -5], colors=COL[w], linewidths=1.3)
ax.axhline(nt - 1, color="k", ls="--"); ax.text(2, nt, "↑ Forecast: sum of filtered waves (statistical extrapolation)", fontsize=8, fontweight="bold")
ax.axvline(49, color="gold", lw=1.5); ax.text(50, i0 + 1, "Hadramout", color="goldenrod")
yt = np.arange(i0, nt + NF, 5); ax.set_yticks(yt); ax.set_yticklabels([str(pd.Timestamp(t_all[j]).date()) for j in yt])
ax.set_xlabel("Longitude (°E)"); ax.set_title("(a) OLR anomaly 10°S–10°N (shading) + wave filters (contours)", fontsize=10)
ax.set_ylim(i0, nt + NF - 1)
fig.colorbar(pm, ax=ax, label="OLR anomaly (W/m²)", shrink=0.8)

# (b) U850
UH = np.full((nt + NF - i0, len(WLON)), np.nan, np.float32)
band = (WLAT >= -10) & (WLAT <= 10)
for k, j in enumerate(range(i0, nt + NF)):
    d = pd.Timestamp(t_all[j]).date()
    if d in WIND_HOV:
        UH[k] = WIND_HOV[d][0][band].mean(0)
# fill isolated missing days by time interpolation
for c in range(UH.shape[1]):
    col = UH[:, c]; bad = np.isnan(col)
    if bad.any() and (~bad).sum() > 2:
        col[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
pm2 = ax2.pcolormesh(WLON, np.arange(i0, nt + NF), UH, cmap="PuOr_r", vmin=-15, vmax=15, shading="auto")
cs = ax2.contour(WLON, np.arange(i0, nt + NF), UH, levels=[0], colors="k", linewidths=0.8)
for w in ["MJO", "Kelvin"]:
    m = waves.meridional_mean(filt[w], lat, -10, 10)
    ax2.contour(lon, np.arange(i0, nt + NF), m[i0:], levels=[-30, -20, -10], colors=COL[w], linewidths=1.2)
ax2.axhline(nt - 1, color="k", ls="--"); ax2.text(2, nt, "↑ ECMWF dynamical forecast (to +15 d)", fontsize=8, fontweight="bold")
ax2.axvline(49, color="gold", lw=1.5); ax2.text(50, i0 + 1, "Hadramout", color="goldenrod")
ax2.set_yticks(yt); ax2.set_yticklabels([str(pd.Timestamp(t_all[j]).date()) for j in yt])
ax2.set_xlabel("Longitude (°E)"); ax2.set_ylim(i0, nt + NF - 1)
ax2.set_title("(b) 850 hPa zonal wind U 10°S–10°N: westerly (+, orange) / easterly (−, purple); MJO & Kelvin contours", fontsize=10)
fig.colorbar(pm2, ax=ax2, label="U850 (m/s)", shrink=0.8)
stamp(fig, "Time–Longitude Hovmöller: OLR & U850 – ECMWF IFS data")
imgs["hov"] = b64(fig)

# --- 4. phase diagrams
def phase_ax(ax, x, y, split, labels, title, r=3.5):
    for ang in range(0, 360, 45):
        a = np.deg2rad(ang); ax.plot([np.cos(a), r * np.cos(a)], [np.sin(a), r * np.sin(a)], color="#bbb", lw=1)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#888"))
    for i, lab in enumerate(labels):
        a = np.deg2rad(22.5 + 45 * i); ax.text(2.9 * np.cos(a), 2.9 * np.sin(a), lab, ha="center", va="center", fontsize=8, color="#555")
    ax.plot(x[:split], y[:split], "-", color="#333", lw=1.2)
    ax.scatter(x[:split], y[:split], c=np.arange(split), cmap="viridis", s=18, zorder=3)
    if split < len(x):
        ax.plot(x[split - 1:], y[split - 1:], "r--o", ms=3, lw=1.2, label="Forecast")
    ax.plot(x[split - 1], y[split - 1], "*", color="gold", ms=16, mec="k", label="Today")
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_aspect("equal"); ax.axhline(0, color="#999", lw=.6); ax.axvline(0, color="#999", lw=.6)
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.legend(loc="lower left", fontsize=8)

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
mx, my, mamp, mph, mlon = waves.mjo_proxy_index(filt["MJO"], lat, lon)
i0 = nt - 40
phase_ax(a1, mx[i0:], my[i0:], nt - i0,
         ["5 Maritime Cont.", "6 W. Pacific", "7 W. Pacific", "8 W. Hemisphere", "1 Africa", "2 Indian Ocean", "3 Indian Ocean", "4 Maritime Cont."],
         f"MJO – SCORPIO RMM-like index (30–96 d filtered OLR)\nlast 40 days to {t[-1]} + {NF}-day forecast (red)")
a1.set_xlabel("RMM1-like"); a1.set_ylabel("RMM2-like")
bx, by, bamp, bph = waves.bsiso_proxy_index(filt["BSISO"], lat, lon)
phase_ax(a2, bx[i0:], by[i0:], nt - i0, ["1", "2", "3", "4", "5", "6", "7", "8"],
         f"BSISO – SCORPIO index (25–90 d filtered OLR)\nlast 40 days to {t[-1]} + {NF}-day forecast (red)")
a2.set_xlabel("−OLR equatorial Indian Ocean (70–100E)"); a2.set_ylabel("−OLR Bay of Bengal / India (12–22N)")
fig.tight_layout(rect=[0, 0, 1, 0.95])
stamp(fig, "MJO & BSISO Phase Diagrams – ECMWF IFS data")
imgs["phase"] = b64(fig)

# --- 5. local point (Radaa) + amplitude
HB_LAT = (lat >= 14) & (lat <= 17); HB_LON = (lon >= 47) & (lon <= 51)
def box(f): return f[:, HB_LAT][:, :, HB_LON].mean((1, 2))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(13, 8))
tot = np.zeros(nt + NF)
xs = np.arange(nt - 45, nt + NF)
for w in SHOW:
    s = box(filt[w]); tot += s
    a1.plot(xs, s[nt - 45:], color=COL[w], label=EN[w])
a1.plot(xs, tot[nt - 45:], "k", lw=2.5, label="Total (all waves)")
a1.bar(np.arange(nt - 45, nt), box(anom)[nt - 45:], color="gray", alpha=.3, label="Raw anomaly (obs)")
a1.axvline(nt - 1, color="k", ls="--"); a1.axhspan(-100, -10, color="blue", alpha=.05); a1.set_ylim(-80, 80)
a1.set_ylabel("OLR anomaly (W/m²)\nnegative = convection/rain"); a1.legend(ncol=3, fontsize=8)
a1.set_title("Hadramout / Yemen (14–17N, 47–51E area mean) – contribution of each wave + forecast", fontweight="bold")
for w in SHOW:
    a = waves.wave_amplitude(filt[w], lat, lon, (40, 100, -10, 25))
    a2.plot(xs, a[nt - 45:], color=COL[w], label=EN[w])
a2.axvline(nt - 1, color="k", ls="--"); a2.set_ylabel("RMS amplitude (W/m²)")
a2.set_title("Wave amplitude over Indian Ocean / Arabian Peninsula (40–100E, 10S–25N)", fontweight="bold"); a2.legend(fontsize=8)
for ax in (a1, a2):
    tk = xs[::7]; ax.set_xticks(tk); ax.set_xticklabels([str(pd.Timestamp(t_all[j]).date())[5:] for j in tk])
stamp(fig, "Local Outlook – Hadramout, Yemen – ECMWF IFS data")
fig.tight_layout(rect=[0, 0, 1, 0.97])
imgs["local"] = b64(fig)


# =====================================================================  ADVANCED ("Scorpion" tool-kit)
# --- 6. VP200 anomaly + 200 hPa wind (upper-level outflow) : obs, +5, +10 d
adv_days_sorted = sorted(ADV)
if len(ADV) >= 10:
    chi_all = {}
    for d in adv_days_sorted:
        chi_all[d] = advanced.velocity_potential(ADV[d][("d", "200")].astype(float), WLAT, WLON)
    an_days = [d for d in adv_days_sorted if d <= pd.Timestamp(t[-1]).date()]
    chi_mean = np.mean([chi_all[d] for d in an_days], 0)
    u2m = np.mean([ADV[d][("u", "200")] for d in an_days], 0); v2m = np.mean([ADV[d][("v", "200")] for d in an_days], 0)
    pick = [nt - 1, nt + 4, nt + 9, nt + 14]
    fig, axs = plt.subplots(len(pick), 1, figsize=(13, 3.3 * len(pick)), constrained_layout=True)
    for ax, i in zip(axs, pick):
        d = pd.Timestamp(t_all[i]).date()
        if d not in chi_all:
            ax.set_visible(False); continue
        chia = (chi_all[d] - chi_mean) / 1e6
        chia = chia - chia.mean(1, keepdims=True)          # remove zonal mean → planetary wave pattern
        vmax = max(6, np.percentile(np.abs(chia), 98))
        lv = np.linspace(-vmax, vmax, 13)
        cf = ax.contourf(WLON, WLAT, chia, levels=lv, cmap="BrBG_r", extend="both")
        ax.contour(WLON, WLAT, chia, levels=lv[lv < 0][::2], colors="teal", linewidths=0.8)
        ax.contour(WLON, WLAT, chia, levels=lv[lv > 0][::2], colors="saddlebrown", linewidths=0.8, linestyles="dashed")
        ua = ADV[d][("u", "200")] - u2m; va = ADV[d][("v", "200")] - v2m
        q = ax.quiver(WLON[::3], WLAT[::3], ua[::3, ::3], va[::3, ::3], color="k", scale=500, width=0.0014, pivot="mid")
        ax.quiverkey(q, 0.93, 1.04, 10, "10 m/s", labelpos="W", fontproperties={"size": 7}, coordinates="axes")
        # MJO/Kelvin OLR contours for coupling
        ax.contour(lon, lat, filt["MJO"][i], levels=[-30, -20, -10], colors=COL["MJO"], linewidths=1.4)
        ax.contour(lon, lat, filt["Kelvin"][i], levels=[-30, -20, -10], colors=COL["Kelvin"], linewidths=1.0)
        ax.plot(coast["x"], coast["y"], "k", lw=0.5); ax.plot(48.8, 15.9, "*", color="gold", ms=12, mec="k")
        k = i - nt + 1
        ax.set_xlim(0, 360); ax.set_ylim(-30, 30)
        ax.set_title(f"{d} – " + ("Observed" if k <= 0 else f"ECMWF forecast +{k} d"), fontsize=10, fontweight="bold")
    fig.colorbar(cf, ax=axs, orientation="horizontal", fraction=0.02, pad=0.02,
                 label="VP200 anomaly, zonal mean removed (10⁶ m²/s): green = upper-level DIVERGENCE (rising, MJO/Kelvin active), brown = convergence   |   arrows: 200 hPa wind anomaly")
    stamp(fig, "Upper-level outflow: 200 hPa velocity potential & wind – ECMWF IFS")
    imgs["vp200"] = b64(fig)

    # --- 7. Kelvin-mode projection Hovmöller (Hough structure on u850 & Φ850) + VP200 eq. Hovmöller
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    nd = len(adv_days_sorted)
    KP = np.full((nd, len(WLON)), np.nan); VPH = np.full((nd, len(WLON)), np.nan)
    band = np.abs(WLAT) <= 10
    for k, d in enumerate(adv_days_sorted):
        u8 = ADV[d][("u", "850")].astype(float); z8 = ADV[d][("gh", "850")].astype(float)
        pr, _ = advanced.kelvin_projection(u8, z8, WLAT, WLON, h_eq=25)
        KP[k] = pr
        _c = (chi_all[d] - chi_mean) / 1e6; _c = _c - _c.mean(1, keepdims=True)
        VPH[k] = _c[band].mean(0)
    KP = KP - np.nanmean(KP, 0)                    # remove time-mean (monsoon) → Kelvin anomaly
    KP = KP / (np.nanstd(KP) or 1)
    yy = np.arange(nd)
    pm = a1.pcolormesh(WLON, yy, KP, cmap="RdBu_r", vmin=-3, vmax=3, shading="auto")
    a1.contour(WLON, yy, KP, levels=[1, 2], colors="k", linewidths=0.7)
    # overlay OLR Kelvin-filtered contours on the same dates
    kel = waves.meridional_mean(filt["Kelvin"], lat, -10, 10)
    idx = [int(np.where(t_all == np.datetime64(d))[0][0]) for d in adv_days_sorted]
    a1.contour(lon, yy, kel[idx], levels=[-30, -20, -10], colors=COL["Kelvin"], linewidths=1.3)
    n_an = len(an_days)
    a1.axhline(n_an - 0.5, color="k", ls="--"); a1.text(2, n_an, "↑ ECMWF forecast", fontsize=8, fontweight="bold")
    ytk = yy[::5]; a1.set_yticks(ytk); a1.set_yticklabels([str(adv_days_sorted[j]) for j in ytk])
    a1.set_xlabel("Longitude (°E)"); a1.axvline(49, color="gold", lw=1.5)
    a1.set_title("(a) Kelvin-mode projection of (u, Φ) at 850 hPa [Hough ψ=exp(−y²/2), hₑ=25 m]\nstandardized; red = westerly Kelvin phase; blue contours = OLR Kelvin filter", fontsize=9)
    fig.colorbar(pm, ax=a1, label="σ", shrink=0.8)
    # 3-day running mean in time + robust scaling
    VPs = VPH.copy()
    for k in range(nd):
        lo_, hi_ = max(0, k - 1), min(nd, k + 2)
        VPs[k] = np.nanmean(VPH[lo_:hi_], 0)
    vlim = np.nanpercentile(np.abs(VPs), 97)
    pm2 = a2.pcolormesh(WLON, yy, VPs, cmap="BrBG_r", vmin=-vlim, vmax=vlim, shading="auto")
    mj = waves.meridional_mean(filt["MJO"], lat, -10, 10)
    a2.contour(lon, yy, mj[idx], levels=[-30, -20, -10], colors=COL["MJO"], linewidths=1.4)
    a2.axhline(n_an - 0.5, color="k", ls="--"); a2.text(2, n_an, "↑ ECMWF forecast", fontsize=8, fontweight="bold")
    a2.set_yticks(ytk); a2.set_yticklabels([str(adv_days_sorted[j]) for j in ytk]); a2.axvline(49, color="gold", lw=1.5)
    a2.set_xlabel("Longitude (°E)")
    a2.set_title("(b) VP200 anomaly 10°S–10°N (zonal mean removed, 3-day mean): green = divergent outflow; red contours = OLR MJO filter", fontsize=9)
    fig.colorbar(pm2, ax=a2, label="10⁶ m²/s", shrink=0.8)
    stamp(fig, "Kelvin-mode projection & VP200 Hovmöller – ECMWF IFS")
    imgs["kelvinproj"] = b64(fig)

# --- 7b. SCORPION PLOT : RMM-like phase space with observed body + multiple forecast tails
WAVE_SETS = [["MJO"], ["MJO", "Kelvin", "ER"], ["MJO", "ER"], ["MJO", "BSISO"]]
# ECMWF dynamical OLR forecast (daily mean ttr) for the tail – fetched from latest cycle if available
ecm_olr_f = None
try:
    from scorpio import ecmwf as _e
    _cyc = WIND_CYCLE
    _fd = []
    for k in range(1, NF + 1):
        d = pd.Timestamp(t_all[nt - 1 + k]).date()
        fld = _e.fetch_forecast_olr_day(_cyc, d) if hasattr(_e, "fetch_forecast_olr_day") else None
        if fld is None:
            break
        _fd.append(fld)
    if len(_fd) >= 5:
        ecm_olr_f = np.stack(_fd)
except Exception:
    ecm_olr_f = None
tracks = advanced.scorpion_tracks(anom, lat, lon, NF, WAVE_SETS, ecmwf_olr_fcst=ecm_olr_f)

fig, ax = plt.subplots(figsize=(9.5, 9.5), constrained_layout=True)
rr = 3.2
for angd in range(0, 360, 45):
    a_ = np.deg2rad(angd); ax.plot([np.cos(a_), rr * np.cos(a_)], [np.sin(a_), rr * np.sin(a_)], color="#bbb", lw=1)
ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#777", lw=1.2))
labels = ["5 Maritime Cont.", "6 W. Pacific", "7 W. Pacific", "8 W. Hemisphere", "1 W. Hem./Africa", "2 Indian Ocean", "3 Indian Ocean", "4 Maritime Cont."]
for i, lab in enumerate(labels):
    a_ = np.deg2rad(22.5 + 45 * i); ax.text(2.85 * np.cos(a_), 2.85 * np.sin(a_), lab, ha="center", va="center", fontsize=9, color="#444")
for lab, xy in (("Indian Ocean", (0, -2.6)), ("Western Pacific", (0, 2.6)), ("Western Hemisphere & Africa", (-2.6, 0)), ("Maritime Continent", (2.6, 0))):
    ax.text(*xy, lab, ha="center", va="center", fontsize=8, color="#888", style="italic", rotation=(90 if xy[1] == 0 else 0))
xo, yo, po = tracks["observed"]
n_o = len(xo)
# body: colour by time, thick
for i in range(n_o - 1):
    ax.plot(xo[i:i + 2], yo[i:i + 2], color=plt.cm.Greys(0.35 + 0.6 * i / n_o), lw=2.5, solid_capstyle="round")
ax.scatter(xo, yo, c=np.arange(n_o), cmap="Greys", s=22, zorder=4, edgecolor="k", linewidth=0.3)
obs_dates = [pd.Timestamp(d).date() for d in t_all[nt - n_o:nt]]
for i in range(0, n_o, 10):
    ax.annotate(str(obs_dates[i])[5:], (xo[i], yo[i]), fontsize=7, xytext=(4, 4), textcoords="offset points", color="#333")
# tails
TAIL_STYLE = {"MJO": ("#d62728", "-"), "MJO+Kelvin+ER": ("#1f77b4", "-"), "MJO+ER": ("#2ca02c", "--"),
              "MJO+BSISO": ("#9467bd", "--"), "ECMWF": ("black", ":")}
for name, (x, y, p) in tracks.items():
    if name == "observed":
        continue
    col, ls = TAIL_STYLE.get(name, ("gray", "-"))
    ax.plot(x, y, ls, color=col, lw=2, label=f"{name}  (+{len(x) - 1} d)")
    ax.scatter(x[::5], y[::5], color=col, s=18, zorder=5)
    ax.annotate(f"+{len(x) - 1}", (x[-1], y[-1]), fontsize=8, color=col, xytext=(4, -8), textcoords="offset points", fontweight="bold")
ax.plot(xo[-1], yo[-1], "*", color="gold", ms=20, mec="k", zorder=6, label=f"Today ({obs_dates[-1]})")
ax.set_xlim(-rr, rr); ax.set_ylim(-rr, rr); ax.set_aspect("equal")
ax.axhline(0, color="#999", lw=.6); ax.axvline(0, color="#999", lw=.6)
ax.set_xlabel("RMM1-like"); ax.set_ylabel("RMM2-like")
ax.set_title("Scorpion plot – observed track (grey body, last 40 d) and forecast tails branching from today\n"
             "statistical wave extrapolations (MJO / MJO+Kelvin+ER / MJO+ER / MJO+BSISO)" + (" + ECMWF dynamical OLR" if "ECMWF" in tracks else ""),
             fontsize=10)
ax.legend(loc="lower left", fontsize=8, framealpha=0.9)
stamp(fig, "Scorpion Plot – multi-track MJO phase space – ECMWF IFS data")
imgs["scorpion"] = b64(fig)
SCORP_ROWS = "".join(
    f"<tr><td>{n}</td><td dir='ltr'>{int(v[2][0])}</td><td dir='ltr'>{int(v[2][min(5, len(v[2]) - 1)])}</td><td dir='ltr'>{int(v[2][min(10, len(v[2]) - 1)])}</td><td dir='ltr'>{int(v[2][-1])}</td><td dir='ltr'>{np.hypot(v[0][-1], v[1][-1]):.2f}</td></tr>"
    for n, v in tracks.items() if n != "observed")

# --- 8. Wheeler–Kiladis spectrum of the current record
kk, fr, ratio, logP = advanced.wk_spectrum(anom, lat, lon, seg_len=min(96, nt), overlap=60)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
c1 = a1.contourf(kk, fr, logP, levels=20, cmap="viridis"); fig.colorbar(c1, ax=a1, label="log10 power")
c2 = a2.contourf(kk, fr, ratio, levels=np.arange(1.1, 2.05, 0.1), cmap="YlOrRd", extend="max"); fig.colorbar(c2, ax=a2, label="power / red background")
curves = advanced.dispersion_curves()
for ax in (a1, a2):
    for name, lst in curves.items():
        for (kx, fy) in lst:
            ax.plot(kx, fy, color="w" if ax is a1 else "k", lw=0.8)
    ax.axvline(0, color="gray", lw=0.6)
    for per in (3, 6, 30, 96):
        ax.axhline(1 / per, color="gray", lw=0.5, ls=":"); ax.text(14.2, 1 / per, f"{per} d", fontsize=7, va="bottom", ha="right")
    ax.set_xlim(-15, 15); ax.set_ylim(0, 0.5); ax.set_xlabel("zonal wavenumber (east > 0)"); ax.set_ylabel("frequency (cpd)")
    # wave boxes
    for w in ["MJO", "Kelvin", "ER"]:
        dd = waves.WAVES[w]
        ax.add_patch(plt.Rectangle((dd["kmin"], 1 / dd["pmax"]), dd["kmax"] - dd["kmin"], 1 / dd["pmin"] - 1 / dd["pmax"],
                                   fill=False, ec=COL[w], lw=1.5, ls="--"))
        ax.text(dd["kmin"], 1 / dd["pmin"], w, color=COL[w], fontsize=8, va="bottom", fontweight="bold")
a1.set_title(f"(a) Symmetric OLR power (15°S–15°N), {nt}-day ECMWF record", fontsize=10)
a2.set_title("(b) Signal / red background (smoothed) – shaded ≥ 1.1; > 1.3 ≈ significant wave activity; curves: Kelvin, ER, MRG (h=8, 25, 90 m)", fontsize=9)
stamp(fig, "Wheeler–Kiladis wavenumber–frequency spectrum – ECMWF IFS")
imgs["wk"] = b64(fig)

# --- 9. Hindcast skill of the statistical wave forecast
skill_glob, ncase = advanced.hindcast_skill(anom, lat, lon, SHOW, nfcst=NF, n_cases=10)
skill_reg, _ = advanced.hindcast_skill(anom, lat, lon, SHOW, nfcst=NF, n_cases=10, box=(30, 110, -15, 30))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
ld = np.arange(1, NF + 1)
for ax, sk, ttl in ((a1, skill_glob, "Tropics 20°S–20°N"), (a2, skill_reg, "Arabian Peninsula / Indian Ocean (30–110E)")):
    for w in SHOW:
        ax.plot(ld, sk[w], "-o", ms=4, color=COL[w], label=EN[w])
    ax.axhline(0.5, color="gray", ls="--", lw=1); ax.text(NF, 0.51, "useful skill (r = 0.5)", ha="right", fontsize=8, color="gray")
    ax.set_ylim(-0.2, 1.05); ax.set_xlim(1, NF); ax.set_xlabel("forecast lead (days)"); ax.set_ylabel("pattern correlation")
    ax.set_title(f"{ttl} – {ncase} hindcast cases from this record", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
stamp(fig, "Hindcast skill of SCORPIO wave forecasts")
imgs["skill"] = b64(fig)
SKILL_ROWS = "".join(f"<tr><td>{EN[w]}</td><td dir='ltr'>{next((int(l+1) for l in range(NF) if skill_reg[w][l] < 0.5), NF)}</td><td dir='ltr'>{skill_reg[w][4]:.2f}</td><td dir='ltr'>{skill_reg[w][9]:.2f}</td></tr>" for w in SHOW)

# --- outlook table
rows = []
for k in range(NF):
    i = nt + k; v = tot[i]
    dom = max(SHOW, key=lambda w: abs(box(filt[w])[i]))
    state = "🌧️ حمل نشط مرجّح" if v < -10 else ("☀️ قمع/جفاف مرجّح" if v > 10 else "⛅ محايد")
    rows.append((str(pd.Timestamp(t_all[i]).date()), f"{v:+.1f}", waves.WAVES[dom]["name_ar"], state))
table = "".join(f"<tr><td>{a}</td><td dir='ltr'>{b}</td><td>{c}</td><td>{d}</td></tr>" for a, b, c, d in rows)

_st = {}
try:
    import json as _json
    _st = _json.load(open(os.path.join(os.path.dirname(__file__), "data", "status.json")))
except Exception:
    pass
STATUS_TXT = (f"التحديث القادم ≈ {str(_st.get('next_run', ''))[:16]} UTC" if _st.get("next_run") else "")

html = f"""<!DOCTYPE html><html lang="ar" dir="rtl"><head><meta charset="utf-8">
<title>SCORPIO – توقعات الأمواج الاستوائية {t[-1]}</title>
<style>
body{{font-family:Segoe UI,Tahoma,sans-serif;background:#0e1b2a;color:#eee;margin:0;padding:18px}}
h1{{margin:0;font-size:1.7rem}} h2{{color:#ffd166;border-right:5px solid #d62728;padding-right:10px;margin-top:34px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}}
.card{{background:#16283d;border-radius:12px;padding:12px 18px;min-width:180px}}
.card b{{display:block;font-size:1.25rem;color:#7fd1ff}} .card span{{font-size:.8rem;color:#9fb3c8}}
img{{width:100%;max-width:1400px;border-radius:10px;background:#fff;display:block}}
table{{border-collapse:collapse;width:100%;max-width:900px;background:#16283d}} td,th{{border:1px solid #2a3f58;padding:6px 10px;text-align:right}}
th{{background:#1f3550}} .ok{{background:#1b5e20;padding:8px 14px;border-radius:8px;display:inline-block}}
.leg span{{display:inline-block;margin-left:16px}} .sq{{display:inline-block;width:14px;height:14px;margin-left:4px;vertical-align:middle;border-radius:3px}}
small{{color:#9fb3c8}}
</style></head><body>
<h1>🦂 SCORPIO – خرائط توقعات الأمواج الاستوائية (ECMWF) – Scorpion Advanced <small style="font-size:1rem;color:#ffd166">| Designed by Ahmed Omar Dhafer</small></h1>
<small>MJO • BSISO • كلفن • روسبي — ترشيح Wheeler–Kiladis + تمديد Wheeler–Weickmann | أُنشئ {dt.datetime.now():%Y-%m-%d %H:%M}</small>
<div class="cards">
<div class="card"><b>{t[-1]}</b><span>آخر رصد OLR</span></div>
<div class="card"><b>{t_all[-1]}</b><span>نهاية التوقع (+{NF} يوم)</span></div>
<div class="card"><b>طور {int(mph[nt-1])} – سعة {mamp[nt-1]:.2f}</b><span>MJO – مؤشر SCORPIO (آخر رصد {t[-1]})</span></div>
<div class="card"><b>طور {int(bph[nt-1])} – سعة {bamp[nt-1]:.2f}</b><span>BSISO – تقدير SCORPIO</span></div>
</div>
<div class="ok">✅ بيانات حقيقية محدَّثة تلقائياً — {src}</div>
<p><small>🔁 التحديث التلقائي: كل 6 ساعات عبر <code>scheduler.py</code> — آخر بناء {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC | دورة رياح ECMWF: {WIND_CYCLE} | {STATUS_TXT}</small></p>
<p class="leg">{"".join(f'<span><i class="sq" style="background:{COL[w]}"></i>{waves.WAVES[w]["name_ar"]}</span>' for w in SHOW)}
<span>⭐ حضرموت</span> <span>➜ أسهم: رياح 850 hPa (التجارية/الموسمية) من ECMWF – مقياس 10 m/s</span> <span>🦂 مسارات Scorpion: خط متصل = مسار مركز الحمل المرصود للموجة، خط متقطع بسهم = مساره المتوقع (+5/+10/+15)، نقطة كبيرة = موقعه بتاريخ الخريطة</span> — التظليل الأزرق = حمل نشط/أمطار، الأحمر = قمع. الخط المتصل = مركز الحمل النشط للموجة، المنقّط = طورها القامع.</p>
<h2>1. خريطة التوقع العالمية (30°S–30°N) + رياح 850 hPa + مسارات Scorpion</h2><img src="{imgs['global']}">
<h2>2. شبه الجزيرة العربية والمحيط الهندي + رياح 850 hPa</h2><img src="{imgs['arabia']}">
<h2>3. مخطط هوفمولر (الزمن × خط الطول): OLR والرياح المدارية U عند 850 hPa مع التوقع</h2><img src="{imgs['hov']}">
<p><small>(a) شذوذ OLR مع امتداد الأمواج إحصائياً فوق الخط المتقطع. (b) الرياح المدارية U عند 850 hPa: البرتقالي = رياح غربية (تسبق/ترافق الطور النشط لـ MJO وكلفن)، البنفسجي = رياح شرقية (تجارية)؛ الجزء فوق الخط المتقطع هو التوقع الديناميكي للنموذج الأوروبي حتى +15 يوم. الميل نحو اليمين مع الزمن = انتشار شرقي، نحو اليسار = انتشار غربي.</small></p>
<h2>4. أطوار MJO و BSISO (حتى آخر رصد + توقع 15 يوماً)</h2><img src="{imgs['phase']}">
<h2>5. التوقع المحلي – حضرموت/اليمن</h2><img src="{imgs['local']}">
<h3>جدول التوقع اليومي – حضرموت (متوسط 14–17N, 47–51E)</h3>
<table><tr><th>التاريخ</th><th>مجموع الأمواج (W/m²)</th><th>الموجة المهيمنة</th><th>الحالة</th></tr>{table}</table>
<h2>6. التدفق العلوي: جهد السرعة 200 hPa ورياح 200 hPa (ECMWF)</h2>
{('<img src="' + imgs['vp200'] + '">') if 'vp200' in imgs else '<p>غير متاح في هذا التشغيل</p>'}
<p><small>الأخضر (VP200 سالب) = تباعد علوي وصعود عام مرافق للطور النشط لـ MJO/كلفن؛ البني = تقارب علوي وهبوط. الأسهم شذوذ رياح 200 hPa. كونتورات MJO (أحمر) وكلفن (أزرق) من OLR للتحقق من الاقتران الرأسي.</small></p>
<h2>7. إسقاط نمط كلفن (دوال Hough) وهوفمولر VP200</h2>
{('<img src="' + imgs['kelvinproj'] + '">') if 'kelvinproj' in imgs else '<p>غير متاح في هذا التشغيل</p>'}
<p><small>(a) إسقاط (u, Φ) عند 850 hPa على البنية العرضية النظرية لموجة كلفن ψ=exp(−y²/2) — تقنية التعرف المحلي على كلفن المستخدمة مع تحليلات ECMWF (بدون ترشيح زمني مسبق)، ويمتد على توقع ECMWF الديناميكي. (b) VP200 استوائي: انتشار شرقي للتباعد العلوي = بصمة MJO.</small></p>
<h2>7b. Scorpion Plot – فضاء طور MJO متعدد المسارات</h2><img src="{imgs['scorpion']}">
<table><tr><th>مسار التوقع</th><th>الطور اليوم</th><th>+5 أيام</th><th>+10 أيام</th><th>+15 يوم</th><th>السعة عند +15</th></tr>{SCORP_ROWS}</table>
<p><small>الجسم الرمادي = المسار المرصود (40 يوماً، شذوذ OLR غير مرشَّح بمتوسط 11 يوماً). الذيول الملوّنة تتفرّع من اليوم (⭐): امتدادات إحصائية بتركيبات موجية مختلفة — تباعُد الذيول = عدم يقين التوقع؛ تقاربها = ثقة أعلى. داخل الدائرة (سعة < 1) = MJO ضعيف. الأطوار مرتبطة بمواقع الحمل: 2–3 المحيط الهندي (مطر مرجّح لجنوب شبه الجزيرة)، 8–1 نصف الكرة الغربي/أفريقيا.</small></p>
<h2>8. طيف Wheeler–Kiladis للسجل الحالي</h2><img src="{imgs['wk']}">
<p><small>يحدد أي الأمواج نشطة فعلاً في الفترة الحالية: القيم > 1.3 داخل صندوق موجة ما تعني إشارة معنوية، وتزيد الثقة بتوقع تلك الموجة.</small></p>
<h2>9. التحقق من مهارة التوقع (Hindcast)</h2><img src="{imgs['skill']}">
<table><tr><th>الموجة</th><th>أيام المهارة المفيدة (r ≥ 0.5) – المنطقة العربية</th><th>r عند +5 أيام</th><th>r عند +10 أيام</th></tr>{SKILL_ROWS}</table>
<p><small>يُعاد تشغيل التوقع من 10 تواريخ سابقة في السجل ويُقارن بالحقل المرشَّح الكامل؛ تُستخدم النتيجة لتحديد المدى الذي يجب الوثوق به لكل موجة.</small></p>
<h2>المنهجية باختصار</h2>
<p>OLR يومي من النموذج الأوروبي ECMWF IFS HRES 0.25° (Open Data، محدَّث يومياً؛ متوسط دورات 00/06/12/18 UTC للحقل ttr). شذوذات ← FFT ثنائي (زمن×خط طول) ← عزل كل موجة في مجال (k, ω) وفق منحنيات تشتت Matsuno ← حشو صفري بعد آخر رصد يعطي امتداداً إحصائياً (Wheeler & Weickmann 2001). الأسهم = رياح 850 hPa (متوسط يومي) من آخر دورة تشغيل ECMWF HRES: تحليل لليوم الأخير وتوقع ديناميكي حتى 360 ساعة للأيام التالية؛ تُظهر الرياح التجارية الشمالية الشرقية/الجنوبية الشرقية وتدفق الرياح الموسمية الجنوبية الغربية وتقاربها مع مناطق الحمل النشط. مهارة متوقعة: MJO/BSISO ≈ 2–3 أسابيع، كلفن/روسبي ≈ أسبوع. الأيام الأخيرة من التوقع أقل موثوقية.</p>
</body></html>"""
out = os.environ.get("SCORPIO_OUT") or os.path.join(os.path.dirname(__file__), "SCORPIO_dashboard.html")
open(out, "w", encoding="utf-8").write(html)
print("written", out, len(html) // 1024, "KB")
