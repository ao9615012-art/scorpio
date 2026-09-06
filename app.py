"""
SCORPIO – Space-time Convectively-coupled wave Outlook & Real-time Propagation
              Index Observer
Streamlit application – tropical wave forecast maps (MJO / BSISO / Kelvin / ER / MRG / TD)
"""
import sys, os, datetime as dt
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorpio import data, waves, ecmwf, ecmwf_wind

st.set_page_config(page_title="SCORPIO – خرائط توقعات الأمواج الاستوائية", page_icon="🦂",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
html, body, [class*="css"] {direction: rtl;}
.stApp {font-family: "Segoe UI", Tahoma, sans-serif;}
h1,h2,h3 {text-align:right}
.metric-card{background:#0e1b2a;border-radius:12px;padding:10px 14px;color:#fff;margin-bottom:6px}
.small{font-size:0.8rem;color:#999}
</style>""", unsafe_allow_html=True)

COAST = np.load(os.path.join(os.path.dirname(__file__), "data", "coast110.npz"))


# ------------------------------------------------------------------ cached loaders
@st.cache_data(ttl=3 * 3600, show_spinner="⏳ تحميل/تحديث بيانات OLR من النموذج الأوروبي ECMWF …")
def get_olr(ndays, _stamp):
    return ecmwf.load_olr_ecmwf(ndays)


@st.cache_data(ttl=3 * 3600, show_spinner="🌬️ تحميل رياح 850 hPa (تحليل + توقع ECMWF) …")
def get_wind(day_list, _stamp):
    w, cyc = ecmwf_wind.wind_for_days(list(day_list), workers=3)
    return w, cyc


@st.cache_resource
def start_auto_updater():
    """Background thread: every 3 h pull new GFS days + indices, then clear caches."""
    import threading, time as _time
    def loop():
        while True:
            _time.sleep(3 * 3600)
            try:
                new = ecmwf.update(dt.date.today() - dt.timedelta(days=15))
                if new:
                    st.cache_data.clear()
            except Exception:
                pass
    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th


start_auto_updater()


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def get_indices():
    try:
        omi = data.load_omi()
    except Exception as e:
        omi = None
    try:
        bs = data.load_bsiso()
    except Exception:
        bs = None
    return omi, bs


@st.cache_data(show_spinner="🌀 ترشيح طيفي (Wheeler–Kiladis) + تمديد توقعي …")
def run_filter(olr, lon, wave_list, nfcst, _key):
    a = waves.anomalies(olr)
    return a, waves.filter_forecast(a, lon, wave_list, nfcst=nfcst)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🦂 SCORPIO")
    st.caption("Space-time Convectively-coupled wave Outlook & Real-time Propagation Index Observer")
    st.markdown("---")
    nfcst = st.slider("أيام التوقع", 5, 25, 15)
    ndays = st.select_slider("طول السجل المستخدم (يوم)", [180, 240, 300, 365], value=240)
    sel_waves = st.multiselect("الأمواج المعروضة", list(waves.WAVES),
                               default=["MJO", "Kelvin", "ER", "BSISO"],
                               format_func=lambda w: waves.WAVES[w]["name_ar"])
    region = st.selectbox("النطاق الجغرافي", ["العالم الاستوائي", "شبه الجزيرة العربية والمحيط الهندي",
                                              "أفريقيا", "آسيا الموسمية", "المحيط الهادئ"])
    contour_lvl = st.slider("عتبة الشذوذ (W/m²)", 5, 30, 10)
    show_wind = st.checkbox("أسهم رياح 850 hPa (ECMWF)", value=True)
    st.markdown("---")
    if st.button("🔄 إعادة تحميل البيانات"):
        with st.spinner("جلب أحدث أيام ECMWF …"):
            ecmwf.update(dt.date.today() - dt.timedelta(days=20))
        st.cache_data.clear()
        st.rerun()
    st.caption("🔁 التحديث التلقائي كل 3 ساعات من ECMWF Open Data")
    st.markdown('<p class="small">المنهجية: ترشيح طيفي زمكاني (Wheeler & Kiladis 1999) '
                'مع تمديد إحصائي بالحشو الصفري (Wheeler & Weickmann 2001). '
                'المؤشرات: OMI من NOAA PSL، BSISO من IPRC/Kikuchi.</p>', unsafe_allow_html=True)

REGIONS = {
    "العالم الاستوائي": (0, 360, -30, 30),
    "شبه الجزيرة العربية والمحيط الهندي": (25, 110, -20, 30),
    "أفريقيا": (340, 60, -25, 25),
    "آسيا الموسمية": (50, 160, -15, 30),
    "المحيط الهادئ": (120, 290, -25, 25),
}

# ------------------------------------------------------------------ load
try:
    # stamp = today's date → the cache is naturally invalidated once per day
    t, lat, lon, olr, src, meta = get_olr(ndays, str(dt.date.today()))
except Exception as e:
    st.error(f"تعذّر تحميل بيانات OLR: {e}")
    st.stop()

all_waves = list(waves.WAVES)
anom, filt = run_filter(olr, lon, all_waves, nfcst, str(t[-1]) + str(len(t)))
nt = len(t)
t_all = np.concatenate([t, t[-1] + np.arange(1, nfcst + 1).astype("timedelta64[D]")])
omi, bsiso_df = get_indices()
WIND, WIND_CYCLE = ({}, None)
if show_wind:
    _days = tuple(pd.Timestamp(x).date() for x in t_all[nt - 1:])
    WIND, WIND_CYCLE = get_wind(_days, str(dt.date.today()))
last_obs = pd.Timestamp(t[-1]).date()

# ------------------------------------------------------------------ header
st.title("خرائط توقعات الأمواج الاستوائية – MJO • BSISO • كلفن • روسبي (ECMWF)")
st.caption("Designed by Ahmed Omar Dhafer")
c1, c2, c3, c4 = st.columns(4)
c1.metric("آخر رصد OLR", str(last_obs))
c2.metric("نهاية التوقع", str(pd.Timestamp(t_all[-1]).date()))
mx, my, mamp, mph, mlon = waves.mjo_proxy_index(filt["MJO"], lat, lon)
c3.metric(f"MJO (تقدير SCORPIO) – {last_obs}", f"طور {int(mph[nt-1])}", f"سعة {mamp[nt-1]:.2f}")
bx, by, bamp, bph = waves.bsiso_proxy_index(filt["BSISO"], lat, lon)
c4.metric("BSISO (تقدير SCORPIO)", f"طور {int(bph[nt-1])}", f"سعة {bamp[nt-1]:.2f}")
age = (dt.date.today() - last_obs).days
st.caption(f"المصدر: {src}")
if age <= 2:
    st.success(f"✅ البيانات محدَّثة تلقائياً – آخر يوم رصد قبل {age} يوم (ECMWF Open Data يصدر 4 مرات يومياً)")
else:
    st.warning(f"⚠️ آخر رصد قبل {age} يوم – اضغط «إعادة تحميل البيانات» للتحديث")

tab_map, tab_hov, tab_phase, tab_reg, tab_doc = st.tabs(
    ["🗺️ خريطة التوقع", "📈 مخطط هوفمولر", "🌀 أطوار MJO / BSISO", "📍 المنطقة المحلية", "📚 المنهجية"])


# ------------------------------------------------------------------ helpers
def sub_lon(field, lon, lon1, lon2):
    """Handle regions crossing Greenwich (lon1>lon2)."""
    if lon1 <= lon2:
        s = (lon >= lon1) & (lon <= lon2)
        return field[..., s], lon[s]
    s1 = lon >= lon1
    s2 = lon <= lon2
    f = np.concatenate([field[..., s1], field[..., s2]], -1)
    l = np.concatenate([lon[s1] - 360, lon[s2]])
    return f, l


def coast_trace(lon1, lon2):
    x, y = COAST["x"].copy(), COAST["y"]
    if lon1 > lon2:
        x = np.where(x > 180, x - 360, x)
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color="#222", width=0.8),
                      hoverinfo="skip", showlegend=False)


def map_figure(day_idx, lon1, lon2, lat1, lat2, show, thresh):
    fig = go.Figure()
    # total filtered anomaly (sum of selected waves) as shaded background
    total = sum(filt[w][day_idx] for w in show) if show else anom[min(day_idx, nt - 1)]
    tot, lsub = sub_lon(total, lon, lon1, lon2)
    fig.add_trace(go.Contour(z=tot, x=lsub, y=lat, colorscale="RdBu", zmid=0,
                             zmin=-40, zmax=40, contours=dict(start=-40, end=40, size=5,
                             showlines=False),
                             colorbar=dict(title="OLR شذوذ<br>W/m²", len=0.8),
                             name="مجموع الأمواج", hovertemplate="lon %{x}°, lat %{y}°<br>%{z:.1f} W/m²"))
    for w in show:
        f, lsub = sub_lon(filt[w][day_idx], lon, lon1, lon2)
        col = waves.WAVES[w]["color"]
        # negative anomaly (enhanced convection) solid, positive dashed
        fig.add_trace(go.Contour(z=f, x=lsub, y=lat, showscale=False,
                                 contours=dict(coloring="none", start=-thresh * 3, end=-thresh, size=thresh),
                                 line=dict(color=col, width=2), name=f"{waves.WAVES[w]['name_ar']} (حمل نشط)",
                                 hoverinfo="skip"))
        fig.add_trace(go.Contour(z=f, x=lsub, y=lat, showscale=False,
                                 contours=dict(coloring="none", start=thresh, end=thresh * 3, size=thresh),
                                 line=dict(color=col, width=1.2, dash="dot"),
                                 name=f"{waves.WAVES[w]['name_ar']} (قمع)", hoverinfo="skip"))
    fig.add_trace(coast_trace(lon1, lon2))
    d_map = pd.Timestamp(t_all[day_idx]).date()
    if show_wind and d_map in WIND:
        u, v = WIND[d_map]
        WLAT, WLON = ecmwf_wind.LAT, ecmwf_wind.LON
        regional = (lon2 - lon1) % 360 < 200
        st_ = 2 if regional else 3
        sc = 0.35 if regional else 0.6            # degrees per m/s
        xs, ys, hx, hy, txt = [], [], [], [], []
        for i in range(0, len(WLAT), st_):
            for j in range(0, len(WLON), st_):
                x0 = WLON[j]; y0 = WLAT[i]
                if lon1 > lon2 and x0 > 180:
                    x0 -= 360
                uu, vv = float(u[i, j]), float(v[i, j])
                x1 = x0 + uu * sc; y1 = y0 + vv * sc
                xs += [x0, x1, None]; ys += [y0, y1, None]
                hx.append(x1); hy.append(y1); txt.append(f"U={uu:.1f} V={vv:.1f} m/s")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="#111", width=1),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=hx, y=hy, mode="markers", marker=dict(size=4, color="#111"),
                                 text=txt, hovertemplate="%{text}", name="رياح 850 hPa (ECMWF)"))
    if lon1 > lon2:
        lon1 -= 360
    # Radaa / Yemen marker
    fig.add_trace(go.Scatter(x=[48.8], y=[15.9], mode="markers+text", text=["حضرموت"],
                             textposition="top center", marker=dict(size=9, color="gold", symbol="star"),
                             name="حضرموت – اليمن"))
    d = pd.Timestamp(t_all[day_idx]).date()
    kind = "رصد" if day_idx < nt else f"توقع +{day_idx - nt + 1} يوم"
    fig.update_layout(title=f"{d}  ({kind})", height=520, margin=dict(l=10, r=10, t=50, b=10),
                      xaxis=dict(range=[lon1, lon2], title="خط الطول"),
                      yaxis=dict(range=[lat1, lat2], title="خط العرض", scaleanchor="x", scaleratio=1),
                      legend=dict(orientation="h", y=-0.15), plot_bgcolor="#f7f7f7")
    return fig


# ------------------------------------------------------------------ TAB 1 – map
with tab_map:
    lon1, lon2, lat1, lat2 = REGIONS[region]
    colA, colB = st.columns([3, 1])
    with colB:
        st.markdown("#### اختيار اليوم")
        day_off = st.slider("يوم (0 = آخر رصد، موجب = توقع)", -20, nfcst, 0)
        day_idx = nt - 1 + day_off
        st.markdown("""
        <div class="metric-card">
        <b>قراءة الخريطة</b><br>
        • التظليل الأزرق: شذوذ OLR سالب ⇒ سحب/حمل نشط ⇒ فرصة أمطار.<br>
        • التظليل الأحمر: شذوذ موجب ⇒ قمع/جفاف.<br>
        • الخطوط المتصلة الملوّنة: مركز الحمل النشط لكل موجة.<br>
        • الخطوط المنقّطة: الطور القامع للموجة.
        </div>""", unsafe_allow_html=True)
        st.markdown("**ألوان الأمواج**")
        for w in sel_waves:
            st.markdown(f'<span style="color:{waves.WAVES[w]["color"]};font-weight:bold">■</span> {waves.WAVES[w]["name_ar"]}',
                        unsafe_allow_html=True)
    with colA:
        st.plotly_chart(map_figure(day_idx, lon1, lon2, lat1, lat2, sel_waves, contour_lvl),
                        width="stretch")

    st.markdown("#### لوحة التوقع اليومي (شريط الأيام)")
    steps = list(range(nt - 1, nt + nfcst, max(1, nfcst // 5)))
    cols = st.columns(len(steps))
    for c, i in zip(cols, steps):
        with c:
            fig = map_figure(i, lon1, lon2, lat1, lat2, sel_waves, contour_lvl)
            fig.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
                              title=dict(font=dict(size=12)), coloraxis_showscale=False)
            fig.update_traces(showscale=False, selector=dict(type="contour"))
            st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ TAB 2 – Hovmöller
with tab_hov:
    st.markdown("#### مخطط هوفمولر (الزمن × خط الطول) – متوسط 10°S–10°N")
    hw = st.selectbox("الموجة", ["الكل (شذوذ خام + أمواج)"] + sel_waves,
                      format_func=lambda w: waves.WAVES[w]["name_ar"] if w in waves.WAVES else w)
    nback = st.slider("عدد أيام الرصد المعروضة", 30, 120, 60)
    i0 = nt - nback
    fig = go.Figure()
    if hw.startswith("الكل"):
        raw = waves.meridional_mean(anom, lat, -10, 10)
        fig.add_trace(go.Heatmap(z=raw[i0:], x=lon, y=t[i0:], colorscale="RdBu", zmid=0, zmin=-50, zmax=50,
                                 colorbar=dict(title="W/m²")))
        for w in sel_waves:
            m = waves.meridional_mean(filt[w], lat, -10, 10)[i0:]
            fig.add_trace(go.Contour(z=m, x=lon, y=t_all[i0:], showscale=False,
                                     contours=dict(coloring="none", start=-45, end=-contour_lvl, size=contour_lvl),
                                     line=dict(color=waves.WAVES[w]["color"], width=1.8),
                                     name=waves.WAVES[w]["name_ar"]))
    else:
        m = waves.meridional_mean(filt[hw], lat, -10, 10)[i0:]
        fig.add_trace(go.Heatmap(z=m, x=lon, y=t_all[i0:], colorscale="RdBu", zmid=0,
                                 colorbar=dict(title="W/m²")))
    fig.add_shape(type="line", xref="paper", x0=0, x1=1, y0=str(last_obs), y1=str(last_obs),
                  line=dict(color="black", dash="dash"))
    fig.add_annotation(xref="paper", x=0.01, y=str(last_obs), text="آخر رصد ↑ توقع", showarrow=False,
                       yanchor="bottom", bgcolor="rgba(255,255,255,0.7)")
    fig.add_shape(type="line", yref="paper", y0=0, y1=1, x0=49, x1=49, line=dict(color="gold", width=1.5))
    fig.add_annotation(yref="paper", y=1.0, x=49, text="حضرموت", showarrow=False, yanchor="bottom")
    fig.update_layout(height=650, xaxis_title="خط الطول (°E)", yaxis_title="التاريخ",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig, width="stretch")
    st.caption("الميل لأعلى-يمين = انتشار شرقي (MJO/كلفن)، الميل لأعلى-يسار = انتشار غربي (روسبي/MRG/TD).")

    if show_wind:
        st.markdown("#### الرياح المدارية U عند 850 hPa (10°S–10°N) – تحليل + توقع ECMWF الديناميكي")
        hov_back = st.slider("أيام التحليل للرياح", 15, 60, 30, key="hovwind")
        hd = tuple(pd.Timestamp(x).date() for x in t_all[nt - hov_back:])
        WH, _ = get_wind(hd, str(dt.date.today()))
        WLAT, WLON = ecmwf_wind.LAT, ecmwf_wind.LON
        band = (WLAT >= -10) & (WLAT <= 10)
        UH = np.full((len(hd), len(WLON)), np.nan)
        for k, d in enumerate(hd):
            if d in WH:
                UH[k] = WH[d][0][band].mean(0)
        figu = go.Figure()
        figu.add_trace(go.Heatmap(z=UH, x=WLON, y=list(t_all[nt - hov_back:]), colorscale="PuOr_r", zmid=0,
                                  zmin=-15, zmax=15, colorbar=dict(title="U850 m/s")))
        figu.add_trace(go.Contour(z=UH, x=WLON, y=list(t_all[nt - hov_back:]), showscale=False,
                                  contours=dict(coloring="none", start=0, end=0, size=1),
                                  line=dict(color="black", width=1), name="U=0"))
        for w in [x for x in ["MJO", "Kelvin"] if x in sel_waves]:
            m = waves.meridional_mean(filt[w], lat, -10, 10)[nt - hov_back:]
            figu.add_trace(go.Contour(z=m, x=lon, y=list(t_all[nt - hov_back:]), showscale=False,
                                      contours=dict(coloring="none", start=-45, end=-contour_lvl, size=contour_lvl),
                                      line=dict(color=waves.WAVES[w]["color"], width=1.6), name=waves.WAVES[w]["name_ar"]))
        figu.add_shape(type="line", xref="paper", x0=0, x1=1, y0=str(last_obs), y1=str(last_obs),
                       line=dict(color="black", dash="dash"))
        figu.add_annotation(xref="paper", x=0.01, y=str(last_obs), text="↑ توقع ECMWF الديناميكي", showarrow=False,
                            yanchor="bottom", bgcolor="rgba(255,255,255,0.7)")
        figu.add_shape(type="line", yref="paper", y0=0, y1=1, x0=49, x1=49, line=dict(color="gold", width=1.5))
        figu.update_layout(height=600, xaxis_title="خط الطول (°E)", yaxis_title="التاريخ",
                           margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(figu, width="stretch")
        st.caption(f"برتقالي = رياح غربية، بنفسجي = رياح شرقية (تجارية). دورة التوقع: {WIND_CYCLE}")

# ------------------------------------------------------------------ TAB 3 – phase diagrams
with tab_phase:
    cA, cB = st.columns(2)

    def phase_diagram(x, y, dates, title, labels, fcst_from=None):
        fig = go.Figure()
        r = 4
        for ang in range(0, 360, 45):
            a = np.deg2rad(ang)
            fig.add_shape(type="line", x0=np.cos(a), y0=np.sin(a), x1=r * np.cos(a), y1=r * np.sin(a),
                          line=dict(color="#bbb", width=1))
        fig.add_shape(type="circle", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="#888"))
        for i, lab in enumerate(labels):
            a = np.deg2rad(22.5 + 45 * i)
            fig.add_annotation(x=3.2 * np.cos(a), y=3.2 * np.sin(a), text=lab, showarrow=False,
                               font=dict(size=11, color="#555"))
        n = len(x)
        col = np.linspace(0, 1, n)
        split = n if fcst_from is None else fcst_from
        fig.add_trace(go.Scatter(x=x[:split], y=y[:split], mode="lines+markers",
                                 marker=dict(size=5, color=col[:split], colorscale="Viridis"),
                                 line=dict(color="#333", width=1.5), name="رصد",
                                 text=[str(d) for d in dates[:split]], hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f})"))
        if fcst_from is not None:
            fig.add_trace(go.Scatter(x=x[split - 1:], y=y[split - 1:], mode="lines+markers",
                                     marker=dict(size=5, color="red"), line=dict(color="red", dash="dot"),
                                     name="توقع", text=[str(d) for d in dates[split - 1:]],
                                     hovertemplate="%{text}<br>(%{x:.2f}, %{y:.2f})"))
        fig.add_trace(go.Scatter(x=[x[split - 1]], y=[y[split - 1]], mode="markers",
                                 marker=dict(size=14, color="gold", symbol="star", line=dict(color="black", width=1)),
                                 name="اليوم"))
        fig.update_layout(title=title, height=520, xaxis=dict(range=[-r, r], zeroline=True),
                          yaxis=dict(range=[-r, r], scaleanchor="x", zeroline=True),
                          margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.1))
        return fig

    with cA:
        st.markdown("#### MJO – مؤشر SCORPIO (OLR مرشَّح 30–96 يوم) في إحداثيات RMM مع التوقع")
        i0m = nt - 40
        fig = phase_diagram(mx[i0m:], my[i0m:], [pd.Timestamp(d).date() for d in t_all[i0m:]],
                            f"40 يوم رصد حتى {last_obs} + توقع {nfcst} يوم (أحمر)",
                            ["5 القارة البحرية", "6 غرب الهادئ", "7 غرب الهادئ", "8 نصف الكرة الغربي",
                             "1 أفريقيا", "2 المحيط الهندي", "3 المحيط الهندي", "4 القارة البحرية"],
                            fcst_from=nt - i0m)
        fig.update_xaxes(title="RMM1-like")
        fig.update_yaxes(title="RMM2-like")
        st.plotly_chart(fig, width="stretch")
        if omi is not None:
            st.caption(f"للمقارنة – مؤشر OMI الرسمي (NOAA PSL) آخر تحديث {omi.index[-1]}: طور {int(omi.iloc[-1]['phase'])}, سعة {omi.iloc[-1]['amp']:.2f}")

    with cB:
        st.markdown("#### BSISO – تقدير SCORPIO من OLR المرشَّح (25–90 يوم) مع التوقع")
        i0 = nt - 40
        fig = phase_diagram(bx[i0:], by[i0:], [pd.Timestamp(d).date() for d in t_all[i0:]],
                            "40 يوم رصد + توقع (أحمر)",
                            ["1", "2", "3 شمال المحيط الهندي", "4 خليج البنغال/الهند", "5", "6", "7", "8 خط الاستواء"],
                            fcst_from=nt - i0)
        fig.update_xaxes(title="−OLR استوائي (70–100E)")
        fig.update_yaxes(title="−OLR خليج البنغال (12–22N)")
        st.plotly_chart(fig, width="stretch")
        if bsiso_df is not None:
            st.caption(f"مؤشر IPRC/Kikuchi الرسمي (آخر تحديث {bsiso_df.index[-1]}):")
            st.dataframe(bsiso_df.iloc[-5:], width="stretch")

    st.markdown("#### سعة الأمواج فوق المحيط الهندي / شبه الجزيرة العربية (40–100E, 10S–25N) مع التوقع")
    fig = go.Figure()
    for w in sel_waves:
        a = waves.wave_amplitude(filt[w], lat, lon, (40, 100, -10, 25))
        fig.add_trace(go.Scatter(x=t_all[nt - 60:nt], y=a[nt - 60:nt], name=waves.WAVES[w]["name_ar"],
                                 line=dict(color=waves.WAVES[w]["color"], width=2)))
        fig.add_trace(go.Scatter(x=t_all[nt - 1:], y=a[nt - 1:], showlegend=False,
                                 line=dict(color=waves.WAVES[w]["color"], width=2, dash="dot")))
    fig.add_shape(type="line", yref="paper", y0=0, y1=1, x0=str(last_obs), x1=str(last_obs),
                  line=dict(color="black", dash="dash"))
    fig.update_layout(height=350, yaxis_title="RMS (W/m²)", margin=dict(l=10, r=10, t=20, b=10),
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

# ------------------------------------------------------------------ TAB 4 – local point
with tab_reg:
    st.markdown("#### توقع مساهمة كل موجة عند نقطة محددة")
    c1, c2 = st.columns(2)
    plat = c1.number_input("خط العرض", -30.0, 30.0, 15.5, 0.5)
    plon = c2.number_input("خط الطول (°E)", 0.0, 360.0, 49.0, 0.5)
    ii = np.argmin(np.abs(lat - plat)); jj = np.argmin(np.abs(lon - plon))
    fig = go.Figure()
    tot = np.zeros(nt + nfcst)
    for w in sel_waves:
        s = filt[w][:, ii, jj]
        tot += s
        fig.add_trace(go.Scatter(x=t_all[nt - 45:], y=s[nt - 45:], name=waves.WAVES[w]["name_ar"],
                                 line=dict(color=waves.WAVES[w]["color"])))
    fig.add_trace(go.Scatter(x=t_all[nt - 45:], y=tot[nt - 45:], name="المجموع", line=dict(color="black", width=3)))
    fig.add_trace(go.Bar(x=t[nt - 45:], y=anom[nt - 45:, ii, jj], name="الشذوذ الخام (رصد)",
                         marker_color="rgba(120,120,120,0.35)"))
    fig.add_shape(type="line", yref="paper", y0=0, y1=1, x0=str(last_obs), x1=str(last_obs),
                  line=dict(color="black", dash="dash"))
    fig.add_hrect(y0=-200, y1=-contour_lvl, fillcolor="blue", opacity=0.05, line_width=0)
    fig.update_layout(height=450, yaxis_title="شذوذ OLR (W/m²) — سالب = حمل/أمطار", yaxis_range=[-80, 80],
                      margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")

    # simple outlook table
    rows = []
    for k in range(nfcst):
        i = nt + k
        v = tot[i]
        dom = max(sel_waves, key=lambda w: abs(filt[w][i, ii, jj])) if sel_waves else "-"
        state = ("🌧️ حمل نشط مرجّح" if v < -contour_lvl else
                 "☀️ قمع/جفاف مرجّح" if v > contour_lvl else "⛅ محايد")
        rows.append(dict(التاريخ=str(pd.Timestamp(t_all[i]).date()), **{"المجموع W/m²": round(float(v), 1)},
                         الموجة_المهيمنة=waves.WAVES[dom]["name_ar"] if dom != "-" else "-", الحالة=state))
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.download_button("⬇️ تنزيل جدول التوقع CSV", pd.DataFrame(rows).to_csv(index=False).encode("utf-8-sig"),
                       f"scorpio_outlook_{plat}_{plon}.csv", "text/csv")

# ------------------------------------------------------------------ TAB 5 – docs
with tab_doc:
    st.markdown(r"""
### المنهجية العلمية لـ SCORPIO
**1. البيانات:** الإشعاع طويل الموجة الصادر (OLR) اليومي على شبكة عالمية (≈1.9°) من NOAA PSL عبر بروتوكول OpenDAP،
مقتصرة على النطاق 30°S–30°N. الشذوذ = القيمة − المتوسط − الاتجاه الخطي لكل نقطة شبكية.

**2. الترشيح الطيفي الزمكاني (Wheeler & Kiladis 1999):** تحويل فورييه ثنائي الأبعاد (الزمن × خط الطول) لكل دائرة عرض،
ثم الإبقاء فقط على منطقة (عدد موجي k، تردد ω) لكل نوع موجة:

| الموجة | k | الدورة (يوم) | العمق المكافئ h | الاتجاه |
|---|---|---|---|---|
| MJO | 1–5 شرقي | 30–96 | – | شرق |
| Kelvin | 1–14 شرقي | 2.5–20 | 8–90 م | شرق |
| ER (روسبي n=1) | 1–10 غربي | 9.7–48 | 8–90 م | غرب |
| BSISO | ±6 | 25–90 | – | شمال/شرق |
| MRG | 1–10 غربي | 3–10 | 8–90 م | غرب |
| TD-type | 6–20 غربي | 2.5–5 | – | غرب |

منحنيات التشتت من نظرية Matsuno (1966) بمعامل β = 2.28×10⁻¹¹ m⁻¹s⁻¹.

**3. التوقع (Wheeler & Weickmann 2001):** يُحشى السجل بأصفار بعد آخر رصد قبل التحويل. لأن الأنماط المحتفظ بها ضيقة الطيف
ومنتشرة، فإن الإشارة المرشَّحة داخل نافذة الحشو تمثل امتداداً إحصائياً للموجة. المهارة المتوقعة:
MJO/BSISO ≈ 15–20 يوماً، كلفن/روسبي ≈ 5–10 أيام، MRG/TD ≈ 2–4 أيام. **الأيام الأخيرة من التوقع أقل موثوقية دائماً.**

**4. المؤشرات المرجعية:** OMI (Kiladis et al. 2014) من NOAA PSL بتحويل إلى إحداثيات RMM (RMM1 ≈ −PC2، RMM2 ≈ PC1)،
ومؤشر BSISO ثنائي النمط (Kikuchi et al. 2012) من IPRC للمقارنة التاريخية، مع مؤشر تقريبي آني يحسبه التطبيق من الحقل المرشَّح.

**5. حدود الاستخدام:** هذا نموذج إحصائي، وليس نموذجاً عددياً ديناميكياً؛ يُستخدم للتوقع دون الموسمي (أسابيع 1–3)
ولتحديد نوافذ الأمطار المرجحة على اليمن وشبه الجزيرة العربية عند تزامن الطور النشط لـ MJO مع موجة كلفن أو روسبي.

**المراجع:** Wheeler & Kiladis (1999) J. Atmos. Sci.; Wheeler & Weickmann (2001) Mon. Wea. Rev.; Wheeler & Hendon (2004);
Kiladis et al. (2009) Rev. Geophys.; Kiladis et al. (2014); Kikuchi, Wang & Kajikawa (2012) Clim. Dyn.
""")
