
## أسهل طريقة للتشغيل (بدون خوادم)
1. انسخ مجلد `scorpio_app` كاملاً إلى جهازك (يلزم فقط تثبيت Python 3.10+).
2. انقر نقراً مزدوجاً على **`SCORPIO_start.bat`** (ويندوز) أو **`SCORPIO_start.command`** (ماك) — أو شغّل `python SCORPIO_start.py`.
3. في المرة الأولى يثبّت المكتبات، يحمّل بيانات ECMWF، يفتح `SCORPIO_dashboard.html` في المتصفح،
   ويسجّل نفسه تلقائياً ليعمل كل يوم الساعة 07:30 (Task Scheduler في ويندوز / cron في ماك ولينكس).
   بعدها تتحدث الخرائط يومياً من تلقاء نفسها — افتح فقط ملف `SCORPIO_dashboard.html`.

# 🦂 SCORPIO – Tropical Wave Forecast Maps (ECMWF)
**Designed by Ahmed Omar Dhafer**

Space-time Convectively-coupled wave Outlook & Real-time Propagation Index Observer.
MJO / BSISO / Kelvin / Equatorial Rossby forecasts from ECMWF IFS Open Data OLR
(Wheeler–Kiladis filtering + Wheeler–Weickmann statistical extension), with ECMWF
850 hPa wind (arrows + U850 Hovmöller, dynamical forecast to +15 d).

## "Scorpion" advanced tool-kit (`scorpio/advanced.py`)
- **VP200 & 200 hPa wind** – velocity potential solved from ECMWF divergence (upper-level MJO/Kelvin outflow), analysis + forecast to +15 d
- **Kelvin-mode Hough projection** of (u, Φ) at 850 hPa – local Kelvin identification without pre-filtering
- **Wheeler–Kiladis spectrum** of the current record with dispersion curves – which waves are truly active
- **Hindcast skill** – re-forecasts from 10 earlier dates → useful lead per wave (global & Arabian region)

## Quick start
```bash
pip install -r requirements.txt
python update_data.py 240      # backfill OLR record (once; then daily via cron)
python build_report.py         # -> SCORPIO_dashboard.html (self-contained)
streamlit run app.py           # interactive app
```
## Layout
- `app.py` – Streamlit app (map, Hovmöller OLR+U850, phases, local point, methods)
- `build_report.py` – static HTML dashboard with all figures
- `update_data.py` – auto-updater (cron-friendly)
- `scorpio/ecmwf.py` – OLR from ECMWF `ttr` (byte-range GRIB, 1° regrid, cache)
- `scorpio/ecmwf_wind.py` – 850 hPa u/v analysis + forecast
- `scorpio/waves.py` – dispersion curves, spectral masks, filter/forecast, MJO & BSISO indices
- `scorpio/data.py` – legacy NOAA loaders (reanalysis, OMI, BSISO reference)
