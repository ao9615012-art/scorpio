"""
SCORPIO – automatic updater / scheduler
=======================================
Runs forever:  every N hours (default 24 = once a day; ECMWF 00 UTC cycle is complete by ~07 UTC, aligned to ECMWF cycle publication ~ 00/06/12/18 UTC + 7 h)
  1. download any missing ECMWF OLR days (last 20 d) + slow backfill of older days
  2. fetch newest 850 hPa wind cycle
  3. rebuild SCORPIO_dashboard.html  (atomic replace → the file is never half-written)
  4. write data/status.json (last run, next run, data range, errors)

Usage:
    python scheduler.py                 # loop forever, once a day
    python scheduler.py --interval 3    # every 3 h
    python scheduler.py --once          # single update then exit (for cron / GitHub Actions)
"""
import os, sys, time, json, argparse, datetime as dt, subprocess, traceback, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STATUS = os.path.join(HERE, "data", "status.json")
LOG = os.path.join(HERE, "data", "scheduler.log")


def log(msg):
    line = f"[{dt.datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(**kw):
    st = {}
    if os.path.exists(STATUS):
        try:
            st = json.load(open(STATUS))
        except Exception:
            st = {}
    st.update(kw)
    json.dump(st, open(STATUS, "w"), indent=2, default=str, ensure_ascii=False)


def run_once(backfill_days=240):
    from scorpio import ecmwf
    t0 = time.time()
    today = dt.date.today()
    errors = []

    # 1. recent OLR days
    try:
        new = ecmwf.update(today - dt.timedelta(days=20), today - dt.timedelta(days=1), log=log, workers=2)
        log(f"OLR recent: +{len(new)} new days")
    except Exception as e:
        errors.append(f"olr-recent: {e}"); log(f"OLR recent failed: {e}")

    # 2. rebuild dashboard (subprocess → isolated, atomic write)
    out = os.path.join(HERE, "SCORPIO_dashboard.html")
    tmp = out + ".tmp"
    try:
        env = dict(os.environ, SCORPIO_OUT=tmp)
        r = subprocess.run([sys.executable, "-u", os.path.join(HERE, "build_report.py")],
                           capture_output=True, text=True, timeout=3600, env=env)
        tail = "\n".join(l for l in r.stdout.splitlines() if "Warning" not in l)[-800:]
        if r.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, out)
            log("dashboard rebuilt ✓")
        else:
            errors.append("report: " + (r.stderr[-500:] or tail))
            log(f"report failed: {r.stderr[-300:]}")
    except Exception as e:
        errors.append(f"report: {e}"); log(f"report exception: {e}")

    # 3. slow backfill (max 15 older days per run, so a run never takes too long)
    try:
        have = set(ecmwf.available_days())
        start = today - dt.timedelta(days=backfill_days)
        missing = [start + dt.timedelta(i) for i in range(backfill_days) if start + dt.timedelta(i) not in have]
        missing = [d for d in missing if d < today - dt.timedelta(days=20)]
        if missing:
            chunk = sorted(missing)[::-1]        # newest missing first
            log(f"OLR backfill: {len(missing)} missing – time-boxed 4 min")
            tb = time.time(); n = 0
            for d in chunk:
                if time.time() - tb > 240:
                    break
                if ecmwf.fetch_day(d) is not None:
                    n += 1
            log(f"OLR backfill: +{n} days")
    except Exception as e:
        errors.append(f"olr-backfill: {e}"); log(f"backfill failed: {e}")

    days = ecmwf.available_days()
    write_status(last_run=dt.datetime.utcnow(), duration_s=round(time.time() - t0),
                 olr_days=len(days), olr_first=days[0] if days else None, olr_last=days[-1] if days else None,
                 errors=errors, ok=not errors)
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=24, help="hours between runs (default 24 = once a day)")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    while True:
        log(f"=== update start (interval {a.interval} h) ===")
        try:
            errs = run_once()
            log("=== done" + (f" with {len(errs)} error(s)" if errs else " OK") + " ===")
        except Exception:
            log("FATAL:\n" + traceback.format_exc())
        if a.once:
            break
        nxt = dt.datetime.utcnow() + dt.timedelta(hours=a.interval)
        write_status(next_run=nxt)
        log(f"sleeping until {nxt:%Y-%m-%d %H:%M} UTC")
        time.sleep(a.interval * 3600)


if __name__ == "__main__":
    main()
