"""
SCORPIO – one-click start (double-click this file, nothing else to configure)
=============================================================================
What it does, automatically:
  1. installs the few Python packages it needs (first time only)
  2. downloads today's ECMWF data and rebuilds SCORPIO_dashboard.html
  3. opens the dashboard in your browser
  4. registers itself to repeat every day at 07:30 (Windows Task Scheduler / cron on Mac & Linux)
     -> after the first double-click the maps update themselves every day, forever.
"""
import os, sys, subprocess, webbrowser, platform, getpass

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
os.chdir(HERE)


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, **kw)


def install_requirements():
    marker = os.path.join(HERE, "data", ".deps_ok")
    if os.path.exists(marker):
        return
    print("Installing packages (first run only) …")
    subprocess.run([PY, "-m", "pip", "install", "-q", "-r", os.path.join(HERE, "requirements.txt")])
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").write("ok")


def register_daily():
    """Make the OS run this script once a day (07:30 local) – silent if already registered."""
    system = platform.system()
    try:
        if system == "Windows":
            task = "SCORPIO_daily_update"
            if task in sh(["schtasks", "/Query"]).stdout:
                return "already registered (Task Scheduler)"
            pyw = PY.replace("python.exe", "pythonw.exe")  # no console window
            cmd = f'"{pyw if os.path.exists(pyw) else PY}" "{os.path.join(HERE, "SCORPIO_start.py")}" --quiet'
            r = sh(["schtasks", "/Create", "/F", "/SC", "DAILY", "/ST", "07:30", "/TN", task, "/TR", cmd])
            return "registered in Windows Task Scheduler" if r.returncode == 0 else f"could not register: {r.stderr.strip()}"
        else:  # macOS / Linux -> crontab
            line = f'30 7 * * * cd "{HERE}" && "{PY}" SCORPIO_start.py --quiet >> data/cron.log 2>&1'
            cur = sh("crontab -l 2>/dev/null").stdout
            if "SCORPIO_start.py" in cur:
                return "already registered (cron)"
            r = subprocess.run("crontab -", shell=True, input=cur.rstrip("\n") + "\n" + line + "\n", text=True, capture_output=True)
            return "registered in cron (daily 07:30)" if r.returncode == 0 else f"could not register: {r.stderr.strip()}"
    except Exception as e:
        return f"could not register: {e}"


def main():
    quiet = "--quiet" in sys.argv
    install_requirements()
    print("Updating ECMWF data and rebuilding the dashboard … (1–3 minutes)")
    r = subprocess.run([PY, os.path.join(HERE, "scheduler.py"), "--once"])
    out = os.path.join(HERE, "SCORPIO_dashboard.html")
    if r.returncode == 0 and os.path.exists(out):
        print("Dashboard updated ✓")
    else:
        print("Update failed – see data/scheduler.log (the previous dashboard is kept)")
    print("Daily auto-update:", register_daily())
    if not quiet:
        webbrowser.open("file://" + out)
        input("\nDone. Press Enter to close.")


if __name__ == "__main__":
    main()
