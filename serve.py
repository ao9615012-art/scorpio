"""SCORPIO web server: serves the latest dashboard at / and refreshes data once a day in the background."""
import os, sys, threading, time, subprocess, datetime as dt
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
PORT = int(os.environ.get("PORT", 8080))


class H(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.path = "/SCORPIO_dashboard.html"
        return super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *a):
        pass


def updater():
    while True:
        print(f"[{dt.datetime.utcnow():%F %T} UTC] daily update start", flush=True)
        subprocess.run([sys.executable, "scheduler.py", "--once"])
        print(f"[{dt.datetime.utcnow():%F %T} UTC] update done – next in 24 h", flush=True)
        time.sleep(24 * 3600)


if __name__ == "__main__":
    if "--no-update" not in sys.argv:
        threading.Thread(target=updater, daemon=True).start()
    print(f"SCORPIO serving on 0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
