"""Auto-updater (ECMWF-only): python update_data.py [ndays]"""
import sys, os, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scorpio import ecmwf
days = int(sys.argv[1]) if len(sys.argv) > 1 else 240
new = ecmwf.update(dt.date.today() - dt.timedelta(days=days), log=print)
print("added", len(new), "days; latest:", ecmwf.available_days()[-1] if ecmwf.available_days() else None)
