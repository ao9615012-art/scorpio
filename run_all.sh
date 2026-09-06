#!/usr/bin/env bash
# Start SCORPIO: background auto-updater (once a day) + Streamlit app
cd "$(dirname "$0")"
pip install -q -r requirements.txt
nohup python3 -u scheduler.py --interval 6 >> data/scheduler.out 2>&1 &
echo "scheduler pid $!"
streamlit run app.py
