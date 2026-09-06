#!/bin/bash
# SCORPIO – bring the public link up: server + daily updater + Cloudflare tunnel. Writes the URL to data/public_url.txt
cd "$(dirname "$0")"
mkdir -p data
[ -x /tmp/cloudflared ] || { curl -sL -m 120 -o /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /tmp/cloudflared; }
python3 -c "import cfgrib" 2>/dev/null || pip install -q cfgrib eccodes >/dev/null 2>&1
python3 -u serve.py > data/serve.log 2>&1 &
SERVER=$!
sleep 2
/tmp/cloudflared tunnel --url http://localhost:8080 --no-autoupdate 2>&1 | tee data/tunnel.log | while read -r line; do
  echo "$line"
  u=$(echo "$line" | grep -o 'https://[a-z0-9-]*\.trycloudflare\.com')
  [ -n "$u" ] && echo "$u" > data/public_url.txt && echo "PUBLIC_URL=$u"
done
kill $SERVER 2>/dev/null
