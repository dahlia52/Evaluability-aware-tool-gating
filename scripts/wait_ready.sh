#!/usr/bin/env bash
# Usage: scripts/wait_ready.sh <port> [logfile]
P=$1; L=${2:-}
for i in $(seq 1 180); do
  c=$(curl -s -o /dev/null -w %{http_code} -m 3 "http://localhost:$P/v1/models")
  [ "$c" = "200" ] && { echo "port $P READY"; exit 0; }
  if [ -n "$L" ] && grep -qE "Engine core initialization failed|RuntimeError" "$L" 2>/dev/null; then
    echo "port $P FAILED"; grep -E "Error|RuntimeError" "$L" | tail -3; exit 1
  fi
  sleep 10
done
echo "port $P TIMEOUT"; exit 1
