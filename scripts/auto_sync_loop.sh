#!/bin/bash
# Background loop to sync progress to bucket every 15 minutes
#
# Usage:
#   bash scripts/auto_sync_loop.sh &

set -euo pipefail
cd "$(dirname "$0")/.."

INTERVAL=900 # 15 minutes

echo "[auto-sync] Starting background sync loop (every 15m)"

while true; do
    # Run the sync script
    bash scripts/sync_to_bucket.sh > /tmp/last_sync.log 2>&1 || echo "[auto-sync] Sync failed, retrying in next cycle"
    
    echo "[auto-sync] Last sync finished at $(date)"
    sleep $INTERVAL
done
