#!/bin/bash
# Set up automated backup cron jobs

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "🕐 Setting up automated backup cron jobs"
echo "========================================"

# Check if cron is available
if ! command -v crontab &> /dev/null; then
    echo "❌ crontab not available on this system"
    exit 1
fi

# Create cron job entries
CRON_ENTRIES="
# Nerion Curriculum Database Protection
# Generated: $(date)

# Hourly backup (every hour)
0 * * * * cd $PROJECT_DIR && ./scripts/backup_curriculum.sh >> out/learning/backup.log 2>&1

# Database monitoring (every 30 minutes)
*/30 * * * * cd $PROJECT_DIR && ./scripts/monitor_curriculum.sh >> out/learning/monitor.log 2>&1

# Daily full backup to GCS (at 3 AM)
0 3 * * * cd $PROJECT_DIR && ./scripts/backup_curriculum.sh && gcloud storage cp out/learning/curriculum.sqlite gs://nerion-training-data/curriculum/daily_backup_\$(date +\%Y\%m\%d).sqlite 2>&1 | tee -a out/learning/gcs_sync.log
"

# Show proposed cron jobs
echo "Proposed cron jobs:"
echo "$CRON_ENTRIES"
echo ""

read -p "Install these cron jobs? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "❌ Cancelled"
    exit 0
fi

# Backup existing crontab
crontab -l > /tmp/crontab_backup_$(date +%s).txt 2>/dev/null || true

# Add new entries (avoiding duplicates)
(crontab -l 2>/dev/null | grep -v "Nerion Curriculum Database Protection" || true; echo "$CRON_ENTRIES") | crontab -

echo "✅ Cron jobs installed!"
echo ""
echo "Verify with: crontab -l"
echo "Logs will be in:"
echo "  - out/learning/backup.log"
echo "  - out/learning/monitor.log"
echo "  - out/learning/gcs_sync.log"
