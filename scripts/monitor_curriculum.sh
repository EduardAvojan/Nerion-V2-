#!/bin/bash
# Curriculum Database Monitoring Script
# Checks database health and alerts on issues

DB_PATH="out/learning/curriculum.sqlite"
ALERT_FILE="out/learning/curriculum_alerts.log"
MIN_LESSONS=500  # Alert if lessons drop below this

check_database() {
    if [ ! -f "$DB_PATH" ]; then
        echo "[$(date)] ❌ CRITICAL: Database file missing!" | tee -a "$ALERT_FILE"
        return 1
    fi
    
    # Check integrity
    if ! sqlite3 "$DB_PATH" "PRAGMA integrity_check;" | grep -q "ok"; then
        echo "[$(date)] ❌ CRITICAL: Database integrity check failed!" | tee -a "$ALERT_FILE"
        return 1
    fi
    
    # Check lesson count
    LESSON_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM lessons;")
    if [ "$LESSON_COUNT" -lt "$MIN_LESSONS" ]; then
        echo "[$(date)] ⚠️  WARNING: Lesson count dropped to $LESSON_COUNT (threshold: $MIN_LESSONS)" | tee -a "$ALERT_FILE"
        return 1
    fi
    
    # Check WAL mode
    WAL_MODE=$(sqlite3 "$DB_PATH" "PRAGMA journal_mode;")
    if [ "$WAL_MODE" != "wal" ]; then
        echo "[$(date)] ⚠️  WARNING: WAL mode not enabled (current: $WAL_MODE)" | tee -a "$ALERT_FILE"
    fi
    
    echo "[$(date)] ✅ Database healthy: $LESSON_COUNT lessons, integrity OK"
    return 0
}

sync_to_gcs() {
    if command -v gcloud &> /dev/null; then
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        if gcloud storage cp "$DB_PATH" "gs://nerion-training-data/curriculum/auto_sync_${TIMESTAMP}.sqlite" 2>/dev/null; then
            echo "[$(date)] ☁️  Synced to GCS: auto_sync_${TIMESTAMP}.sqlite"
            
            # Keep only last 50 auto-syncs
            gcloud storage ls gs://nerion-training-data/curriculum/auto_sync_*.sqlite | sort -r | tail -n +51 | \
                xargs -I {} gcloud storage rm {} 2>/dev/null || true
        else
            echo "[$(date)] ⚠️  GCS sync failed" | tee -a "$ALERT_FILE"
        fi
    fi
}

# Run checks
check_database
if [ $? -eq 0 ]; then
    sync_to_gcs
fi
