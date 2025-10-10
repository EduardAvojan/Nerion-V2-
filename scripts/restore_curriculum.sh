#!/bin/bash
# Curriculum Database Restoration Script

set -e

BACKUP_DIR="backups/curriculum"
DB_PATH="out/learning/curriculum.sqlite"

echo "🔄 Curriculum Database Restoration"
echo "=================================="

# List available backups
echo ""
echo "Available backups:"
echo "1. Local daily backups"
find "$BACKUP_DIR/daily" -name "*.sqlite" -exec ls -lh {} \; 2>/dev/null | tail -10 | nl
echo ""
echo "2. Before-write backups (most recent)"
find "$BACKUP_DIR/before_write" -name "*.sqlite" -exec ls -lh {} \; 2>/dev/null | tail -5 | nl
echo ""
echo "3. GCS backups"
gcloud storage ls gs://nerion-training-data/curriculum/*.sqlite 2>/dev/null | tail -10 | nl || echo "   (GCS not available)"

echo ""
echo "Please specify backup to restore:"
echo "  - Path to local backup file"
echo "  - 'latest' for most recent daily backup"
echo "  - 'gcs:latest' for most recent GCS backup"
echo ""

read -p "Backup to restore: " BACKUP_CHOICE

if [ "$BACKUP_CHOICE" = "latest" ]; then
    BACKUP_FILE=$(find "$BACKUP_DIR/daily" -name "*.sqlite" | sort -r | head -1)
elif [ "$BACKUP_CHOICE" = "gcs:latest" ]; then
    GCS_FILE=$(gcloud storage ls gs://nerion-training-data/curriculum/*.sqlite | sort -r | head -1)
    BACKUP_FILE="/tmp/curriculum_restore_$(date +%s).sqlite"
    gcloud storage cp "$GCS_FILE" "$BACKUP_FILE"
else
    BACKUP_FILE="$BACKUP_CHOICE"
fi

if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Verify backup
echo "🔍 Verifying backup integrity..."
if sqlite3 "$BACKUP_FILE" "PRAGMA integrity_check;" | grep -q "ok"; then
    LESSON_COUNT=$(sqlite3 "$BACKUP_FILE" "SELECT COUNT(*) FROM lessons;")
    echo "✅ Backup verified: $LESSON_COUNT lessons"
else
    echo "❌ Backup integrity check FAILED"
    exit 1
fi

# Create backup of current DB before restoring
if [ -f "$DB_PATH" ]; then
    CURRENT_BACKUP="${DB_PATH}.before_restore_$(date +%Y%m%d_%H%M%S)"
    cp "$DB_PATH" "$CURRENT_BACKUP"
    echo "💾 Current DB backed up to: $CURRENT_BACKUP"
fi

# Restore
echo "🔄 Restoring database..."
cp "$BACKUP_FILE" "$DB_PATH"

# Verify restoration
NEW_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM lessons;")
echo ""
echo "✅ Restoration complete!"
echo "   Lessons restored: $NEW_COUNT"
echo "   Source: $BACKUP_FILE"

# Enable WAL mode
sqlite3 "$DB_PATH" "PRAGMA journal_mode=WAL;"
echo "✅ WAL mode re-enabled"
