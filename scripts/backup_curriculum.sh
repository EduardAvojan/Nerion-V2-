#!/bin/bash
# Curriculum Database Backup Script
# Protects against data loss with multiple backup strategies

set -e

DB_PATH="out/learning/curriculum.sqlite"
BACKUP_DIR="backups/curriculum"
GCS_BUCKET="nerion-training-data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🔒 Curriculum Database Backup System"
echo "===================================="

# 1. Create backup directory structure
mkdir -p "$BACKUP_DIR/daily"
mkdir -p "$BACKUP_DIR/hourly"
mkdir -p "$BACKUP_DIR/before_write"

# 2. Check if database exists and is valid
if [ ! -f "$DB_PATH" ]; then
    echo -e "${RED}❌ Database not found at $DB_PATH${NC}"
    exit 1
fi

# 3. Verify database integrity
echo "🔍 Checking database integrity..."
if sqlite3 "$DB_PATH" "PRAGMA integrity_check;" | grep -q "ok"; then
    echo -e "${GREEN}✅ Database integrity OK${NC}"
else
    echo -e "${RED}❌ Database integrity check FAILED${NC}"
    exit 1
fi

# 4. Get lesson count for verification
LESSON_COUNT=$(sqlite3 "$DB_PATH" "SELECT COUNT(*) FROM lessons;")
echo "📊 Current lesson count: $LESSON_COUNT"

# 5. Create timestamped backup
BACKUP_FILE="$BACKUP_DIR/daily/curriculum_${TIMESTAMP}.sqlite"
echo "💾 Creating backup: $BACKUP_FILE"
cp "$DB_PATH" "$BACKUP_FILE"

# 6. Verify backup
BACKUP_COUNT=$(sqlite3 "$BACKUP_FILE" "SELECT COUNT(*) FROM lessons;")
if [ "$BACKUP_COUNT" -eq "$LESSON_COUNT" ]; then
    echo -e "${GREEN}✅ Backup verified: $BACKUP_COUNT lessons${NC}"
else
    echo -e "${RED}❌ Backup verification FAILED${NC}"
    exit 1
fi

# 7. Create compressed archive
echo "📦 Creating compressed archive..."
gzip -c "$BACKUP_FILE" > "${BACKUP_FILE}.gz"

# 8. Upload to GCS (if available)
if command -v gcloud &> /dev/null; then
    echo "☁️  Uploading to GCS..."
    gcloud storage cp "$BACKUP_FILE" "gs://${GCS_BUCKET}/curriculum/backups/" 2>/dev/null || \
        echo -e "${YELLOW}⚠️  GCS upload failed or not configured${NC}"
else
    echo -e "${YELLOW}⚠️  gcloud not available, skipping GCS upload${NC}"
fi

# 9. Clean up old backups (keep last 30 days)
echo "🧹 Cleaning old backups (keeping last 30 days)..."
find "$BACKUP_DIR/daily" -name "*.sqlite" -mtime +30 -delete 2>/dev/null || true
find "$BACKUP_DIR/daily" -name "*.sqlite.gz" -mtime +30 -delete 2>/dev/null || true

# 10. Summary
echo ""
echo "✅ Backup complete!"
echo "   Location: $BACKUP_FILE"
echo "   Size: $(du -h "$BACKUP_FILE" | cut -f1)"
echo "   Lessons: $LESSON_COUNT"
echo "   Timestamp: $TIMESTAMP"
