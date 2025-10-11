#!/bin/bash
# Check curriculum coverage by CEFR level

echo "📊 Curriculum Coverage Report"
echo "================================"
echo ""

# Check current database
if [ ! -f "out/learning/curriculum.sqlite" ]; then
    echo "❌ Database not found: out/learning/curriculum.sqlite"
    exit 1
fi

echo "Current Lessons by CEFR Level:"
echo "--------------------------------"

sqlite3 out/learning/curriculum.sqlite <<EOF
SELECT
  CASE
    WHEN name LIKE 'a1_%' OR name LIKE 'offline_a1_%' THEN 'A1 (Beginner)'
    WHEN name LIKE 'a2_%' OR name LIKE 'offline_a2_%' THEN 'A2 (Elementary)'
    WHEN name LIKE 'b1_%' OR name LIKE 'offline_b1_%' THEN 'B1 (Intermediate)'
    WHEN name LIKE 'b2_%' OR name LIKE 'offline_b2_%' THEN 'B2 (Upper Intermediate)'
    WHEN name LIKE 'c1_%' OR name LIKE 'offline_c1_%' OR
         name LIKE '%refactoring%' OR name LIKE '%performance%' OR
         name LIKE '%security%' THEN 'C1 (Advanced)'
    WHEN name LIKE 'c2_%' OR name LIKE 'offline_c2_%' THEN 'C2 (Expert)'
    ELSE 'Other/Uncategorized'
  END as level,
  COUNT(*) as count
FROM lessons
GROUP BY level
ORDER BY
  CASE level
    WHEN 'A1 (Beginner)' THEN 1
    WHEN 'A2 (Elementary)' THEN 2
    WHEN 'B1 (Intermediate)' THEN 3
    WHEN 'B2 (Upper Intermediate)' THEN 4
    WHEN 'C1 (Advanced)' THEN 5
    WHEN 'C2 (Expert)' THEN 6
    ELSE 7
  END;
EOF

echo ""
echo "Total Lessons:"
sqlite3 out/learning/curriculum.sqlite "SELECT COUNT(*) FROM lessons;"

echo ""
echo "================================"
echo ""
echo "Available CEFR Categories:"
echo "  a1  - Beginner (16 types)"
echo "  a2  - Elementary (16 types)"
echo "  b1  - Intermediate (16 types)"
echo "  b2  - Upper Intermediate (16 types)"
echo "  c1  - Advanced (22 types)"
echo "  c2  - Expert (40+ types)"
echo ""
echo "To generate specific category:"
echo "  ./generate_batch_lessons.sh 100 a1    # 100 beginner lessons"
echo "  ./generate_batch_lessons.sh 200 b2    # 200 security lessons"
echo ""
