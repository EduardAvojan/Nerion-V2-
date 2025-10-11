#!/bin/bash
# Quick test script for GitHub quality scraper

echo "🧪 Testing GitHub Quality Scraper"
echo "=================================="
echo ""

# Check if GitHub token is set
if [ -z "$GITHUB_TOKEN" ]; then
    echo "⚠️  GITHUB_TOKEN not set"
    echo "   Using unauthenticated access (60 req/hour)"
    echo "   For better rate limits, set GITHUB_TOKEN:"
    echo "   export GITHUB_TOKEN=ghp_your_token_here"
    echo ""
else
    echo "✅ GITHUB_TOKEN found"
    echo "   Using authenticated access (5000 req/hour)"
    echo ""
fi

# Test 1: API Connector Demo
echo "Test 1: GitHub API Connector"
echo "-----------------------------"
python -m nerion_digital_physicist.data_mining.github_api_connector
echo ""

# Test 2: Small scrape run
echo "Test 2: Quality Scraper (10 commits)"
echo "-------------------------------------"
python -m nerion_digital_physicist.data_mining.run_scraper \
    --target 10 \
    --db test_scraper.db \
    --test \
    --min-quality 50

echo ""
echo "=================================="
echo "✅ Test Complete"
echo ""

# Show results
if [ -f test_scraper.db ]; then
    echo "📊 Results:"
    echo ""
    sqlite3 test_scraper.db "SELECT COUNT(*) as total FROM lessons" 2>/dev/null && \
    sqlite3 test_scraper.db "SELECT category, COUNT(*) as count FROM lessons GROUP BY category ORDER BY category" 2>/dev/null && \
    echo "" && \
    echo "Sample lessons:" && \
    sqlite3 test_scraper.db "SELECT name, category, json_extract(metadata, '$.quality_score') as score FROM lessons LIMIT 5" 2>/dev/null
else
    echo "⚠️  No database created - check for errors above"
fi

echo ""
echo "To run full scrape:"
echo "  python -m nerion_digital_physicist.data_mining.run_scraper --target 1000 --db github_lessons.db"
