"""CLI script to run the GitHub quality scraper.

Usage:
    python -m nerion_digital_physicist.data_mining.run_scraper --target 100 --test

    # Full production run
    python -m nerion_digital_physicist.data_mining.run_scraper --target 10000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    # Load .env from project root
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # dotenv not installed, try manual loading
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())

from nerion_digital_physicist.data_mining.github_api_connector import (
    GitHubAPIConnector,
    build_search_queries,
)
from nerion_digital_physicist.data_mining.github_quality_scraper import (
    GitHubQualityScraper,
    QualityThresholds,
)


def integrate_scraper_with_api(
    scraper: GitHubQualityScraper,
    connector: GitHubAPIConnector,
    target_count: int,
    test_mode: bool = False
):
    """Integrate the scraper with GitHub API connector.

    Args:
        scraper: Quality scraper instance
        connector: GitHub API connector
        target_count: Target number of quality commits to collect
        test_mode: If True, only process first query with limited results
    """
    queries = build_search_queries()

    if test_mode:
        print("🧪 TEST MODE: Using limited queries and results")
        queries = queries[:1]  # Only first query
        max_per_query = 50
    else:
        max_per_query = 1000  # GitHub API limit per query

    total_processed = 0
    total_accepted = 0

    for query_idx, query in enumerate(queries, 1):
        if scraper.stats.accepted >= target_count:
            print(f"\n✅ Target reached! Collected {scraper.stats.accepted} quality commits")
            break

        print(f"\n{'='*60}")
        print(f"Query {query_idx}/{len(queries)}: {query}")
        print(f"{'='*60}\n")

        commit_count = 0

        for commit_json in connector.search_commits(query, max_results=max_per_query):
            commit_count += 1
            scraper.stats.fetched += 1
            total_processed += 1

            # Progress update every 10 commits
            if commit_count % 10 == 0:
                print(f"  Processed {commit_count} commits from this query...")

            try:
                # Extract commit metadata
                commit_data = connector.extract_commit_data(commit_json)
                if not commit_data:
                    continue

                # Stage 1: Message filter
                if not scraper.passes_message_filter(commit_data):
                    scraper.stats.filtered_message += 1
                    continue

                # Stage 2: File filter
                if not scraper.passes_file_filter(commit_data):
                    scraper.stats.filtered_file_type += 1
                    continue

                # Fetch full commit details with patches
                repo = commit_data.repo
                sha = commit_data.sha

                full_commit = connector.fetch_commit_diff(repo, sha)
                if not full_commit:
                    continue

                # Update files list from full commit (search results don't include files)
                commit_data.files = [f["filename"] for f in full_commit.get("files", [])]

                # Extract code changes for Python files
                for file_path in commit_data.files:
                    if not file_path.endswith('.py'):
                        continue

                    changes = connector.extract_file_changes(full_commit, file_path, repo)
                    if not changes:
                        continue

                    before_code, after_code = changes
                    commit_data.before_code = before_code
                    commit_data.after_code = after_code

                    # Stage 3: Size filter
                    if not scraper.passes_size_filter(before_code, after_code):
                        scraper.stats.filtered_size += 1
                        continue

                    # Stage 4: Syntax validation
                    if not scraper.validate_syntax(before_code, after_code):
                        scraper.stats.filtered_syntax += 1
                        continue

                    # Stage 5: Quality assessment
                    if not scraper.assess_quality(commit_data):
                        scraper.stats.filtered_quality += 1
                        continue

                    # Stage 6: Infer category
                    commit_data.category = scraper.infer_category(commit_data)

                    # Stage 7: Synthesize test code
                    test_code = scraper.synthesize_test_code(commit_data)

                    # Stage 8: Save to database
                    scraper.save_lesson(commit_data, test_code)
                    total_accepted += 1

                    print(f"  ✅ Accepted: {commit_data.sha[:8]} ({commit_data.category}) - Score: {commit_data.quality_score}")

                    # Show progress every 10 accepted
                    if total_accepted % 10 == 0:
                        scraper.stats.print_progress()

                    # Only process one file per commit to avoid duplicates
                    break

                # Check if target reached
                if scraper.stats.accepted >= target_count:
                    break

            except Exception as e:
                print(f"  ⚠️  Error processing commit: {e}")
                scraper.stats.errors += 1
                continue

        print(f"\n  Query complete: {commit_count} commits processed")

    # Final statistics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    scraper.stats.print_progress()

    print(f"\nDatabase: {scraper.db_path}")
    print(f"Total commits processed: {total_processed}")
    print(f"Quality commits saved: {scraper.stats.accepted}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GitHub Quality Scraper - Mine high-quality Python bug fixes for GNN training"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1000,
        help="Target number of quality commits to collect (default: 1000)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("github_lessons.db"),
        help="Database path for storing lessons (default: github_lessons.db)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode (limited queries and results)"
    )
    parser.add_argument(
        "--min-quality",
        type=int,
        default=60,
        help="Minimum quality score threshold (0-100, default: 60)"
    )
    parser.add_argument(
        "--github-token",
        type=str,
        default=None,
        help="GitHub personal access token (optional, increases rate limit)"
    )

    args = parser.parse_args()

    # Check for GitHub token in environment if not provided
    github_token = args.github_token or os.getenv("GITHUB_TOKEN")

    if not github_token:
        print("⚠️  Warning: No GitHub token provided")
        print("   Rate limit: 60 requests/hour (unauthenticated)")
        print("   To increase to 5000/hour, set GITHUB_TOKEN environment variable")
        print("   Get a token at: https://github.com/settings/tokens")
        print()

        response = input("Continue without token? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Set GITHUB_TOKEN and try again.")
            sys.exit(0)
    else:
        print("✅ Using authenticated GitHub access (5000 req/hour)")

    # Setup quality thresholds
    thresholds = QualityThresholds(min_quality_score=args.min_quality)

    # Initialize scraper and connector
    print(f"\n🚀 Initializing GitHub Quality Scraper")
    print(f"   Target: {args.target} quality commits")
    print(f"   Database: {args.db}")
    print(f"   Quality threshold: {args.min_quality}/100")
    print(f"   Test mode: {args.test}")
    print()

    scraper = GitHubQualityScraper(
        db_path=args.db,
        github_token=github_token,
        thresholds=thresholds
    )

    connector = GitHubAPIConnector(token=github_token)

    # Check rate limit before starting
    rate = connector.check_rate_limit()
    if rate:
        print(f"📊 Rate limit: {rate.get('remaining', 'unknown')}/{rate.get('limit', 'unknown')} remaining")
        reset_time = rate.get('reset', 0)
        if reset_time:
            from datetime import datetime
            reset_dt = datetime.fromtimestamp(reset_time)
            print(f"   Resets at: {reset_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Run scraper
    try:
        integrate_scraper_with_api(
            scraper=scraper,
            connector=connector,
            target_count=args.target,
            test_mode=args.test
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        print("\nPartial results saved to database")
        scraper.stats.print_progress()
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ Scraping complete!")
    print(f"   Quality lessons saved: {scraper.stats.accepted}")
    print(f"   Database: {args.db}")


if __name__ == "__main__":
    main()
