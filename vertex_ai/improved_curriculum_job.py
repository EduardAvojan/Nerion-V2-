#!/usr/bin/env python3
"""
IMPROVED Vertex AI training job with better error handling and timeout management.

Key improvements:
1. Increased timeout per cycle (30 minutes instead of 20)
2. Exponential backoff on failures
3. Checkpoint every 5 cycles (not 10)
4. Continue on timeout instead of failing
5. Better error reporting
"""
import argparse
import os
import sys
import signal
import time
from pathlib import Path
from google.cloud import storage
from nerion_digital_physicist.learning_orchestrator import LearningOrchestrator


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Cycle timed out")


def upload_to_gcs(local_path: str, bucket_name: str, blob_name: str):
    """Upload file to Google Cloud Storage with detailed error handling."""
    try:
        print(f"📤 Attempting upload: {local_path} -> gs://{bucket_name}/{blob_name}")
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        # Check if file exists and get size
        file_size = Path(local_path).stat().st_size
        print(f"   - File size: {file_size / 1024:.2f} KB")

        blob.upload_from_filename(local_path)
        print(f"✅ UPLOADED: {local_path} to gs://{bucket_name}/{blob_name}")
        return True
    except FileNotFoundError as e:
        print(f"❌ UPLOAD FAILED: File not found: {local_path}")
        print(f"   Error: {e}")
        return False
    except Exception as e:
        print(f"❌ UPLOAD FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=100)
    parser.add_argument("--bucket", type=str, required=True, help="GCS bucket for output")
    parser.add_argument("--provider", type=str, default="vertexai:gemini-2.0-flash-exp")  # Faster, more reliable
    parser.add_argument("--project-id", type=str, default=None, help="GCP Project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI location")
    parser.add_argument("--category", type=str, default=None, help="Lesson category to focus on")
    args = parser.parse_args()

    # Set environment variables for Vertex AI
    if args.project_id:
        os.environ["NERION_V2_VERTEX_PROJECT_ID"] = args.project_id
    os.environ["NERION_V2_VERTEX_LOCATION"] = args.location

    # Increase timeout for Vertex AI calls (more generous)
    os.environ["NERION_V2_REQUEST_TIMEOUT"] = "1200"  # 20 minutes per request

    print(f"🚀 IMPROVED Vertex AI Curriculum Generation Job")
    print(f"   Cycles: {args.cycles}")
    print(f"   Bucket: {args.bucket}")
    print(f"   Provider: {args.provider}")
    print(f"   Category: {args.category or 'ALL (random)'}")
    print(f"   Project: {os.getenv('NERION_V2_VERTEX_PROJECT_ID', 'from-default')}")
    print(f"   Location: {args.location}")
    print()

    orchestrator = LearningOrchestrator(category_filter=args.category)

    successful = 0
    failed = 0
    timeouts = 0
    consecutive_failures = 0

    for i in range(args.cycles):
        print(f"\n{'='*60}")
        print(f"CYCLE {i+1}/{args.cycles}")
        print(f"{'='*60}")

        # Set a 30-minute timeout per cycle (more generous than before)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1800)  # 30 minutes in seconds

        try:
            orchestrator.run_cycle(provider=args.provider)
            successful += 1
            consecutive_failures = 0  # Reset on success
            signal.alarm(0)  # Cancel the alarm
            print(f"✅ Success!")

        except TimeoutError:
            timeouts += 1
            consecutive_failures += 1
            print(f"⏰ TIMEOUT in cycle {i+1}: Cycle took longer than 30 minutes, skipping...")
            signal.alarm(0)  # Cancel the alarm

            # If too many consecutive failures, take a longer break
            if consecutive_failures >= 3:
                print(f"⚠️  3 consecutive failures, taking 60s break...")
                time.sleep(60)
                consecutive_failures = 0

        except Exception as e:
            failed += 1
            consecutive_failures += 1
            print(f"❌ ERROR in cycle {i+1}: {e}")
            signal.alarm(0)  # Cancel the alarm

            # Exponential backoff on failures
            if consecutive_failures >= 3:
                backoff = min(120, 10 * (2 ** (consecutive_failures - 3)))  # Cap at 120s
                print(f"⚠️  {consecutive_failures} consecutive failures, backing off {backoff}s...")
                time.sleep(backoff)
                consecutive_failures = 0

        # Progress update
        print(f"\nProgress: {i+1}/{args.cycles} | Success: {successful} | Failed: {failed} | Timeouts: {timeouts}")
        print(f"Success rate: {successful/(i+1)*100:.1f}%")

        # Upload checkpoint every 5 cycles (more frequent than before)
        if (i + 1) % 5 == 0:
            db_path = "out/learning/curriculum.sqlite"
            job_id = os.getenv('CLOUD_ML_JOB_ID') or os.getenv('JOB_ID') or f"manual_{int(time.time())}"
            print(f"\n{'='*60}")
            print(f"📦 CHECKPOINT at cycle {i+1}")
            print(f"   Job ID: {job_id}")
            print(f"{'='*60}")

            if Path(db_path).exists():
                blob_name = f"curriculum/curriculum_{job_id}_checkpoint_cycle{i+1}.sqlite"
                success = upload_to_gcs(db_path, args.bucket, blob_name)
                if success:
                    print(f"✅ Checkpoint uploaded successfully at cycle {i+1}")
                else:
                    print(f"⚠️  Checkpoint upload failed at cycle {i+1} - continuing anyway")
            else:
                print(f"⚠️  Database file not found at {db_path} - skipping checkpoint")

    print(f"\n{'='*60}")
    print(f"✅ Job Complete: {successful}/{args.cycles} successful")
    print(f"   Failed: {failed}")
    print(f"   Timeouts: {timeouts}")
    print(f"   Success rate: {successful/args.cycles*100:.1f}%")
    print(f"{'='*60}")

    # Upload final curriculum database to GCS
    db_path = "out/learning/curriculum.sqlite"
    job_id = os.getenv('CLOUD_ML_JOB_ID') or os.getenv('JOB_ID') or f"manual_{int(time.time())}"

    print(f"\n{'='*60}")
    print(f"📦 FINAL UPLOAD")
    print(f"   Job ID: {job_id}")
    print(f"   Database: {db_path}")
    print(f"{'='*60}")

    if Path(db_path).exists():
        blob_name = f"curriculum/curriculum_{job_id}_final.sqlite"
        success = upload_to_gcs(db_path, args.bucket, blob_name)
        if success:
            print(f"✅ Final curriculum uploaded to GCS successfully!")
            print(f"   Location: gs://{args.bucket}/{blob_name}")
        else:
            print(f"❌ CRITICAL: Final upload failed - lessons may be lost!")
            sys.exit(1)
    else:
        print(f"❌ CRITICAL: Database file not found at {db_path}!")
        sys.exit(1)


if __name__ == "__main__":
    main()
