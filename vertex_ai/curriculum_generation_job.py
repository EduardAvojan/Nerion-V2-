#!/usr/bin/env python3
"""
Vertex AI training job for massive curriculum generation.

This runs ON Vertex AI infrastructure, not your local machine.
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
        print(f"   - Storage client initialized")
        bucket = storage_client.bucket(bucket_name)
        print(f"   - Bucket reference created: {bucket_name}")
        blob = bucket.blob(blob_name)
        print(f"   - Blob reference created: {blob_name}")

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
    parser.add_argument("--provider", type=str, default="vertexai:gemini-2.5-pro")
    parser.add_argument("--project-id", type=str, default=None, help="GCP Project ID")
    parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI location")
    parser.add_argument("--category", type=str, default=None, help="Lesson category to focus on")
    args = parser.parse_args()

    # Set environment variables for Vertex AI
    if args.project_id:
        os.environ["NERION_V2_VERTEX_PROJECT_ID"] = args.project_id
    os.environ["NERION_V2_VERTEX_LOCATION"] = args.location

    # Increase timeout for Vertex AI calls (curriculum generation can take 10+ minutes per lesson)
    os.environ["NERION_V2_REQUEST_TIMEOUT"] = "900"  # 15 minutes

    print(f"🚀 Vertex AI Curriculum Generation Job")
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

    for i in range(args.cycles):
        print(f"\n{'='*60}")
        print(f"CYCLE {i+1}/{args.cycles}")
        print(f"{'='*60}")

        # Set a 20-minute timeout per cycle to prevent getting stuck
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(1200)  # 20 minutes in seconds

        try:
            orchestrator.run_cycle(provider=args.provider)
            successful += 1
            signal.alarm(0)  # Cancel the alarm
        except TimeoutError:
            print(f"⏰ TIMEOUT in cycle {i+1}: Cycle took longer than 20 minutes, skipping...")
            failed += 1
            signal.alarm(0)  # Cancel the alarm
        except Exception as e:
            print(f"ERROR in cycle {i+1}: {e}")
            failed += 1
            signal.alarm(0)  # Cancel the alarm

        # Progress update
        print(f"\nProgress: {i+1}/{args.cycles} | Success: {successful} | Failed: {failed}")

        # Upload checkpoint every 10 cycles to avoid losing progress
        if (i + 1) % 10 == 0:
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
