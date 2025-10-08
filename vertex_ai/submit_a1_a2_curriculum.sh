#!/bin/bash
# Submit A1-A2 curriculum generation job to Vertex AI
# Generates 54 lessons: 26 A1 + 28 A2 using Gemini 2.5 Pro

set -e

PROJECT_ID="${NERION_V2_VERTEX_PROJECT_ID:-nerion-vertex-project}"
REGION="${NERION_V2_VERTEX_LOCATION:-us-central1}"
BUCKET_NAME="nerion-training-data"
IMAGE_URI="gcr.io/${PROJECT_ID}/nerion-trainer:latest"
JOB_NAME="a1-a2-curriculum-$(date +%Y%m%d-%H%M%S)"

echo "🎯 A1-A2 Curriculum Generation - Vertex AI"
echo "============================================"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Job Name: ${JOB_NAME}"
echo "   Target: 54 A1-A2 lessons (26 A1 + 28 A2)"
echo "   Model: Gemini 2.5 Pro"
echo "   Estimated Cost: ~$51.84 @ $0.96/lesson"
echo ""

# Build using Cloud Build (no local Docker needed!)
echo "📦 Building Docker image with Cloud Build..."
gcloud builds submit --config=cloudbuild.yaml --project=${PROJECT_ID} .

# Submit A1 lessons job (26 lessons)
echo "🚀 Submitting A1 Lessons Job (26 lessons)..."
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name="${JOB_NAME}-a1" \
  --worker-pool-spec=machine-type=n1-highmem-4,replica-count=1,container-image-uri=${IMAGE_URI} \
  --args="--cycles=26,--bucket=${BUCKET_NAME},--provider=vertexai:gemini-2.5-pro,--category=a1,--project-id=${PROJECT_ID},--location=${REGION}" \
  --project=${PROJECT_ID}

echo ""
echo "✅ A1 Job submitted!"

# Submit A2 lessons job (28 lessons)
echo "🚀 Submitting A2 Lessons Job (28 lessons)..."
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name="${JOB_NAME}-a2" \
  --worker-pool-spec=machine-type=n1-highmem-4,replica-count=1,container-image-uri=${IMAGE_URI} \
  --args="--cycles=28,--bucket=${BUCKET_NAME},--provider=vertexai:gemini-2.5-pro,--category=a2,--project-id=${PROJECT_ID},--location=${REGION}" \
  --project=${PROJECT_ID}

echo ""
echo "✅ A2 Job submitted!"
echo ""
echo "📊 Jobs Running:"
echo "   - A1 Lessons: 26 lessons (categories: variables, loops, conditionals, functions, etc.)"
echo "   - A2 Lessons: 28 lessons (categories: lists, dicts, files, strings, tuples, etc.)"
echo ""
echo "🔍 Monitor jobs at:"
echo "   https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "💰 Estimated completion time: ~3-5 hours"
echo "💰 Estimated cost: ~$51.84 (54 lessons @ $0.96/lesson)"
echo ""
echo "📥 Results will be uploaded to:"
echo "   gs://${BUCKET_NAME}/curriculum/curriculum_*_a1.sqlite"
echo "   gs://${BUCKET_NAME}/curriculum/curriculum_*_a2.sqlite"
