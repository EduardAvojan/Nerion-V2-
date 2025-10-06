#!/bin/bash
# Submit curriculum generation job to Vertex AI using Cloud Build

set -e

PROJECT_ID="nerion-vertex-project"
REGION="us-central1"
BUCKET_NAME="nerion-training-data"
IMAGE_URI="gcr.io/${PROJECT_ID}/nerion-trainer:latest"
JOB_NAME="curriculum-gen-$(date +%Y%m%d-%H%M%S)"

echo "🚀 Submitting Curriculum Generation Job to Vertex AI"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Job Name: ${JOB_NAME}"
echo ""

# Build using Cloud Build (no local Docker needed!)
echo "📦 Building Docker image with Cloud Build..."
gcloud builds submit --config=cloudbuild.yaml --project=${PROJECT_ID} .

# Submit training job with parallel workers
echo "🎯 Submitting Vertex AI Custom Job..."
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec=machine-type=n1-highmem-8,replica-count=1,container-image-uri=${IMAGE_URI} \
  --args="--cycles=2000,--bucket=${BUCKET_NAME},--provider=vertexai:gemini-2.5-pro,--project-id=${PROJECT_ID},--location=${REGION}" \
  --project=${PROJECT_ID}

echo ""
echo "✅ Job submitted successfully!"
echo "   Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
