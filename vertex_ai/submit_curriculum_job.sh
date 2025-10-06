#!/bin/bash
# Submit curriculum generation job to Vertex AI

set -e

PROJECT_ID="nerion-vertex-project"
REGION="us-central1"
BUCKET_NAME="nerion-training-data"  # You'll need to create this
IMAGE_URI="gcr.io/${PROJECT_ID}/nerion-trainer:latest"
JOB_NAME="curriculum-gen-$(date +%Y%m%d-%H%M%S)"

echo "🚀 Submitting Curriculum Generation Job to Vertex AI"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Job Name: ${JOB_NAME}"
echo ""

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -f Dockerfile.vertex -t ${IMAGE_URI} .

echo "📤 Pushing to Container Registry..."
docker push ${IMAGE_URI}

# Submit training job
echo "🎯 Submitting Vertex AI Custom Job..."
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec=machine-type=n1-standard-8,replica-count=10,container-image-uri=${IMAGE_URI} \
  --args="--cycles=100,--bucket=${BUCKET_NAME},--provider=vertexai:gemini-2.5-pro" \
  --project=${PROJECT_ID}

echo ""
echo "✅ Job submitted successfully!"
echo "   Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
