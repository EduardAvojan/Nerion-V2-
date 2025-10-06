#!/bin/bash
# Submit CEFR-targeted curriculum generation job to Vertex AI

set -e

PROJECT_ID="nerion-vertex-project"
REGION="us-central1"
BUCKET_NAME="nerion-training-data"
IMAGE_URI="gcr.io/${PROJECT_ID}/nerion-trainer:latest"
JOB_NAME="cefr-curriculum-$(date +%Y%m%d-%H%M%S)"

echo "🎯 CEFR-Targeted Curriculum Generation - Vertex AI"
echo "=================================================="
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Job Name: ${JOB_NAME}"
echo "   Target: 245 CEFR lessons (sequential)"
echo ""

# Build and push Docker image
echo "📦 Building Docker image..."
docker build -f Dockerfile.vertex -t ${IMAGE_URI} .

echo "📤 Pushing to Container Registry..."
docker push ${IMAGE_URI}

# Submit training job with CEFR mode
echo "🚀 Submitting Vertex AI Custom Job..."
gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name=${JOB_NAME} \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=${IMAGE_URI} \
  --args="--cycles=245,--bucket=${BUCKET_NAME},--provider=vertexai:gemini-2.5-flash,--mode=cefr" \
  --project=${PROJECT_ID}

echo ""
echo "✅ Job submitted successfully!"
echo "   Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "📊 This will generate 245 CEFR-targeted lessons:"
echo "   - 30 A1 (beginner)"
echo "   - 30 A2 (elementary)"
echo "   - 40 B1 (intermediate)"
echo "   - 35 B2 (upper intermediate)"
echo "   - 70 C1 (advanced)"
echo "   - 40 C2 (expert)"
