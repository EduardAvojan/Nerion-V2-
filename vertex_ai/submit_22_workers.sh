#!/bin/bash
# Submit 22 parallel curriculum generation jobs to Vertex AI with n1-highmem-8

set -e

PROJECT_ID="nerion-vertex-project"
REGION="us-central1"
BUCKET_NAME="nerion-training-data"
IMAGE_URI="gcr.io/${PROJECT_ID}/nerion-trainer:latest"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Number of parallel jobs
NUM_JOBS=22
CYCLES_PER_JOB=200
TOTAL_CYCLES=$((NUM_JOBS * CYCLES_PER_JOB))

echo "🚀 Submitting ${NUM_JOBS} Parallel Curriculum Generation Jobs to Vertex AI"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Machine: n1-highmem-8 (8 CPUs, 52GB RAM)"
echo "   Cycles per job: ${CYCLES_PER_JOB}"
echo "   Total cycles: ${TOTAL_CYCLES}"
echo ""

# Build using Cloud Build (no local Docker needed!)
echo "📦 Building Docker image with Cloud Build..."
gcloud builds submit --config=cloudbuild.yaml --project=${PROJECT_ID} .

echo ""
echo "🎯 Submitting ${NUM_JOBS} parallel jobs..."

# Submit multiple jobs in parallel
for i in $(seq 1 $NUM_JOBS); do
    JOB_NAME="curriculum-gen-${TIMESTAMP}-worker${i}"

    echo "   Submitting job ${i}/${NUM_JOBS}: ${JOB_NAME}"

    gcloud ai custom-jobs create \
      --region=${REGION} \
      --display-name=${JOB_NAME} \
      --worker-pool-spec=machine-type=n1-highmem-8,replica-count=1,container-image-uri=${IMAGE_URI} \
      --args="--cycles=${CYCLES_PER_JOB},--bucket=${BUCKET_NAME},--provider=vertexai:gemini-2.5-pro,--project-id=${PROJECT_ID},--location=${REGION}" \
      --project=${PROJECT_ID} &

    # Small delay to avoid rate limiting
    sleep 2
done

# Wait for all background jobs to complete
wait

echo ""
echo "✅ All ${NUM_JOBS} jobs submitted successfully!"
echo "   Total cycles: ${TOTAL_CYCLES}"
echo "   Machine type: n1-highmem-8 (52GB RAM to prevent OOM)"
echo "   Monitor at: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "📊 Each job will upload checkpoints to: gs://${BUCKET_NAME}/curriculum/"
echo "   Use merge script to combine all checkpoints when complete"
