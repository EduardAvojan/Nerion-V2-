#!/bin/bash
# Submit 22 parallel curriculum generation jobs, one per category

set -e

PROJECT_ID="nerion-vertex-project"
REGION="us-central1"
BUCKET_NAME="nerion-training-data"
IMAGE_URI="gcr.io/${PROJECT_ID}/nerion-trainer:latest"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

# Categories (must match learning_orchestrator.py)
CATEGORIES=(
    "refactoring"
    "bug_fixing"
    "feature_implementation"
    "performance_optimization"
    "code_comprehension"
    "security_hardening"
    "testing_strategies"
    "api_design"
    "concurrency_patterns"
    "data_structures"
    "algorithmic_optimization"
    "error_recovery"
    "database_design"
    "observability"
    "deployment_cicd"
    "caching_strategies"
    "message_queues"
    "resource_management"
    "distributed_systems"
    "scaling_patterns"
    "data_validation"
    "configuration_management"
)

NUM_JOBS=${#CATEGORIES[@]}
CYCLES_PER_JOB=100
TOTAL_CYCLES=$((NUM_JOBS * CYCLES_PER_JOB))

echo "🚀 Submitting ${NUM_JOBS} Category-Specific Curriculum Generation Jobs"
echo "   Project: ${PROJECT_ID}"
echo "   Region: ${REGION}"
echo "   Machine: n1-highmem-8 (8 CPUs, 52GB RAM)"
echo "   Cycles per category: ${CYCLES_PER_JOB}"
echo "   Total cycles: ${TOTAL_CYCLES}"
echo "   Expected unique lessons: ~1100-1540 (no cross-category duplicates)"
echo ""

# Build using Cloud Build
echo "📦 Building Docker image with Cloud Build..."
gcloud builds submit --config=cloudbuild.yaml --project=${PROJECT_ID} .

echo ""
echo "🎯 Submitting ${NUM_JOBS} category-specific jobs..."

# Submit one job per category
for i in "${!CATEGORIES[@]}"; do
    CATEGORY="${CATEGORIES[$i]}"
    JOB_NAME="curriculum-${TIMESTAMP}-${CATEGORY}"
    JOB_NUM=$((i + 1))

    echo "   [${JOB_NUM}/${NUM_JOBS}] Submitting: ${CATEGORY} → ${JOB_NAME}"

    gcloud ai custom-jobs create \
      --region=${REGION} \
      --display-name=${JOB_NAME} \
      --worker-pool-spec=machine-type=n1-highmem-8,replica-count=1,container-image-uri=${IMAGE_URI} \
      --args="--cycles=${CYCLES_PER_JOB},--bucket=${BUCKET_NAME},--provider=vertexai:gemini-2.5-pro,--project-id=${PROJECT_ID},--location=${REGION},--category=${CATEGORY}" \
      --project=${PROJECT_ID} &

    # Small delay to avoid rate limiting
    sleep 2
done

# Wait for all background submissions to complete
wait

echo ""
echo "✅ All ${NUM_JOBS} category-specific jobs submitted!"
echo ""
echo "📊 Summary:"
echo "   Total cycles: ${TOTAL_CYCLES}"
echo "   Machine type: n1-highmem-8 (52GB RAM to prevent OOM)"
echo "   Zero cross-category duplicates (each worker has unique category)"
echo "   Monitor: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
echo ""
echo "💾 Checkpoints will be uploaded to: gs://${BUCKET_NAME}/curriculum/"
echo "   Each category creates separate checkpoint files"
echo "   Merge all checkpoints when jobs complete"
