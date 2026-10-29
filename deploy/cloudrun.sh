#!/usr/bin/env bash
# Deploy the demo to Cloud Run. Run from the repo root after:
#   gcloud auth login && gcloud config set project <PROJECT_ID>
#   (one time) store the key as a secret, reading it from YOUR shell so it is never printed:
#     printf %s "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
#   (one time) let the service account read it:
#     gcloud secrets add-iam-policy-binding gemini-api-key \
#       --member="serviceAccount:$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')-compute@developer.gserviceaccount.com" \
#       --role=roles/secretmanager.secretAccessor
set -euo pipefail

REGION="${REGION:-asia-southeast1}"
SERVICE="${SERVICE:-cited}"
[ -d data/index ] || { echo "data/index missing: run 'cited ingest' first" >&2; exit 1; }

# Stage a minimal build context so data/, .env and .venv never go up with the source.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R pyproject.toml README.md src "$STAGE"/
mkdir -p "$STAGE/data" && cp -R data/index "$STAGE/data/index"
cp deploy/Dockerfile "$STAGE/Dockerfile"

# One instance, because the demo guard keeps its counters in memory.
gcloud run deploy "$SERVICE" --source "$STAGE" --region "$REGION" \
  --allow-unauthenticated \
  --max-instances 1 --min-instances 0 --memory 512Mi --cpu 1 --concurrency 10 --timeout 120 \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars CITED_DEMO_IP_PER_MIN=4,CITED_DEMO_IP_PER_DAY=20,CITED_DEMO_GLOBAL_PER_DAY=120,CITED_TRUSTED_HOPS=1,CITED_RPM=20
