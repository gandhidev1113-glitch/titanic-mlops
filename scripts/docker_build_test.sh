#!/usr/bin/env bash
# =============================================================================
# scripts/docker_build_test.sh
# Build both Docker targets and run a quick smoke test against the API.
# Usage: bash scripts/docker_build_test.sh
# =============================================================================
set -euo pipefail

TAG="${1:-latest}"
API_PORT=8000
CONTAINER_NAME="titanic-api-smoke"

echo "🔨  Building TRAIN image (titanic-train:$TAG)..."
docker build --target train  -t "titanic-train:$TAG" .

echo "🔨  Building INFERENCE image (titanic-api:$TAG)..."
docker build --target inference -t "titanic-api:$TAG" .

echo "🚀  Starting inference container on port $API_PORT..."
docker run -d \
  --name "$CONTAINER_NAME" \
  -p "$API_PORT:8000" \
  -v "$(pwd)/models:/app/models" \
  "titanic-api:$TAG"

cleanup() {
  echo "🧹  Stopping & removing container..."
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "⏳  Waiting for API to be ready..."
for i in $(seq 1 20); do
  if curl -sf "http://localhost:$API_PORT/health" >/dev/null 2>&1; then
    echo "✅  API is up!"
    break
  fi
  echo "   attempt $i/20..."
  sleep 2
done

echo ""
echo "📋  GET /health"
curl -s "http://localhost:$API_PORT/health" | python3 -m json.tool

echo ""
echo "📋  GET /ready"
curl -s "http://localhost:$API_PORT/ready" | python3 -m json.tool

echo ""
echo "📋  POST /predict (sample passenger)"
curl -s -X POST "http://localhost:$API_PORT/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 29.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 80.0,
    "embarked": "S",
    "title": "Mrs"
  }' | python3 -m json.tool

echo ""
echo "🎉  Smoke test passed!"
