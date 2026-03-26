#!/bin/bash

echo "Building Docker image for local testing..."
docker build --build-arg PREDICT_MODE=predict -t nncv-submission:peak .

echo "Running Docker container for local testing..."
docker run --network none --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:peak

echo "Comparing results:"
uv run inspect_results.py