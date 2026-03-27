#!/bin/bash

echo "Building Docker image for local testing..."
docker build --build-arg PREDICT_MODE=predict_ood -t nncv-submission:ood .

echo "Running Docker container for local testing..."
docker run --network none --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:ood

echo "Plotting results:"
uv run inspect_results.py