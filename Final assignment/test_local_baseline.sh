#!/bin/bash

echo "Building Docker image for local testing..."
docker build -t nncv-submission:baseline .

echo "Running Docker container for local testing..."
docker run --network none --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:baseline

echo "Plotting results:"
uv run inspect_results.py