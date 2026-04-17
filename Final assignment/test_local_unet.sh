#!/bin/bash

echo "Building Docker image for local testing for U-Net..."
docker build -t nncv-submission:unet .

echo "Running Docker container for local testing..."
docker run --network none --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:unet

echo "Plotting results:"
uv run inspect_results.py