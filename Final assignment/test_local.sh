#!/bin/bash

echo "Building Docker image for local testing..."
docker build -t nncv-submission:latest .

echo "Running Docker container for local testing..."
docker run --network none --rm \
  -v "$(pwd)/local_data:/data" \
  -v "$(pwd)/local_output:/output" \
  nncv-submission:latest