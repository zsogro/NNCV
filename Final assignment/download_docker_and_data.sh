#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --partition=gpu_a100
#SBATCH --time=3:00:00

# Pull container from dockerhub
apptainer pull container_v2.sif docker://zsogro/nncv-dinov3:v2

# Use the huggingface-cli package inside the container to download the data
# mkdir -p data
# apptainer exec container.sif \
#     huggingface-cli download TimJaspersTue/5LSM0 --local-dir ./data --repo-type dataset --resume-download
