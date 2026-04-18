# Final Assignment: Cityscape Challenge  

## Prerequisits
My model is using the DINOv3 backbone with a multi-depth segmentation head and an OOD detector.

1. You can get the DINO backbone weights from Facebook AI research [(ViT-L/16 distilled)](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), just sign up and they are very nice and send you the download link.
2. To use the model with this backbone, clone the [dinov3](https://github.com/facebookresearch/dinov3) into this directory (the Final Assignment dir).
3. To train the model, you will need the `torchmetrics`, `termcolors` and for the OOD detectors the `normflows` libraries, but worry not my friend, I already built the Docker container for ya (45 mins from my life), so just pull it with the `download_docker_and_data.sh` by changing the line to this: 
   ```bash
   apptainer pull container_v3.sif docker://zsogro/nncv-dinov3:v3
   ```
4. Change the last line in `jobscript_slurm.sh` to
   ```bash
   srun apptainer exec --nv --env-file .env container_v3.sif /bin/bash main.sh
   ```
5. You should have and init `uv`, if not, change the helpers when testing to run the inspect_results.py accordingly.

## Training
### 1. Segmentation Model
1. Change the experiment id in `jobscript_slurm.sh` to something logical.
2. Make sure that main.sh runs the `train.py`, then just type `sbatch jobscript_slurm.sh` and hit enter. Then to check the status type `watch squeue`.

### 2. Out-of-Distribution Model
1. Change the experiment id in `jobscript_slurm.sh` to something logical.
2. Choose between OOD_Detector_v1 and v2, you have to change lines `144` and `225` in `train_ood.py`.
3. Make sure that main.sh runs the `train_ood.py`, then just type `sbatch jobscript_slurm.sh` and hit enter. Then to check the status type `watch squeue`.

## Testing
I made a few helper scripts so you don't have to build the Docker when you want to test it. 
1. Place your images to the `local_data` directory.
2. In case of testing the OOD, select between v1/v2 and choose the threshold in `predict_ood.py` in lines `183-184`, and change the ood weights in the Dockerfile accordingly.
3. Type `./test_local_peak.sh` or `./test_local_ood.sh`.
4. Inspect the results in the `colorized_output` directory.
