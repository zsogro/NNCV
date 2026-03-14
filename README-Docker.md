# Docker Container Guide

- Add libraries to the `Dockerfile`:
```bash
  RUN pip3 install wandb torchmetrics termcolor
  ``` 
### Container for Training
- Build your training image from the `NNCV` folder:
  ```bash
  docker build -t nncv-dinov3:latest .
  ```
- Tag it, for example `v2`:
  ```bash
  docker tag nncv-dinov3:latest zsogro/nncv-dinov3:v2
  ```
- Push it to Docker Hub
  ```bash
  docker push zsogro/nncv-dinov3:v2
  ```
- Change the line in the `download_docker_and_data.sh` :
```bash
  apptainer pull container_v2.sif docker://zsogro/nncv-dinov3:v2
  ```