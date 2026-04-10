FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*


# create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"


# install dependencies (add any additional dependencies your project requires)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    pillow torchmetrics termcolor normflows

# Define the argument with a default value
ARG PREDICT_MODE=predict

WORKDIR /app

COPY dinov3 /app/dinov3
COPY head.py /app/head.py
COPY ood_model.py /app/ood_model.py
COPY model_L16_All_MLP8_res1024.pt /app/model.pt
COPY ood_detector_weights_v2.pt /app/ood_detector_weights.pt
COPY model.py /app/model.py
COPY "${PREDICT_MODE}.py" /app/predict.py
ENTRYPOINT ["python", "/app/predict.py"]