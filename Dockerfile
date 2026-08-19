FROM --platform=linux/amd64 python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    POSE_BACKEND=horse10_mmpose \
    HEADLESS=1 \
    INPUT_VIDEO=datasets/raw/test.mp4 \
    OUTPUT_VIDEO=/app/output/annotated.mp4 \
    CUDA_VISIBLE_DEVICES=""

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt constraints.txt ./

RUN pip install --no-cache-dir "pip<25" wheel \
    && pip install --no-cache-dir -c constraints.txt setuptools==69.5.1 "numpy==1.26.4" cython \
    && pip install --no-cache-dir --no-build-isolation --no-deps -c constraints.txt xtcocotools \
    && pip install --no-cache-dir -c constraints.txt torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -c constraints.txt mmengine \
    && pip install --no-cache-dir -c constraints.txt mmcv==2.1.0 \
        -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html \
    && pip install --no-cache-dir -c constraints.txt "mmdet>=3.2.0,<3.4.0" \
    && pip install --no-cache-dir -c constraints.txt "mmpose>=1.3.0" --no-deps \
    && pip install --no-cache-dir -c constraints.txt json-tricks munkres matplotlib scipy pillow \
    && pip install --no-cache-dir -c constraints.txt -r requirements.txt "filelock>=3.16.1" \
    && pip install --no-cache-dir -c constraints.txt setuptools==69.5.1 "numpy==1.26.4" \
    && pip install --no-cache-dir --no-build-isolation --no-deps --force-reinstall xtcocotools \
    && pip install --no-cache-dir -c constraints.txt setuptools==69.5.1 "numpy==1.26.4" \
    && python -c "\
import pkg_resources; \
import numpy as np; \
import torch; \
assert np.__version__ == '1.26.4'; \
torch.from_numpy(np.zeros((3, 64, 64), dtype=np.uint8)); \
from ultralytics import YOLO; \
from mmpose.apis.inference import init_model, inference_topdown; \
import xtcocotools; \
print('dependency check ok')\
"

# Model files live outside /app so docker-compose's .:/app bind mount does not hide them.
RUN pip install --no-cache-dir -c constraints.txt openmim \
    && mim download mmpose --config td-hm_hrnet-w48_8xb64-210e_animalpose-256x256 --dest /opt/mmpose-model \
    && pip install --no-cache-dir -c constraints.txt setuptools==69.5.1 "numpy==1.26.4"

ENV MMPOSE_MODEL_DIR=/opt/mmpose-model \
    MMPOSE_CONFIG=/opt/mmpose-model/td-hm_hrnet-w48_8xb64-210e_animalpose-256x256.py

RUN mkdir -p /app/output

COPY . .

CMD ["python", "main.py"]
