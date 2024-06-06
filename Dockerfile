FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel

COPY requirements.txt requirements.txt

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir --upgrade --requirement requirements.txt --find-links https://data.pyg.org/whl/torch-2.2.1+cu118.html --extra-index-url https://download.pytorch.org/whl/cu118
