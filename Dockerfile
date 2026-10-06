FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    TORCH_HOME=/models \
    APP_HOME=/app

WORKDIR /app

COPY . /app
RUN chmod +x /app/*.py

RUN python3 - <<'PY'
import torch
print(torch.__version__)
assert '+rocm' in torch.__version__, f"Unexpected torch build: {torch.__version__}"
PY

RUN python3 -m pip install --no-deps --no-cache-dir -r /app/requirements.txt
RUN python3 /app/download_model.py

RUN mkdir -p /app/output /models /app/corpus

EXPOSE 8080

CMD ["bash", "-lc", "python3 /app/daemon.py --serve & sleep 3; tail -f /dev/null"]
