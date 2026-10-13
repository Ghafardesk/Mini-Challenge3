# syntax=docker/dockerfile:1.7

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
RUN --mount=type=secret,id=hf_token,required=true \
    if [ ! -s /run/secrets/hf_token ]; then \
        echo "HF_TOKEN build secret is missing or empty; provide an approved Hugging Face token." >&2; \
        exit 1; \
    fi && \
    export HF_TOKEN="$(cat /run/secrets/hf_token)" && \
    python3 /app/download_model.py

RUN mkdir -p /app/output /models /app/corpus

CMD ["sleep", "infinity"]
