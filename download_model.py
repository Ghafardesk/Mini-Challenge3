#!/usr/bin/env python3
import os
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TARGET_DIR = Path("/models")
TARGET_DIR.mkdir(parents=True, exist_ok=True)
def main() -> None:
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    target = snapshot_download(repo_id=MODEL_NAME, cache_dir=str(TARGET_DIR), local_dir=str(TARGET_DIR / "all-MiniLM-L6-v2"), local_dir_use_symlinks=False)
    print(f"Downloaded model to {target}")


if __name__ = "__main__":
    main()