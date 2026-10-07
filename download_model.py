#!/usr/bin/env python3
"""
Download script for dual-model multimodal RAG architecture.
Downloads:
  1. Embedding Model: sentence-transformers/all-MiniLM-L6-v2
  2. EasyOCR Models: CRAFT (text detection) + English recognition engine
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


# Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODELS_DIR = Path("./models")
EMBEDDING_TARGET = MODELS_DIR / "all-MiniLM-L6-v2"
EASYOCR_TARGET = MODELS_DIR / "easyocr"


def create_model_directories() -> None:
    """Ensure model directories exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_TARGET.mkdir(parents=True, exist_ok=True)
    EASYOCR_TARGET.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Created model directories at {MODELS_DIR}")


def download_embedding_model() -> bool:
    """Download sentence-transformers embedding model."""
    try:
        print(f"[Info] Downloading embedding model: {EMBEDDING_MODEL}...")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        target = snapshot_download(
            repo_id=EMBEDDING_MODEL,
            cache_dir=str(MODELS_DIR),
            local_dir=str(EMBEDDING_TARGET),
            local_dir_use_symlinks=False,
        )
        print(f"[Success] Embedding model downloaded to: {target}")
        return True
    except Exception as exc:
        print(f"[Error] Failed to download embedding model: {exc}")
        return False


def download_easyocr_models() -> bool:
    """Download EasyOCR models (CRAFT + English recognition)."""
    try:
        print(f"[Info] Downloading EasyOCR models to {EASYOCR_TARGET}...")
        import easyocr

        # Initialize reader with English language; this forces model download
        reader = easyocr.Reader(
            ["en"],
            download_enabled=True,
            model_storage_directory=str(EASYOCR_TARGET),
            gpu=False,
        )
        print("[Success] EasyOCR models downloaded successfully")
        return True
    except Exception as exc:
        print(f"[Error] Failed to download EasyOCR models: {exc}")
        return False


def main() -> int:
    """
    Main entry point for model download.
    
    Returns:
        0 if all downloads succeeded, 1 if any download failed.
    """
    print("[Info] Starting dual-model download process...")
    
    try:
        create_model_directories()
    except Exception as exc:
        print(f"[Error] Failed to create model directories: {exc}")
        return 1
    
    # Download embedding model
    embedding_ok = download_embedding_model()
    
    # Download EasyOCR models
    easyocr_ok = download_easyocr_models()
    
    if embedding_ok and easyocr_ok:
        print("[Info] All models downloaded successfully!")
        return 0
    else:
        if not embedding_ok:
            print("[Warning] Embedding model download failed")
        if not easyocr_ok:
            print("[Warning] EasyOCR models download failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
