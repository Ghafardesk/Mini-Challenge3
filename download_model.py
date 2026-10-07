#!/usr/bin/env python3
"""
Download script for dual-model vision-language RAG architecture.
Downloads models from Hugging Face Hub into local ./models/ directory:
  1. Embedding Model: sentence-transformers/all-MiniLM-L6-v2
  2. Vision-Language Model: meta-llama/Llama-3.2-11B-Vision-Instruct
"""
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download, HfApi


# Configuration
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
VISION_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-11B-Vision-Instruct")

MODELS_DIR = Path("./models")
EMBEDDING_TARGET = MODELS_DIR / "all-MiniLM-L6-v2"
VISION_TARGET = MODELS_DIR / "Llama-3.2-11B-Vision-Instruct"


def create_model_directories() -> None:
    """Ensure model directories exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EMBEDDING_TARGET.mkdir(parents=True, exist_ok=True)
    VISION_TARGET.mkdir(parents=True, exist_ok=True)
    print(f"[Installer] Created model directories at {MODELS_DIR}")


def verify_huggingface_auth() -> bool:
    """
    Verify HuggingFace authentication.
    
    Returns:
        True if authenticated, False otherwise.
    """
    hf_token = os.getenv("HF_TOKEN")
    try:
        api = HfApi(token=hf_token)
        user_info = api.whoami()
        print(f"[Auth] Logged in to Hugging Face as: {user_info['name']}")
        return True
    except Exception as exc:
        print(f"[Warning] HF Token not detected or invalid: {exc}")
        print("[Info] Make sure HF_TOKEN environment variable is set for gated Llama models.")
        return False


def download_embedding_model() -> bool:
    """Download sentence-transformers embedding model."""
    try:
        print(f"[Installer] [1/2] Downloading Embedding Model: {EMBEDDING_MODEL_ID}...")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        target = snapshot_download(
            repo_id=EMBEDDING_MODEL_ID,
            local_dir=str(EMBEDDING_TARGET),
            resume_download=True,
        )
        print(f"[Success] Embedding model downloaded to: {target}")
        return True
    except Exception as exc:
        print(f"[Error] Failed to download embedding model: {exc}")
        return False


def download_vision_model() -> bool:
    """Download Llama-3.2-11B-Vision-Instruct model."""
    try:
        print(f"[Installer] [2/2] Downloading Vision-Language Model: {VISION_MODEL_ID}...")
        hf_token = os.getenv("HF_TOKEN")
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
        
        target = snapshot_download(
            repo_id=VISION_MODEL_ID,
            local_dir=str(VISION_TARGET),
            token=hf_token,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"],
            resume_download=True,
        )
        print(f"[Success] Llama Vision model downloaded to: {target}")
        return True
    except Exception as exc:
        print(f"[Error] Failed to download Llama Vision model: {exc}")
        return False


def main() -> int:
    """
    Main entry point for model download.
    
    Returns:
        0 if all downloads succeeded, 1 if any download failed.
    """
    print("[Installer] Starting dual-model download process...")
    print(f"[Installer] Local Cache Path: {MODELS_DIR.resolve()}")
    
    try:
        create_model_directories()
    except Exception as exc:
        print(f"[Error] Failed to create model directories: {exc}")
        return 1
    
    # Verify Hugging Face authentication
    print()
    verify_huggingface_auth()
    print()
    
    # Download embedding model
    embedding_ok = download_embedding_model()
    
    # Download vision model
    vision_ok = download_vision_model()
    
    if embedding_ok and vision_ok:
        print("\n[Installer] All models downloaded successfully!")
        return 0
    else:
        if not embedding_ok:
            print("[Warning] Embedding model download failed")
        if not vision_ok:
            print("[Warning] Vision model download failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
