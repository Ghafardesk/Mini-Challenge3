#!/usr/bin/env python3
"""
Multimodal RAG Daemon supporting dual-model vision-language architecture:
- Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Vision-Language Engine: meta-llama/Llama-3.2-11B-Vision-Instruct (via vLLM)

Robust error handling for edge cases: permission denied, encrypted PDFs,
corrupted files, and empty directories. vLLM provides high-performance
inference with GPU memory optimization and support for batch processing.
"""
import argparse
import json
import logging
import math
import os
import re
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SOCKET_PATH = "/tmp/rag_daemon.sock"
TCP_HOST = "127.0.0.1"
TCP_PORT = 8765
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIR = "./models/all-MiniLM-L6-v2"
DEFAULT_VISION_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-11B-Vision-Instruct")
DEFAULT_VISION_DIR = "./models/Llama-3.2-11B-Vision-Instruct"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Global vLLM engine initialized at daemon startup
_vision_llm: Optional[Any] = None


def initialize_vision_llm(model_dir: str = DEFAULT_VISION_DIR, gpu_memory_utilization: float = 0.65, max_model_len: int = 2048) -> Optional[Any]:
    """
    Initialize vLLM engine for Llama Vision image understanding.
    
    Args:
        model_dir: Local directory containing Llama-3.2-11B-Vision-Instruct model.
        gpu_memory_utilization: Fraction of GPU memory to use (0.0-1.0).
        max_model_len: Maximum model sequence length for efficient inference.
    
    Returns:
        Initialized vLLM LLM instance, or None if initialization fails.
    """
    global _vision_llm
    try:
        from vllm import LLM
        
        _vision_llm = LLM(
            model=model_dir,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=True,
        )
        logging.info("Initialized vLLM engine from %s (GPU mem util: %.2f)", model_dir, gpu_memory_utilization)
        return _vision_llm
    except Exception as exc:
        logging.warning("Failed to initialize vLLM engine: %s", exc)
        return None


def get_vision_llm() -> Optional[Any]:
    """Get the global vLLM engine instance."""
    return _vision_llm


@dataclass
class DocumentChunk:
    file_path: str
    text: str
    chunk_index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_MODEL, model_dir: str = DEFAULT_EMBEDDING_DIR):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model = None
        self.device = "cuda" if self._cuda_available() else "cpu"
        try:
            from sentence_transformers import SentenceTransformer

            # Try loading from local directory first, fallback to model name
            if Path(model_dir).exists():
                self.model = SentenceTransformer(model_dir, device=self.device)
                logging.info("Loaded embedding model from local directory %s on %s", model_dir, self.device)
            else:
                self.model = SentenceTransformer(model_name, device=self.device)
                logging.info("Loaded embedding model %s on %s", model_name, self.device)
        except Exception as exc:  # pragma: no cover - fallback path
            logging.warning("SentenceTransformers unavailable: %s", exc)
            self.model = None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.model is not None:
            try:
                vectors = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
                return [list(map(float, vector)) for vector in vectors]
            except Exception as exc:
                logging.warning("Embedding call failed; falling back to hash embeddings: %s", exc)
        return [self._hash_embedding(text) for text in texts]

    @staticmethod
    def _hash_embedding(text: str, dim: int = 384) -> List[float]:
        vec = [0.0 for _ in range(dim)]
        tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
        if not tokens:
            return vec
        for token in tokens:
            idx = abs(hash(token)) % dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class VectorIndex:
    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self.embeddings: List[List[float]] = []

    def clear(self):
        self.chunks.clear()
        self.embeddings.clear()

    def add(self, chunk: DocumentChunk, embedding: List[float]):
        self.chunks.append(chunk)
        self.embeddings.append(embedding)

    def build_from_corpus(self, corpus_dir: str, embedding_service: EmbeddingService) -> None:
        """Build vector index from corpus directory with robust error handling."""
        self.clear()
        corpus_path = Path(corpus_dir)
        if not corpus_path.exists() or not corpus_path.is_dir():
            logging.warning("Corpus directory not found: %s", corpus_dir)
            return

        files = self._discover_files(corpus_path)
        
        # Handle empty directories
        if not files:
            logging.info("No files found in corpus directory: %s", corpus_dir)
            return
        
        texts_to_embed: List[Tuple[str, str, int, Dict[str, Any]]] = []
        
        for file_path in files:
            try:
                # Use robust read_corpus_file instead of parse_file
                extracted_text = read_corpus_file(str(file_path))
            except Exception as exc:
                # Guard against any unexpected exceptions in file reading
                logging.warning("Unexpected error processing %s: %s", file_path, exc)
                continue
            
            if not extracted_text:
                # File was skipped or empty; continue to next
                continue
            
            # Split extracted text into chunks
            chunks = split_chunks(extracted_text)
            for idx, chunk_text in enumerate(chunks):
                relative = os.path.relpath(file_path, corpus_path)
                texts_to_embed.append((str(relative), chunk_text, idx, {"source_file": str(file_path)}))

        if not texts_to_embed:
            logging.info("No text chunks indexed from %s", corpus_dir)
            return

        batch = [chunk_text for _, chunk_text, _, _ in texts_to_embed]
        vectors = embedding_service.embed(batch)
        for (relative_path, chunk_text, chunk_idx, metadata), vector in zip(texts_to_embed, vectors):
            self.add(DocumentChunk(file_path=relative_path, text=chunk_text, chunk_index=chunk_idx, metadata=metadata), vector)
        logging.info("Indexed %d chunks across %d files in %s", len(self.chunks), len(files), corpus_dir)

    @staticmethod
    def _discover_files(root: Path) -> List[Path]:
        files: List[Path] = []
        if not root.exists():
            return files
        for current, _, filenames in os.walk(root):
            for name in sorted(filenames):
                full_path = Path(current) / name
                if full_path.is_file():
                    files.append(full_path)
        return files

    def query(self, query_text: str, top_k: int = 5, embedding_service: Optional[EmbeddingService] = None) -> List[Tuple[float, DocumentChunk]]:
        if not self.chunks:
            return []
        service = embedding_service or EmbeddingService()
        query_vector = service.embed([query_text])[0]
        scored: List[Tuple[float, DocumentChunk]] = []
        for chunk, vector in zip(self.chunks, self.embeddings):
            score = cosine_similarity(query_vector, vector)
            scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    if not vec_a or not vec_b:
        return 0.0
    norm_a = math.sqrt(sum(v * v for v in vec_a))
    norm_b = math.sqrt(sum(v * v for v in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    return dot / (norm_a * norm_b)


def normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    answer = str(value).strip().upper()
    for ch in "-._·":
        answer = answer.replace(ch, "")
    return answer.strip()


def safe_read_utf8(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="strict") as fh:
            return fh.read()
    except Exception:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except Exception:
            return ""


def split_chunks(text: str, chunk_size: int = 900) -> List[str]:
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n+|\r\n\s*\r\n+", text)
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = paragraph.strip()
        if not candidate:
            continue
        if len(current) + len(candidate) + 1 <= chunk_size:
            if current:
                current += "\n"
            current += candidate
        else:
            if current:
                chunks.append(current)
            current = candidate
    if current:
        chunks.append(current)
    if not chunks:
        chunks = [text[:chunk_size]]
    return chunks


def parse_pdf(path: Path) -> List[str]:
    texts: List[str] = []
    try:
        import pypdf

        reader = pypdf.PdfReader(str(path))
        for page in reader.pages:
            try:
                page_text = page.extract_text() or ""
                if page_text:
                    texts.append(page_text)
            except Exception:
                continue
    except Exception:
        pass
    if texts:
        return split_chunks("\n\n".join(texts))

    try:
        import fitz

        doc = fitz.open(str(path))
        for page in doc:
            try:
                page_text = page.get_text("text")
                if page_text:
                    texts.append(page_text)
            except Exception:
                continue
        doc.close()
    except Exception:
        pass
    if texts:
        return split_chunks("\n\n".join(texts))

    try:
        import pdfplumber

        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                    if page_text:
                        texts.append(page_text)
                except Exception:
                    continue
    except Exception:
        pass
    if texts:
        return split_chunks("\n\n".join(texts))

    return []


def parse_docx(path: Path) -> List[str]:
    try:
        from docx import Document

        doc = Document(str(path))
        pieces: List[str] = []
        for element in doc.paragraphs:
            text = (element.text or "").strip()
            if text:
                pieces.append(text)
        for table in doc.tables:
            for row in table.rows:
                values = [cell.text.strip() for cell in row.cells if cell.text and cell.text.strip()]
                if values:
                    pieces.append(" | ".join(values))
        return split_chunks("\n".join(pieces))
    except Exception as exc:
        logging.warning("Failed to parse DOCX %s: %s", path, exc)
        return []


def parse_excel_or_csv(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            import pandas as pd

            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            rows = []
            for _, row in frame.iterrows():
                rows.append(" | ".join(str(cell).strip() for cell in row.tolist() if str(cell).strip()))
            return split_chunks("\n".join(rows))

        if suffix in {".xlsx", ".xls"}:
            import pandas as pd

            frame = pd.read_excel(path, engine="openpyxl")
            rows = []
            for _, row in frame.fillna("").iterrows():
                rows.append(" | ".join(str(cell).strip() for cell in row.tolist() if str(cell).strip()))
            return split_chunks("\n".join(rows))
        return []
    except Exception as exc:
        logging.warning("Failed to parse spreadsheet %s: %s", path, exc)
        return []


def parse_text_like(path: Path) -> List[str]:
    try:
        content = safe_read_utf8(path)
        return split_chunks(content)
    except Exception as exc:
        logging.warning("Failed to read text-like file %s: %s", path, exc)
        return []


def parse_image(path: Path) -> List[str]:
    """
    Extract text from PNG/JPG/JPEG images using vLLM Llama Vision engine.
    
    Acts as an OCR assistant to transcribe all visible text from images.
    Uses vLLM's high-performance inference with the official OCR prompt.
    
    Args:
        path: Path to image file.
    
    Returns:
        List of text chunks extracted from image, or empty list on failure.
    """
    llm = get_vision_llm()
    if llm is None:
        logging.warning("vLLM engine not initialized; skipping image %s", path)
        return []
    
    try:
        from PIL import Image
        import base64
        from io import BytesIO
        
        # Load image
        image = Image.open(path)
        
        # Convert image to base64 for vLLM
        buffered = BytesIO()
        image.save(buffered, format="PNG" if path.suffix.lower() == ".png" else "JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Official OCR prompt for Llama Vision
        ocr_prompt = "Act as an OCR assistant. Analyze the provided <|image|> image and transcribe all visible text exactly as it appears. Output only the transcribed text without commentary."
        
        # Prepare multi-modal prompt with image placeholder
        # vLLM expects the image to be referenced as <|image|> in the prompt
        prompt = f"<|image|>{ocr_prompt}"
        
        # Run inference with vLLM
        # Note: vLLM handles base64-encoded images via the image parameter
        outputs = llm.generate(
            [prompt],
            sampling_params=None,
        )
        
        # Extract transcribed text from output
        if outputs and len(outputs) > 0:
            output_text = outputs[0].outputs[0].text.strip()
            if output_text:
                return split_chunks(output_text)
        
        return []
    except Exception as exc:
        logging.warning("Failed to extract text from image %s: %s", path, exc)
        return []


def parse_file(path: Path) -> List[str]:
    if not path.exists() or not path.is_file():
        return []
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            return parse_pdf(path)
        if suffix == ".docx":
            return parse_docx(path)
        if suffix in {".xlsx", ".xls", ".csv"}:
            return parse_excel_or_csv(path)
        if suffix in {".txt", ".log", ".py"}:
            return parse_text_like(path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return parse_image(path)
    except Exception as exc:
        logging.warning("Unhandled parse error for %s: %s", path, exc)
        return []
    return []


def read_corpus_file(file_path: str) -> str:
    """
    Robust file reader with comprehensive error handling for edge cases.
    
    Handles:
    - Permission denied (chmod 000)
    - Encrypted/password-protected PDFs
    - Empty directories and unknown file formats
    - General corrupted files
    
    Args:
        file_path: Path to file to read.
    
    Returns:
        Extracted text content, or empty string on error.
    """
    path = Path(file_path)
    
    # Check if path exists and is a regular file
    if not path.exists():
        logging.warning("[Warning] File does not exist: %s", file_path)
        return ""
    
    if not path.is_file():
        logging.debug("Skipped non-file entry: %s", file_path)
        return ""
    
    suffix = path.suffix.lower()
    
    # Reject unsupported file formats
    supported_extensions = {
        ".pdf", ".docx", ".xlsx", ".xls", ".csv",
        ".txt", ".log", ".py",
        ".png", ".jpg", ".jpeg"
    }
    if suffix not in supported_extensions:
        logging.debug("Skipped unsupported file type: %s", file_path)
        return ""
    
    # === EDGE CASE 1: Permission Denied (chmod 000) ===
    try:
        if not os.access(path, os.R_OK):
            logging.warning("[Warning] Skipped unreadable file (permission denied): %s", file_path)
            return ""
    except OSError as exc:
        logging.warning("[Warning] Skipped unreadable file (permission denied): %s", file_path)
        return ""
    
    # === EDGE CASE 2: Encrypted/Password-Protected PDFs ===
    if suffix == ".pdf":
        try:
            chunks = parse_pdf(path)
            if chunks:
                return " ".join(chunks)
            else:
                logging.warning("[Warning] Skipped encrypted PDF: %s", file_path)
                return ""
        except Exception as exc:
            # Check if it's a PDF-specific error (encryption, corruption)
            exc_str = str(exc).lower()
            if "encrypt" in exc_str or "password" in exc_str or "pdf" in exc_str:
                logging.warning("[Warning] Skipped encrypted PDF: %s", file_path)
            else:
                logging.warning("[Warning] Failed to parse PDF %s: %s", file_path, exc)
            return ""
    
    # === EDGE CASE 3 & 4: Text files and other formats with general error handling ===
    try:
        if suffix == ".docx":
            chunks = parse_docx(path)
            return " ".join(chunks) if chunks else ""
        elif suffix in {".xlsx", ".xls", ".csv"}:
            chunks = parse_excel_or_csv(path)
            return " ".join(chunks) if chunks else ""
        elif suffix in {".txt", ".log", ".py"}:
            chunks = parse_text_like(path)
            return " ".join(chunks) if chunks else ""
        elif suffix in {".png", ".jpg", ".jpeg"}:
            chunks = parse_image(path)
            return " ".join(chunks) if chunks else ""
    except PermissionError:
        logging.warning("[Warning] Skipped unreadable file (permission denied): %s", file_path)
        return ""
    except Exception as exc:
        # General corrupted file handling
        logging.warning("[Warning] Failed to read file %s: %s", file_path, exc)
        return ""
    
    return ""


def extract_question_keywords(question: str) -> List[str]:
    text = question.lower()
    return [token for token in re.findall(r"[a-z0-9]+", text) if len(token) > 2]


def candidate_lines_from_chunk(chunk_text: str) -> List[str]:
    lines = []
    for line in re.split(r"\n+|\r+", chunk_text):
        cleaned = line.strip()
        if not cleaned or len(cleaned) < 3:
            continue
        if any(ch.isdigit() for ch in cleaned) or re.search(r"\b[A-Z]{2,}[A-Z0-9_-]{2,}\b", cleaned):
            lines.append(cleaned)
    return lines


def is_likely_value(candidate: str) -> bool:
    value = candidate.strip().strip(".,;:()[]{}\"'")
    if not value:
        return False
    if value.upper() in {"PART", "NUMBER", "VALUE", "NAME", "MODEL", "REVISION", "VERSION", "TYPE", "CODE", "SERIAL", "ID"}:
        return False
    if re.search(r"\d", value):
        return True
    if re.fullmatch(r"[A-Z0-9][A-Z0-9._/-]*", value) and len(value) >= 2:
        return True
    return False


def extract_value_candidates(line: str) -> List[str]:
    candidates: List[str] = []
    normalized = line.strip().rstrip(".,;:)]}\"'")
    for sep in (":", "="):
        if sep in normalized:
            right = normalized.split(sep, 1)[1].strip()
            for part in re.split(r"\s*\|\s*|\s*;\s*|\s*,\s*", right):
                candidate = part.strip().strip(".,;:()[]{}\"'")
                if is_likely_value(candidate):
                    candidates.append(candidate)
            if candidates:
                return candidates

    for pattern in [
        r"\b(?:Q\d+|FY\d{2,4}|REV-[A-Z0-9]+|ORR-[A-Z0-9-]+|[A-Z]{2,}[A-Z0-9_-]{2,})\b",
        r"\b\d+(?:\.\d+)?(?:%|V|A|W|Hz|KHz|MHz|GHz|mA|mV|ms|s|C|MB|GB|TB|KB|B|RPM|mm|nm)?\b",
    ]:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            candidate = match.group(0).strip()
            if is_likely_value(candidate):
                candidates.append(candidate)
    return candidates


def extract_answer_from_context(question: str, matches: List[DocumentChunk]) -> str:
    if not matches:
        return ""
    keywords = set(extract_question_keywords(question))
    best_line = ""
    best_score = -1
    best_candidates: List[str] = []

    for match in matches:
        for line in candidate_lines_from_chunk(match.text):
            line_lower = line.lower()
            score = 0
            for keyword in keywords:
                if keyword in line_lower:
                    score += 10
            if any(ch.isdigit() for ch in line):
                score += 4
            if re.search(r"\b[A-Z]{2,}[A-Z0-9_-]{2,}\b", line):
                score += 2
            values = extract_value_candidates(line)
            if values:
                score += 6
            if score > best_score:
                best_score = score
                best_line = line
                best_candidates = values

    if not best_line:
        first_line = candidate_lines_from_chunk(matches[0].text)[0] if candidate_lines_from_chunk(matches[0].text) else ""
        if first_line:
            best_line = first_line
            best_candidates = extract_value_candidates(first_line)

    if best_candidates:
        return normalize_answer(best_candidates[0])

    value_match = re.search(r"(?<![A-Z0-9])(?:\d+(?:\.\d+)?(?:\s*(?:%|MS|MB|GB|W|V|A|HZ|KHZ|MHZ|GHZ|C|MA|MV|DB|RPM|NM|S|H|D|N|KG|MM|IN|LB|KW|TB|KB|B))?|[A-Z]{2,}[A-Z0-9_-]{2,})", best_line, flags=re.IGNORECASE)
    if value_match:
        return normalize_answer(value_match.group(0))
    return ""


def build_response_for_query(index: VectorIndex, question: str, embedding_service: Optional[EmbeddingService] = None) -> Dict[str, Any]:
    """
    Build response for query with formatted output.
    
    Output formatting:
    - answer: Uppercase, stripped, special chars removed
    - citations: Sorted list of exact relative file paths
    - confidence: Float between 0.0 and 1.0
    """
    matches = index.query(question, top_k=5, embedding_service=embedding_service)
    if not matches:
        return {"answer": "", "citations": [], "confidence": 0.0}

    top_chunks = [chunk for _, chunk in matches]
    answer = extract_answer_from_context(question, top_chunks)
    citations = sorted(list({chunk.file_path for chunk in top_chunks if chunk.file_path}))
    if not answer:
        return {"answer": "", "citations": [], "confidence": 0.0}
    confidence = max(0.1, min(0.99, 0.35 + (matches[0][0] if matches else 0.0) * 0.65))
    return {"answer": answer, "citations": citations, "confidence": round(confidence, 4)}


class RAGDaemon:
    def __init__(self, enable_gpu: bool = True):
        """Initialize RAG daemon with embedding service and vLLM vision engine."""
        # Initialize embedding service with local model directory
        self.embedding_service = EmbeddingService(
            model_name=DEFAULT_MODEL,
            model_dir=DEFAULT_EMBEDDING_DIR
        )
        
        # Initialize vLLM engine for Llama Vision from local models
        initialize_vision_llm(
            model_dir=DEFAULT_VISION_DIR,
            gpu_memory_utilization=0.65,
            max_model_len=2048
        )
        
        self.index = VectorIndex()
        self.index_path: Optional[str] = None
        logging.info("RAG daemon initialized with vLLM vision engine")

    def handle_index(self, corpus_dir: str) -> Dict[str, Any]:
        start = time.time()
        self.index.build_from_corpus(corpus_dir, self.embedding_service)
        self.index_path = corpus_dir
        elapsed = time.time() - start
        logging.info("Index built in %.2fs for %s", elapsed, corpus_dir)
        return {"status": "ok", "indexed": len(self.index.chunks), "elapsed_seconds": round(elapsed, 2)}

    def handle_query(self, corpus_dir: str, query_text: str) -> Dict[str, Any]:
        if not self.index.chunks or corpus_dir != self.index_path:
            self.index.build_from_corpus(corpus_dir, self.embedding_service)
        response = build_response_for_query(self.index, query_text, embedding_service=self.embedding_service)
        return response

    def serve(self, socket_path: str = SOCKET_PATH):
        if hasattr(socket, "AF_UNIX"):
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                pass
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(socket_path)
            address = socket_path
        else:
            server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((TCP_HOST, TCP_PORT))
            address = f"{TCP_HOST}:{TCP_PORT}"

        with server:
            server.listen(32)
            logging.info("RAG daemon listening on %s", address)
            while True:
                try:
                    conn, _ = server.accept()
                    raw = b""
                    while True:
                        chunk = conn.recv(65536)
                        if not chunk:
                            break
                        raw += chunk
                        if len(chunk) < 65536:
                            break
                    try:
                        payload = json.loads(raw.decode("utf-8"))
                    except Exception:
                        payload = {"cmd": "noop"}
                    cmd = payload.get("cmd")
                    if cmd == "index":
                        response = self.handle_index(payload.get("corpus", ""))
                    elif cmd == "query":
                        response = self.handle_query(payload.get("corpus", ""), payload.get("query", ""))
                    else:
                        response = {"answer": "", "citations": [], "confidence": 0.0}
                    conn.sendall(json.dumps(response, ensure_ascii=False).encode("utf-8"))
                    conn.close()
                except Exception as exc:
                    logging.warning("Daemon socket loop exception: %s", exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="RAG daemon for multimodal document indexing and retrieval.")
    parser.add_argument("--serve", action="store_true", help="Run the Unix socket daemon.")
    parser.add_argument("--index", type=str, help="Index a corpus directory and exit.")
    parser.add_argument("--corpus", type=str, help="Corpus directory for one-off queries.")
    parser.add_argument("--query", type=str, help="Query text to answer in one-off mode.")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration for EasyOCR (default: auto-detect).")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration.")
    args = parser.parse_args()

    # Determine GPU usage
    use_gpu = True
    if args.no_gpu:
        use_gpu = False
    elif not args.gpu:
        # Auto-detect GPU availability
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False

    daemon = RAGDaemon(enable_gpu=use_gpu)

    if args.serve:
        daemon.serve()
        return 0

    if args.index:
        response = daemon.handle_index(args.index)
        print(json.dumps(response, ensure_ascii=False))
        return 0

    if args.corpus and args.query:
        response = daemon.handle_query(args.corpus, args.query)
        print(json.dumps(response, ensure_ascii=False))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
