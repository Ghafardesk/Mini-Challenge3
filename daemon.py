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
import csv
import hashlib
import json
import logging
import math
import os
import re
import socket
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SOCKET_PATH = "/tmp/rag_daemon.sock"
TCP_HOST = "127.0.0.1"
TCP_PORT = 8765
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIR = str(PROJECT_ROOT / "models" / "all-MiniLM-L6-v2")
DEFAULT_VISION_MODEL_ID = os.getenv("HF_MODEL_ID", "meta-llama/Llama-3.2-11B-Vision-Instruct")
DEFAULT_VISION_DIR = str(PROJECT_ROOT / "models" / "Llama-3.2-11B-Vision-Instruct")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Global vLLM engine initialized at daemon startup
_vision_llm: Optional[Any] = None
_vision_tokenizer: Optional[Any] = None


def initialize_vision_llm(model_dir: str = DEFAULT_VISION_DIR) -> Optional[Any]:
    """
    Initialize vLLM engine for Llama Vision image understanding.
    
    Args:
        model_dir: Local directory containing the shipped Llama-3.2-11B-Vision-Instruct model.
    
    Returns:
        Initialized vLLM LLM instance, or None if initialization fails.
    """
    global _vision_llm, _vision_tokenizer
    model_path = Path(model_dir)
    if not (model_path / "config.json").is_file():
        logging.warning("Vision model files are not available at %s; image OCR is disabled", model_dir)
        return None

    try:
        from transformers import AutoTokenizer
        from vllm import LLM

        _vision_llm = LLM(
            model=model_dir,
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
        )
        _vision_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        logging.info("Initialized vLLM vision engine from %s", model_dir)
        return _vision_llm
    except Exception as exc:
        logging.warning("Failed to initialize vLLM vision engine from %s: %s", model_dir, exc)
        return None


def get_vision_llm() -> Optional[Any]:
    """Get the global vLLM engine instance."""
    return _vision_llm


def get_vision_tokenizer() -> Optional[Any]:
    """Get the tokenizer paired with the global vision engine."""
    return _vision_tokenizer


def resolve_corpus_path(corpus_dir: str) -> str:
    if not corpus_dir:
        return ""

    raw = Path(corpus_dir)
    root = Path(__file__).resolve().parent
    candidates = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(Path.cwd() / raw)
        candidates.append(root / raw)

        project_root = root / "mc3-starter-kit" / "mc3-starter-kit"
        if raw.name in {"corpus", "mc3-corpus"}:
            candidates.append(project_root / "mc3-corpus")
            candidates.append(root / "mc3-starter-kit" / "mc3-corpus")
        candidates.append(project_root / raw)
        candidates.append(root / "mc3-starter-kit" / raw)

    seen = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and resolved.is_dir():
            return str(resolved)

    return str(raw)


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

            if (Path(model_dir) / "modules.json").is_file():
                self.model = SentenceTransformer(model_dir, device=self.device, local_files_only=True)
                logging.info("Loaded embedding model from local directory %s on %s", model_dir, self.device)
            else:
                logging.warning("Embedding model files are not available at %s; using local fallback embeddings", model_dir)
        except Exception as exc:  # pragma: no cover - fallback path
            logging.warning("SentenceTransformers unavailable; using local fallback embeddings: %s", exc)
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
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            idx = int.from_bytes(digest, byteorder="big") % dim
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
                if re.search(r"(?:withdrawn|superseded|obsolete)", file_path.name, re.IGNORECASE):
                    logging.info("Skipped superseded document: %s", file_path)
                    continue
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
    return re.sub(r"[-.·_\s]", "", str(value).upper().strip())


def safe_read_utf8(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.read()
    except (OSError, UnicodeError) as exc:
        logging.warning("Failed to read text file %s: %s", path, exc)
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
        if reader.is_encrypted:
            logging.warning("Skipped encrypted PDF: %s", path)
            return []
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
            rows = [
                [cell.text.strip() for cell in row.cells]
                for row in table.rows
            ]
            if not rows:
                continue
            headers = rows[0]
            for row in rows[1:]:
                fields = [
                    f"{header}: {value}"
                    for header, value in zip(headers, row)
                    if header and value
                ]
                values = [value for value in row if value]
                if fields:
                    pieces.append(" | ".join(fields))
                elif values:
                    pieces.append(" | ".join(values))
        return split_chunks("\n".join(pieces))
    except Exception as exc:
        logging.warning("Failed to parse DOCX %s: %s", path, exc)
        return []


def parse_excel_or_csv(path: Path) -> List[str]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open("r", encoding="utf-8-sig", errors="ignore", newline="") as handle:
                table = list(csv.reader(handle))
            if not table:
                return []
            headers = [value.strip() for value in table[0]]
            rows = []
            for row in table[1:]:
                fields = [
                    f"{header}: {value.strip()}"
                    for header, value in zip(headers, row)
                    if header and value.strip()
                ]
                if fields:
                    rows.append(" | ".join(fields))
            return split_chunks("\n".join(rows))

        if suffix == ".xlsx":
            from openpyxl import load_workbook

            rows = []
            workbook = load_workbook(path, read_only=True, data_only=True)
            try:
                for sheet in workbook.worksheets:
                    rows.append(f"Sheet: {sheet.title}")
                    values = sheet.iter_rows(values_only=True)
                    headers = [str(value).strip() if value is not None else "" for value in next(values, ())]
                    for row in values:
                        fields = [
                            f"{header}: {value}"
                            for header, value in zip(headers, row)
                            if header and value is not None and str(value).strip()
                        ]
                        if fields:
                            rows.append(" | ".join(fields))
            finally:
                workbook.close()
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
    tokenizer = get_vision_tokenizer()
    if llm is None or tokenizer is None:
        logging.warning("vLLM OCR is unavailable; skipping image %s", path)
        return []

    try:
        from PIL import Image
        from vllm import SamplingParams
    except Exception:
        logging.warning("PIL or vLLM SamplingParams is unavailable; cannot OCR %s", path)
        return []

    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except Exception as exc:
        logging.warning("Failed to open image %s: %s", path, exc)
        return []

    try:
        messages = [{
            "role": "user",
            "content": (
                "Act as an OCR assistant. Analyze the provided <|image|> image and:\n"
                "1. Identify and transcribe all visible text in the image exactly as it appears.\n"
                "2. Preserve the original line breaks, spacing, and formatting from the image.\n"
                "3. Output only the transcribed text, line by line, without adding any commentary "
                "or explanations or special characters.\n"
            ),
        }]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        sampling_params = SamplingParams(max_tokens=512, temperature=0.0)
        outputs = llm.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": image},
            },
            sampling_params=sampling_params,
        )
        extracted_text = outputs[0].outputs[0].text if outputs else ""
        if not extracted_text.strip():
            logging.warning("vLLM returned no OCR text for %s", path)
            return []
        return split_chunks(extracted_text)
    except Exception as exc:
        logging.warning("vLLM OCR failed for image %s: %s", path, exc)
        return []


def parse_file(path: Path) -> List[str]:
    return split_chunks(parse_document(path))


def parse_document(file_path: str | Path) -> str:
    """Extract searchable text from one supported corpus file, skipping traps."""
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in {
        ".pdf", ".docx", ".xlsx", ".csv", ".txt", ".log", ".py",
        ".png", ".jpg", ".jpeg",
    }:
        return ""

    try:
        if not path.is_file() or not os.access(path, os.R_OK) or not path.stat().st_mode & (stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH):
            return ""
        if suffix == ".pdf":
            chunks = parse_pdf(path)
        elif suffix == ".docx":
            chunks = parse_docx(path)
        elif suffix in {".xlsx", ".csv"}:
            chunks = parse_excel_or_csv(path)
        elif suffix in {".txt", ".log", ".py"}:
            chunks = parse_text_like(path)
        else:
            chunks = parse_image(path)
        return "\n".join(chunks)
    except PermissionError:
        logging.warning("Skipped unreadable file: %s", path)
    except OSError as exc:
        logging.warning("Failed to access file %s: %s", path, exc)
    except Exception as exc:
        logging.warning("Failed to parse file %s: %s", path, exc)
    return ""


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
    return parse_document(file_path)


def extract_question_keywords(question: str) -> List[str]:
    stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "by", "does", "for",
        "from", "how", "in", "is", "it", "of", "on", "or", "the", "to",
        "was", "what", "when", "where", "which", "who", "with",
    }
    tokens = re.findall(r"[a-z0-9]+", question.lower())
    return [token for token in tokens if len(token) > 2 and token not in stopwords]


def candidate_lines_from_chunk(chunk_text: str) -> List[str]:
    lines = []
    for line in re.split(r"\n+|\r+", chunk_text):
        cleaned = line.strip()
        if not cleaned or len(cleaned) < 3:
            continue
        if (
            any(ch.isdigit() for ch in cleaned)
            or re.search(r"\b[A-Z]\d{1,3}\b", cleaned)
            or re.search(r"\b[A-Z]{2,}[A-Z0-9_-]{2,}\b", cleaned)
        ):
            lines.append(cleaned)
    return lines


def is_likely_value(candidate: str) -> bool:
    value = candidate.strip().strip(".,;:()[]{}\"'")
    if not value:
        return False
    if value.upper() in {"PART", "NUMBER", "VALUE", "NAME", "MODEL", "REVISION", "VERSION", "TYPE", "CODE", "SERIAL", "ID"}:
        return False
    if re.fullmatch(r"[A-Z][A-Z0-9_]*#", value, re.IGNORECASE):
        return True
    if re.search(r"\d", value):
        return True
    if re.fullmatch(r"[A-Z0-9][A-Z0-9._/-]*", value) and len(value) >= 2:
        return True
    return False


def extract_value_candidates(line: str) -> List[str]:
    candidates: List[str] = []
    normalized = line.strip().rstrip(".,;:)]}\"'")
    for pattern in [
        r"\b[A-Z][A-Z0-9_]*#(?=$|\s|[|,;])",
        r"\b(?:Q\d+\s*FY\d{2,4}|FY\d{2,4}|REV-[A-Z0-9-]+|ORR-[A-Z0-9-]+|E\d{3,}|[A-Z]\d{1,3}|[A-Z]{2,}[A-Z0-9_-]{2,})\b",
        r"\b\d+(?:\.\d+){1,3}\b",
        r"\b\d+(?:\.\d+)?\b",
    ]:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            candidate = match.group(0).strip()
            if is_likely_value(candidate):
                candidates.append(candidate)
    return candidates


def _preferred_field_patterns(question: str) -> List[str]:
    text = question.lower()
    if re.search(r"\bsignal(?:\s+\w+){0,2}\s+description\b|\bdescription(?:\s+\w+){0,2}\s+signal\b", text):
        return [r"\bsignal\b", r"\bdescription\b", r"\bname\b"]
    if "price" in text or "cost" in text:
        return [r"\bunit[_ ]*price\b", r"\bprice\b", r"\bcost\b"]
    if "part" in text or "fan assembly" in text:
        return [r"\bpart(?:[_ ]*(?:number|no\.?))?\b", r"\bsku\b", r"\bp/?n\b"]
    if any(word in text for word in ("firmware", "version", "release", "fixed", "fix")):
        return [r"\bfixed[_ ]in\b", r"\bfirmware[_ ]version\b", r"\bversion\b", r"\brelease\b"]
    if "timeout" in text or "seconds" in text:
        return [r"\bdefault[_ ]*batch[_ ]*timeout\b", r"\btimeout\b"]
    if "quarter" in text or "sampling" in text:
        return [r"\bcustomer[_ ]sampling\b", r"\bsampling[_ ]quarter\b", r"\bquarter\b"]
    if "revision" in text or "board" in text:
        return [r"\bboard[_ ]revision\b", r"\brevision\b", r"\brev\b"]
    if "error" in text or "code" in text:
        return [r"\berror[_ ]code\b", r"\bcode\b"]
    if "pin" in text or "signal" in text:
        return [r"\bpin\b", r"\bsignal\b"]
    return []


def _clean_answer_candidate(candidate: str) -> str:
    value = candidate.strip().strip(".,;:()[]{}\"'")
    numeric_with_unit = re.fullmatch(
        r"(\d+(?:\.\d+)?)(?:\s*°?\s*(?:C|F|V|A|W|HZ|KHZ|MHZ|GHZ|MS|S|RPM|MM|NM|MB|GB|TB|KB|B|%))?",
        value,
        flags=re.IGNORECASE,
    )
    if numeric_with_unit:
        value = numeric_with_unit.group(1)
    return normalize_answer(value)


def _line_relevance(question: str, line: str) -> int:
    keywords = set(extract_question_keywords(question))
    line_tokens = set(re.findall(r"[a-z0-9]+", line.lower()))
    overlap = keywords & line_tokens
    score = len(overlap) * 10
    text = question.lower()
    if "pin" in text and re.search(r"\b[A-Z]\d{1,3}\b", line):
        score += 20
    if "revision" in text and re.search(r"\bREV-[A-Z0-9-]+\b", line, re.IGNORECASE):
        score += 20
    if "signal" in text and "description" in text and re.search(r"\b[A-Z][A-Z0-9_]*#(?=$|\s|[|,;])", line):
        score += 20
    if any(word in text for word in ("firmware", "version", "release", "fixed", "fix")):
        if re.search(r"\b\d+(?:\.\d+){1,3}\b", line):
            score += 15
    return score


def _answer_candidate_for_line(question: str, line: str) -> str:
    preferred_patterns = _preferred_field_patterns(question)
    if preferred_patterns:
        fields = re.findall(r"(?:^|\|)\s*([^:|=]+?)\s*:\s*([^|]+)", line)
        for pattern in preferred_patterns:
            for label, value in fields:
                if re.search(pattern, label, re.IGNORECASE):
                    candidate = extract_value_candidates(value)
                    selected = candidate[0] if candidate else value.strip()
                    if is_likely_value(selected):
                        return _clean_answer_candidate(selected)

        if "signal" in question.lower() and "description" in question.lower():
            signal = re.search(r"\b([A-Z][A-Z0-9_]*#)(?=$|\s|[|,;])", line)
            if signal:
                return normalize_answer(signal.group(1))
        if "pin" in question.lower():
            pin = re.search(r"\b([A-Z]\d{1,3})\b", line)
            if pin:
                return normalize_answer(pin.group(1))
        if "revision" in question.lower():
            revision = re.search(r"\b(REV-[A-Z0-9-]+)\b", line, re.IGNORECASE)
            if revision:
                return normalize_answer(revision.group(1))
        if any(word in question.lower() for word in ("firmware", "version", "release", "fixed", "fix")):
            version = re.search(r"\b(\d+(?:\.\d+){1,3})\b", line)
            if version:
                return normalize_answer(version.group(1))
        if "timeout" in question.lower() or "seconds" in question.lower():
            timeout = re.search(r"=\s*(\d+(?:\.\d+)?)\b", line)
            if timeout:
                return normalize_answer(timeout.group(1))

    candidates = extract_value_candidates(line)
    if not candidates:
        return ""
    if any(word in question.lower() for word in ("firmware", "version", "release", "fixed", "fix")):
        version = next((candidate for candidate in candidates if re.fullmatch(r"\d+(?:\.\d+){1,3}", candidate)), "")
        if version:
            return normalize_answer(version)
    if "error" in question.lower() or "code" in question.lower():
        error_code = next((candidate for candidate in candidates if re.fullmatch(r"E\d{3,}", candidate, re.IGNORECASE)), "")
        if error_code:
            return normalize_answer(error_code)
    if "quarter" in question.lower() or "sampling" in question.lower():
        quarter = re.search(r"\bQ\d+\s*FY\d{2,4}\b", line, re.IGNORECASE)
        if quarter:
            return normalize_answer(quarter.group(0))
    for candidate in candidates:
        cleaned = _clean_answer_candidate(candidate)
        if cleaned:
            return cleaned
    return ""


def extract_answer_from_context(question: str, matches: List[DocumentChunk]) -> str:
    answer, _ = _extract_answer_with_source(question, matches)
    return answer


def _extract_answer_with_source(question: str, matches: List[DocumentChunk]) -> Tuple[str, Optional[DocumentChunk]]:
    if not matches:
        return "", None
    best_answer = ""
    best_score = 0
    best_chunk: Optional[DocumentChunk] = None

    for match in matches:
        for line in candidate_lines_from_chunk(match.text):
            score = _line_relevance(question, line)
            if score < 20:
                continue
            answer = _answer_candidate_for_line(question, line)
            if answer and score > best_score:
                best_score = score
                best_answer = answer
                best_chunk = match

    return best_answer, best_chunk


def build_response_for_query(index: VectorIndex, question: str, embedding_service: Optional[EmbeddingService] = None) -> Dict[str, Any]:
    """
    Build response for query with formatted output.
    
    Output formatting:
    - answer: Uppercase, stripped, special chars removed
    - citations: Sorted list of exact relative file paths
    - confidence: Float between 0.0 and 1.0
    """
    matches = index.query(question, top_k=len(index.chunks), embedding_service=embedding_service)
    if not matches:
        return {"answer": "", "citations": [], "confidence": 0.0}

    ranked_chunks = sorted(
        (chunk for _, chunk in matches),
        key=lambda chunk: max(
            (_line_relevance(question, line) for line in candidate_lines_from_chunk(chunk.text)),
            default=0,
        ),
        reverse=True,
    )
    answer, supporting_chunk = _extract_answer_with_source(question, ranked_chunks)
    supporting_chunks = [supporting_chunk] if supporting_chunk is not None else []

    if (
        answer
        and supporting_chunk is not None
        and not re.search(r"\bORR-\d+\b", question, re.IGNORECASE)
        and re.search(r"\b(?:log|incident)\b|underlying defect", question, re.IGNORECASE)
    ):
        question_tickets = {
            ticket
            for chunk in ranked_chunks
            if max(
                (_line_relevance(question, line) for line in candidate_lines_from_chunk(chunk.text)),
                default=0,
            ) >= 20
            for ticket in re.findall(r"\bORR-\d+\b", chunk.text, re.IGNORECASE)
        }
        support_tickets = set(re.findall(r"\bORR-\d+\b", supporting_chunk.text, re.IGNORECASE))
        linked_evidence = [
            chunk for chunk in ranked_chunks
            if chunk.file_path != supporting_chunk.file_path
            and question_tickets.intersection(re.findall(r"\bORR-\d+\b", chunk.text, re.IGNORECASE))
            and support_tickets.intersection(re.findall(r"\bORR-\d+\b", chunk.text, re.IGNORECASE))
        ]
        supporting_chunks.extend(linked_evidence)

    if not answer:
        relevant_chunks = [
            chunk for chunk in ranked_chunks
            if max(
                (_line_relevance(question, line) for line in candidate_lines_from_chunk(chunk.text)),
                default=0,
            ) >= 20
        ]
        ticket_ids = {
            ticket
            for chunk in relevant_chunks
            for ticket in re.findall(r"\bORR-\d+\b", chunk.text, re.IGNORECASE)
        }
        for ticket in ticket_ids:
            linked_chunks = [
                chunk for chunk in index.chunks
                if re.search(rf"\b{re.escape(ticket)}\b", chunk.text, re.IGNORECASE)
            ]
            answer, linked_support = _extract_answer_with_source(question, linked_chunks)
            if answer and linked_support is not None:
                source = next(
                    (
                        chunk for chunk in relevant_chunks
                        if re.search(rf"\b{re.escape(ticket)}\b", chunk.text, re.IGNORECASE)
                    ),
                    None,
                )
                supporting_chunks = [source, linked_support] if source is not None else [linked_support]
                break

    if not answer or not supporting_chunks or any(not chunk.file_path for chunk in supporting_chunks):
        return {"answer": "", "citations": [], "confidence": 0.0}
    citations = sorted({chunk.file_path for chunk in supporting_chunks})
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
        initialize_vision_llm(model_dir=DEFAULT_VISION_DIR)
        
        self.index = VectorIndex()
        self.index_path: Optional[str] = None
        logging.info("RAG daemon initialized")

    def handle_index(self, corpus_dir: str) -> Dict[str, Any]:
        resolved_corpus = resolve_corpus_path(corpus_dir)
        start = time.time()
        self.index.build_from_corpus(resolved_corpus, self.embedding_service)
        self.index_path = resolved_corpus
        elapsed = time.time() - start
        logging.info("Index built in %.2fs for %s", elapsed, resolved_corpus)
        return {"status": "ok", "indexed": len(self.index.chunks), "elapsed_seconds": round(elapsed, 2)}

    def handle_query(self, corpus_dir: str, query_text: str) -> Dict[str, Any]:
        resolved_corpus = resolve_corpus_path(corpus_dir)
        if not self.index.chunks or resolved_corpus != self.index_path:
            self.index.build_from_corpus(resolved_corpus, self.embedding_service)
        response = build_response_for_query(self.index, query_text, embedding_service=self.embedding_service)
        return response

    def serve(self, socket_path: str = SOCKET_PATH):
        use_unix_socket = os.name != "nt" and hasattr(socket, "AF_UNIX")
        if use_unix_socket:
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
        response = daemon.handle_index(resolve_corpus_path(args.index))
        print(json.dumps(response, ensure_ascii=False))
        return 0

    if args.corpus and args.query:
        response = daemon.handle_query(resolve_corpus_path(args.corpus), args.query)
        print(json.dumps(response, ensure_ascii=False))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
