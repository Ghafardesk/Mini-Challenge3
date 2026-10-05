#!/usr/bin/env python3
import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

SOCKET_PATH = "/tmp/rag_daemon.sock"
TCP_HOST = "127.0.0.1"
TCP_PORT = 8765
DEFAULT_OUTPUT = Path("/app/output")


def get_client_socket():
    if hasattr(socket, "AF_UNIX"):
        return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM), SOCKET_PATH
    return socket.socket(socket.AF_INET, socket.SOCK_STREAM), (TCP_HOST, TCP_PORT)


def _normalize_answer(value: Any) -> str:
    if value is None:
        return ""
    answer = str(value).strip().upper()
    for ch in "-._·":
        answer = answer.replace(ch, "")
    return answer.strip()


def _normalize_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    answer = _normalize_answer(payload.get("answer", ""))
    citations = payload.get("citations", []) or []
    if not isinstance(citations, list):
        citations = [str(citations)]
    confidence = float(payload.get("confidence", 0.0) or 0.0)
    if confidence < 0:
        confidence = 0.0
    if confidence > 1:
        confidence = 1.0
    return {
        "answer": answer,
        "citations": citations,
        "confidence": round(confidence, 4),
    }


def ensure_daemon() -> None:
    if hasattr(socket, "AF_UNIX") and os.path.exists(SOCKET_PATH):
        return
    if not hasattr(socket, "AF_UNIX") and _tcp_socket_ready():
        return
    daemon_path = Path("/app/daemon.py")
    if not daemon_path.exists():
        return
    subprocess.Popen([sys.executable, str(daemon_path), "--serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(100):
        if hasattr(socket, "AF_UNIX") and os.path.exists(SOCKET_PATH):
            return
        if not hasattr(socket, "AF_UNIX") and _tcp_socket_ready():
            return
        time.sleep(0.1)


def _tcp_socket_ready() -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            sock.connect((TCP_HOST, TCP_PORT))
            return True
    except Exception:
        return False


def send_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    ensure_daemon()
    if hasattr(socket, "AF_UNIX"):
        if not os.path.exists(SOCKET_PATH):
            return {"answer": "", "citations": [], "confidence": 0.0}
    elif not _tcp_socket_ready():
        return {"answer": "", "citations": [], "confidence": 0.0}

    response_data = b""
    sock, address = get_client_socket()
    with sock:
        sock.settimeout(30)
        try:
            sock.connect(address)
            sock.sendall(json.dumps(payload).encode("utf-8"))
            try:
                sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            while True:
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    break
                if not chunk:
                    break
                response_data += chunk
        except Exception:
            return {"answer": "", "citations": [], "confidence": 0.0}

    if not response_data:
        return {"answer": "", "citations": [], "confidence": 0.0}
    try:
        decoded = response_data.decode("utf-8")
        data = json.loads(decoded)
    except Exception:
        return {"answer": "", "citations": [], "confidence": 0.0}
    if payload.get("cmd") == "index":
        return data
    return _normalize_response(data)


def write_output(query_id: str, payload: Dict[str, Any]) -> Path:
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    output_file = DEFAULT_OUTPUT / f"{query_id}_output.json"
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin client for the AMD x Lablab multimodal RAG daemon.")
    parser.add_argument("--index", type=str, help="Path to the corpus to index.")
    parser.add_argument("--corpus", type=str, help="Path to the corpus to query.")
    parser.add_argument("--query-id", type=str, help="Unique output identifier.")
    parser.add_argument("--query", type=str, help="Question text to answer.")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.index:
        result = send_request({"cmd": "index", "corpus": args.index})
        print(json.dumps(result, ensure_ascii=False))
        return 0

    if args.corpus and args.query_id and args.query:
        payload = {
            "cmd": "query",
            "corpus": args.corpus,
            "query_id": args.query_id,
            "query": args.query,
        }
        result = send_request(payload)
        output_file = write_output(args.query_id, result)
        print(str(output_file))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
