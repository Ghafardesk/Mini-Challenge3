import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import daemon


class DefensiveReaderTests(unittest.TestCase):
    def test_unknown_and_unreadable_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            unknown_file = Path(directory, "capture.dat")
            unknown_file.write_text("not indexed", encoding="utf-8")
            unreadable_file = Path(directory, "private.txt")
            unreadable_file.write_text("secret", encoding="utf-8")
            unreadable_file.chmod(0)

            self.assertEqual(daemon.read_corpus_file(str(unknown_file)), "")
            self.assertEqual(daemon.read_corpus_file(str(unreadable_file)), "")

    def test_empty_directory_has_no_files(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(daemon.VectorIndex._discover_files(Path(directory)), [])


class VisionInitializationTests(unittest.TestCase):
    def test_vllm_uses_visible_cuda_device(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text("{}", encoding="utf-8")
            llm_module = types.ModuleType("vllm")
            llm_module.LLM = unittest.mock.Mock(return_value="engine")
            transformers_module = types.ModuleType("transformers")
            transformers_module.AutoTokenizer = types.SimpleNamespace(
                from_pretrained=unittest.mock.Mock(return_value="tokenizer")
            )
            with patch.dict(
                sys.modules,
                {"vllm": llm_module, "transformers": transformers_module},
            ):
                result = daemon.initialize_vision_llm(directory)

        self.assertEqual(result, "engine")
        self.assertEqual(daemon.get_vision_tokenizer(), "tokenizer")
        self.assertEqual(daemon.os.environ["CUDA_VISIBLE_DEVICES"], "0")
        llm_module.LLM.assert_called_once_with(
            model=directory,
            device="cuda",
            tensor_parallel_size=1,
            max_model_len=4096,
            max_num_seqs=16,
            enforce_eager=True,
            trust_remote_code=True,
        )


class PathAndAnswerTests(unittest.TestCase):
    def test_clean_relative_path_is_corpus_relative(self):
        with tempfile.TemporaryDirectory() as directory:
            corpus = Path(directory) / "mc3-starter-kit" / "mc3-starter-kit" / "corpus"
            image = corpus / "specs" / "backplane_pinout.png"
            image.parent.mkdir(parents=True)
            image.touch()
            self.assertEqual(
                daemon.get_clean_relative_path(image, corpus),
                "specs/backplane_pinout.png",
            )

    def test_image_lookup_returns_signal_and_image_citation(self):
        image = daemon.DocumentChunk(
            "specs/backplane_pinout.png",
            "B11 GND B12 SDA B13 SCL B14 THERM_ALERT# B15 PRSNT# B16 GND",
        )
        decoy = daemon.DocumentChunk("planning/roadmap_fy27.docx", "Customer sampling: Q3 FY27")
        index = daemon.VectorIndex()
        index.chunks = [decoy, image]
        index.embeddings = [[1.0], [1.0]]
        embedding_service = types.SimpleNamespace(embed=lambda texts: [[1.0] for _ in texts])

        response = daemon.build_response_for_query(
            index,
            "What is the signal description for Pin 4 on the backplane pinout?",
            embedding_service,
        )

        self.assertEqual(
            response,
            {
                "answer": "THERMALERT#",
                "citations": ["specs/backplane_pinout.png"],
                "confidence": 0.99,
            },
        )
        self.assertEqual(set(response), {"answer", "citations", "confidence"})
        json.dumps(response)

    def test_signal_to_pin_lookup_preserves_image_source(self):
        chunk = daemon.DocumentChunk(
            "specs/backplane_pinout.png",
            "B11 B12 B13 B14 B15 B16\nGND SDA SCL THERM_ALERT# PRSNT# GND",
        )
        answer, source = daemon._extract_answer_with_source(
            "Which backplane pin carries THERM_ALERT# on the TQ-40?",
            [chunk],
        )
        self.assertEqual(answer, "B14")
        self.assertIs(source, chunk)

    def test_image_question_refuses_when_image_ocr_is_unavailable(self):
        chunk = daemon.DocumentChunk("planning/roadmap_fy27.docx", "Customer sampling: Q3 FY27")
        index = daemon.VectorIndex()
        index.chunks = [chunk]
        index.embeddings = [[1.0]]
        embedding_service = types.SimpleNamespace(embed=lambda texts: [[1.0] for _ in texts])

        response = daemon.build_response_for_query(
            index,
            "What is the signal description for Pin 4 on the backplane pinout?",
            embedding_service,
        )
        self.assertEqual(response, {"answer": "", "citations": [], "confidence": 0.0})

    def test_normalize_answer_removes_requested_separators(self):
        self.assertEqual(daemon.normalize_answer(" therm_alert# "), "THERMALERT#")

    def test_output_writer_uses_query_id_and_contract_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            output_file = Path(directory) / "query_vision_01_output.json"
            with (
                patch.object(app, "DEFAULT_OUTPUT", Path(directory)),
                patch.object(
                    app,
                    "send_request",
                    return_value={
                        "answer": "THERMALERT#",
                        "citations": ["specs/backplane_pinout.png"],
                        "confidence": 0.9,
                    },
                ) as send_request,
                patch.object(
                    sys,
                    "argv",
                    [
                        "app.py",
                        "--corpus",
                        "./mc3-starter-kit/corpus",
                        "--query-id",
                        "query_vision_01",
                        "--query",
                        "What is the signal description for Pin 4 on the backplane pinout?",
                    ],
                ),
            ):
                self.assertEqual(app.main(), 0)
            self.assertEqual(output_file.name, "query_vision_01_output.json")
            self.assertEqual(
                set(json.loads(output_file.read_text(encoding="utf-8"))),
                {"answer", "citations", "confidence"},
            )
            self.assertEqual(
                send_request.call_args.args[0]["corpus"],
                daemon.resolve_corpus_path("./mc3-starter-kit/corpus"),
            )


if __name__ == "__main__":
    unittest.main()
