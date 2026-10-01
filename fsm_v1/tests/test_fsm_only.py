from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from typing import Any, Iterable, Mapping

from fsm_only.compiler import compile_jsonl
from fsm_only.inventory import Inventory
from fsm_only.normalize import normalize_phrase
from fsm_only.runtime import InventorySession
from scripts.build_fsm_dataset import build
from scripts.evaluate_fsm import evaluate


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "fsm_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:  # pragma: no cover - assertion gives better context
            raise AssertionError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise AssertionError(f"{path}:{line_number}: JSONL record is not an object")
        records.append(value)
    return records


def record_kind(record: Mapping[str, Any]) -> str:
    return str(record.get("type", record.get("kind", record.get("record_type", "")))).strip().casefold()


def nested_values(value: Any, keys: set[str]) -> Iterable[Any]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in keys:
                yield child
            yield from nested_values(child, keys)
    elif isinstance(value, list):
        for child in value:
            yield from nested_values(child, keys)


def query_texts(records: Iterable[Mapping[str, Any]]) -> set[str]:
    texts: set[str] = set()
    for record in records:
        kind = record_kind(record)
        if kind not in {"query", "query_pattern", "utterance", "heldout", "test_case", "example"}:
            continue
        for key in ("text", "utterance", "query", "user_text"):
            value = record.get(key)
            if isinstance(value, str) and normalize_phrase(value):
                texts.add(normalize_phrase(value))
    return texts


def participant_ids(record: Mapping[str, Any]) -> set[str]:
    values = nested_values(record, {"participant_id", "participant", "subject_id"})
    return {str(value).strip() for value in values if value is not None and str(value).strip()}


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_phrase(text).encode("utf-8")).hexdigest()


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not DATA.is_dir():
            raise AssertionError(f"expected generated dataset directory: {DATA}")
        cls.split_paths = {name: DATA / f"{name}.jsonl" for name in ("train", "validation", "test")}
        for name, path in cls.split_paths.items():
            if not path.is_file():
                raise AssertionError(f"missing {name} split: {path}")
        cls.split_records = {name: read_jsonl(path) for name, path in cls.split_paths.items()}
        heldout_candidates = [
            DATA / "heldout.jsonl",
            DATA / "heldout_examples.jsonl",
            DATA / "examples_heldout.jsonl",
        ]
        cls.heldout_path = next((path for path in heldout_candidates if path.is_file()), None)
        if cls.heldout_path is None:
            matches = sorted(DATA.glob("*held*out*.jsonl"))
            cls.heldout_path = matches[0] if len(matches) == 1 else None
        if cls.heldout_path is None:
            raise AssertionError("missing a distinct held-out JSONL file")
        cls.heldout_records = read_jsonl(cls.heldout_path)

    def test_jsonl_schema_and_nonempty_splits(self) -> None:
        allowed = {
            "config",
            "meta",
            "query",
            "query_pattern",
            "utterance",
            "transition",
            "controller",
            "controller_transition",
        }
        for split, records in self.split_records.items():
            self.assertTrue(records, f"{split} split is empty")
            for index, record in enumerate(records, 1):
                kind = record_kind(record)
                self.assertIn(kind, allowed, f"{split}:{index}: unknown record type {kind!r}")
                declared_split = record.get("split")
                if declared_split is not None:
                    self.assertEqual(declared_split, split, f"{split}:{index}: split label disagrees with file")
                if kind in {"query", "query_pattern", "utterance"}:
                    self.assertTrue(str(record.get("intent", "")).strip(), f"{split}:{index}: missing intent")
                    has_pattern = any(key in record for key in ("pattern", "symbols", "token_labels"))
                    has_text_slots = isinstance(record.get("text", record.get("utterance")), str) and isinstance(
                        record.get("slots", {}), Mapping
                    )
                    self.assertTrue(has_pattern or has_text_slots, f"{split}:{index}: query has no labeled pattern")
                elif kind in {"transition", "controller", "controller_transition"}:
                    for field in ("state", "event", "action", "next_state"):
                        self.assertTrue(str(record.get(field, "")).strip(), f"{split}:{index}: missing {field}")

        self.assertTrue(self.heldout_records, "held-out example file is empty")
        for index, record in enumerate(self.heldout_records, 1):
            self.assertIsInstance(record, dict, f"heldout:{index}: record must be an object")
            texts = query_texts([record])
            self.assertTrue(texts, f"heldout:{index}: no explicit user query text")
            expected = record.get("expected", record.get("expected_action", record.get("action")))
            self.assertIsNotNone(expected, f"heldout:{index}: missing expected behavior")

    def test_sources_and_participants_do_not_cross_splits(self) -> None:
        split_ids: dict[str, set[str]] = {}
        original_rows = 0
        for split, records in self.split_records.items():
            ids: set[str] = set()
            for index, record in enumerate(records, 1):
                ids.update(participant_ids(record))
                provenance = record.get("provenance", record.get("source_provenance"))
                source_text = json.dumps(
                    {"source": record.get("source"), "provenance": provenance},
                    ensure_ascii=False,
                    sort_keys=True,
                ).casefold()
                if any(marker in source_text for marker in ("authoritative", "original", "statelm", "thri")):
                    original_rows += 1
                    self.assertIsInstance(provenance, Mapping, f"{split}:{index}: original-derived row lacks provenance")
                    self.assertTrue(participant_ids(record), f"{split}:{index}: original-derived row lacks participant ID")
                    source_ids = list(
                        nested_values(
                            provenance,
                            {"sample", "sample_id", "case_id", "source_id", "trial_id", "row_id"},
                        )
                    )
                    self.assertTrue(source_ids, f"{split}:{index}: original-derived row lacks a source record ID")
            split_ids[split] = ids

        self.assertGreater(original_rows, 0, "no records are traceable to the original GitHub dataset")
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = split_ids[left] & split_ids[right]
            self.assertFalse(overlap, f"participant leakage between {left} and {right}: {sorted(overlap)}")

    def test_heldout_text_and_hashes_are_quarantined(self) -> None:
        heldout_text = query_texts(self.heldout_records)
        self.assertTrue(heldout_text, "no held-out query text found")
        heldout_hashes = {sha256_text(text) for text in heldout_text}
        explicit_hashes = {
            str(value).casefold()
            for value in nested_values(self.heldout_records, {"sha256", "text_sha256", "query_sha256"})
            if isinstance(value, str)
        }
        heldout_hashes.update(explicit_hashes)

        for split, records in self.split_records.items():
            split_text = query_texts(records)
            overlap = heldout_text & split_text
            self.assertFalse(overlap, f"held-out query text leaked into {split}: {sorted(overlap)}")
            split_hashes = {sha256_text(text) for text in split_text}
            split_hashes.update(
                str(value).casefold()
                for value in nested_values(records, {"sha256", "text_sha256", "query_sha256"})
                if isinstance(value, str)
            )
            self.assertFalse(
                heldout_hashes & split_hashes,
                f"held-out query hash leaked into {split}: {sorted(heldout_hashes & split_hashes)}",
            )

    def test_source_specific_files_and_manifest_hashes(self) -> None:
        manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["split_policy"]["counts"], {"train": 16, "validation": 4, "test": 5})
        self.assertEqual(manifest["split_policy"]["generated_counts"], {"train": 36, "validation": 18, "test": 18})
        for source in ("original_derived", "generated"):
            for split in ("train", "validation", "test"):
                path = DATA / source / f"{split}.jsonl"
                self.assertTrue(path.is_file(), f"missing separated source file {path}")
                metadata = manifest["artifacts"][f"{source}/{split}.jsonl"]
                self.assertEqual(metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(manifest["split_policy"]["similarity_evaluation_counts"], {"validation": 5, "test": 5})
        for split in ("validation", "test"):
            path = DATA / "similarity" / f"{split}.jsonl"
            self.assertTrue(path.is_file(), f"missing cosine-similarity evaluation file {path}")
            metadata = manifest["artifacts"][f"similarity/{split}.jsonl"]
            self.assertEqual(metadata["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertTrue(manifest["heldout_policy"]["excluded_from_train_validation_test"])

    def test_every_heldout_turn_is_excluded_by_exact_text(self) -> None:
        heldout_turns = {
            normalize_phrase(turn["utterance"])
            for case in self.heldout_records
            for turn in case["turns"]
        }
        for split, records in self.split_records.items():
            split_texts = query_texts(records)
            self.assertFalse(heldout_turns & split_texts, f"held-out turn leaked into {split}")

    def test_similarity_validation_and_test_utterances_are_not_training_examples(self) -> None:
        training_texts = query_texts(self.split_records["train"])
        for split in ("validation", "test"):
            rows = read_jsonl(DATA / "similarity" / f"{split}.jsonl")
            evaluation_texts = {
                normalize_phrase(turn["utterance"])
                for row in rows
                for turn in row["turns"]
            }
            self.assertFalse(
                training_texts & evaluation_texts,
                f"similarity {split} utterance leaked into training",
            )

    def test_dataset_builder_reproduces_committed_artifacts_byte_for_byte(self) -> None:
        notebook = ROOT / "notebooks" / "Robot_SLM_Adaptive_FSM_Inventory_Query(AAA).ipynb"
        with tempfile.TemporaryDirectory() as directory:
            rebuilt = Path(directory) / "fsm_v1"
            build(ROOT, notebook, rebuilt)
            committed_paths = {
                path.relative_to(DATA)
                for path in DATA.rglob("*")
                if path.is_file() and path.name != "README.md"
            }
            rebuilt_paths = {
                path.relative_to(rebuilt)
                for path in rebuilt.rglob("*")
                if path.is_file()
            }
            self.assertEqual(rebuilt_paths, committed_paths)
            for relative in sorted(rebuilt_paths):
                self.assertEqual(
                    (rebuilt / relative).read_bytes(),
                    (DATA / relative).read_bytes(),
                    f"rebuilt artifact differs: {relative}",
                )


class RuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_path = DATA / "train.jsonl"
        cls.inventory_path = DATA / "inventory.json"
        if not cls.train_path.is_file() or not cls.inventory_path.is_file():
            raise AssertionError("training split and inventory must exist before runtime tests")
        cls.model = compile_jsonl(cls.train_path)
        cls.inventory = Inventory.load(cls.inventory_path)

    def new_session(self, inventory: Inventory | None = None) -> InventorySession:
        return InventorySession(self.model, inventory or self.inventory)

    def test_compilation_uses_train_only(self) -> None:
        provenance = self.model.artifact.get("provenance")
        self.assertIsInstance(provenance, list)
        self.assertEqual(len(provenance), 1, "compiled model must name only its training source")
        self.assertEqual(Path(provenance[0]["path"]).resolve(), self.train_path.resolve())
        self.assertEqual(
            provenance[0]["sha256"],
            hashlib.sha256(self.train_path.read_bytes()).hexdigest(),
        )
        query_count = sum(
            record_kind(record) in {"query", "query_pattern", "utterance"}
            for record in read_jsonl(self.train_path)
        )
        transition_count = sum(
            record_kind(record) in {"transition", "controller", "controller_transition"}
            for record in read_jsonl(self.train_path)
        )
        self.assertEqual(self.model.artifact["query_automaton"]["example_count"], query_count)
        self.assertEqual(self.model.artifact["controller"]["example_count"], transition_count)

    def test_compilation_and_runtime_are_deterministic(self) -> None:
        second = compile_jsonl(self.train_path)
        self.assertEqual(self.model.artifact, second.artifact)
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            self.model.dump(first_path)
            second.dump(second_path)
            self.assertEqual(first_path.read_bytes(), second_path.read_bytes())

        turns = ["Where is the calculator?", "Where is the tape?", "Blue.", "Where is the stapler?", "Tell me a joke."]
        sessions = (self.new_session(), self.new_session())
        results = [[session.handle(turn).to_dict() for turn in turns] for session in sessions]
        self.assertEqual(results[0], results[1])

    def test_unique_lookup_returns_inventory_location(self) -> None:
        item = next(
            candidate
            for candidate in self.inventory.items
            if candidate.location is not None
            and len(self.inventory.lookup({self.model.entity_slot: candidate.name}, entity_slot=self.model.entity_slot)) == 1
        )
        result = self.new_session().handle(f"Where is the {item.name}?")
        self.assertEqual(result.action, "RETURN_LOCATION")
        self.assertEqual(result.matches, (item.name,))
        self.assertIn(item.location, result.text)

    def test_ambiguity_then_blue_clarification(self) -> None:
        session = self.new_session()
        first = session.handle("Where is the tape?")
        self.assertEqual(first.action, "ASK_WHICH_ONE")
        self.assertGreaterEqual(len(first.matches), 2)
        self.assertTrue(session.awaiting_clarification)
        second = session.handle("Blue.")
        self.assertEqual(second.action, "RETURN_LOCATION")
        self.assertEqual(len(second.matches), 1)
        self.assertIn("blue", second.matches[0].casefold())
        self.assertFalse(session.awaiting_clarification)

    def test_missing_object(self) -> None:
        result = self.new_session().handle("Where is the stapler?")
        self.assertEqual(result.action, "OBJECT_MISSING")
        self.assertEqual(result.matches, ())
        self.assertIn("stapler", result.text.casefold())

    def test_missing_location_from_temporary_inventory_record(self) -> None:
        records = [
            {
                "id": item.id,
                "name": item.name,
                "aliases": list(item.aliases),
                "family": item.family,
                "attributes": dict(item.attributes),
                "location": item.location,
            }
            for item in self.inventory.items
        ]
        records.append(
            {
                "id": "locationless_widget",
                "name": "locationless widget",
                "aliases": [],
                "family": "locationless_widget",
                "attributes": {},
                "location": None,
            }
        )
        result = self.new_session(Inventory.from_records(records)).handle("Where is the locationless widget?")
        self.assertEqual(result.action, "LOCATION_MISSING")
        self.assertEqual(result.matches, ("locationless widget",))
        self.assertIn("location is missing", result.text.casefold())

    def test_unsupported_request(self) -> None:
        result = self.new_session().handle("Tell me a joke.")
        self.assertEqual(result.action, "UNSUPPORTED_REQUEST")
        self.assertEqual(result.matches, ())
        self.assertNotIn("drawer", result.text.casefold())

    def test_artifact_is_learned_fsm_data_with_cosine_embeddings_and_no_llm(self) -> None:
        artifact = self.model.artifact
        self.assertEqual(artifact.get("format"), "fsm-only-v1")
        self.assertEqual(artifact.get("architecture"), "similarity-augmented-fsm")
        self.assertGreater(artifact["query_automaton"]["pattern_count"], 0)
        self.assertGreater(len(artifact["query_automaton"]["states"]), 1)
        self.assertTrue(artifact["controller"]["transitions"])
        self.assertTrue(
            any(state.get("transitions") for state in artifact["query_automaton"]["states"]),
            "compiled artifact contains no learned query transitions",
        )
        serialized = json.dumps(artifact, sort_keys=True).casefold()
        for forbidden in ("openai", "transformers", "torch", "qwen", "llama", "api_key"):
            self.assertNotIn(forbidden, serialized)

        similarity = artifact.get("similarity")
        self.assertIsInstance(similarity, Mapping)
        self.assertEqual(similarity["method"], "tfidf_word_char_ngram_cosine_v1")
        self.assertTrue(similarity["enabled"])
        self.assertTrue(similarity["idf"])
        self.assertTrue(similarity["prototypes"])
        self.assertEqual(similarity["training_scope"], "query patterns from compiler input only")

        forbidden_imports = {"openai", "transformers", "torch", "tensorflow", "langchain", "sentence_transformers"}
        for path in sorted((ROOT / "fsm_only").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
            self.assertFalse(
                imports & forbidden_imports,
                f"{path.name} imports a non-FSM model dependency: {sorted(imports & forbidden_imports)}",
            )

    def test_frozen_model_passes_all_heldout_conversations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            self.model.dump(model_path)
            summary = evaluate(model_path, DATA / "inventory.json", DATA / "heldout.jsonl")
        self.assertTrue(summary["success"], json.dumps(summary, indent=2))
        self.assertEqual(summary["passed_cases"], summary["cases"])
        self.assertEqual(summary["passed_turns"], summary["turns"])

    def test_frozen_model_passes_validation_and_test_suites(self) -> None:
        suites = (
            DATA / "generated" / "validation.jsonl",
            DATA / "generated" / "test.jsonl",
            DATA / "original_derived" / "validation.jsonl",
            DATA / "original_derived" / "test.jsonl",
            DATA / "similarity" / "validation.jsonl",
            DATA / "similarity" / "test.jsonl",
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.json"
            self.model.dump(model_path)
            summaries = [evaluate(model_path, DATA / "inventory.json", path) for path in suites]
        failures = [summary for summary in summaries if not summary["success"]]
        self.assertFalse(failures, json.dumps(failures, indent=2))
        self.assertEqual(sum(summary["passed_cases"] for summary in summaries), 55)
        self.assertEqual(sum(summary["passed_turns"] for summary in summaries), 61)

    def test_cosine_similarity_handles_unseen_query_forms(self) -> None:
        lab_inventory = Inventory.load(self.inventory_path, setting="lab")
        examples = {
            "Where is SD card?": ("RETURN_LOCATION", ("SD cards",)),
            "Where is item SD card?": ("RETURN_LOCATION", ("SD cards",)),
            "Where is tape?": ("ASK_WHICH_ONE", ("blue tape", "white tape")),
            "Where is item x?": ("OBJECT_MISSING", ()),
        }
        for utterance, (expected_action, expected_matches) in examples.items():
            with self.subTest(utterance=utterance):
                result = self.new_session(lab_inventory).handle(utterance)
                self.assertEqual(result.action, expected_action)
                self.assertEqual(result.matches, expected_matches)
                self.assertEqual(result.query_match_method, "cosine_similarity")
                self.assertGreaterEqual(result.query_similarity or 0.0, self.model.similarity["minimum"])

    def test_inventory_cosine_similarity_handles_a_typo_without_forcing_unknown_objects(self) -> None:
        typo = self.new_session().handle("Where is the sd crad?")
        self.assertEqual(typo.action, "RETURN_LOCATION")
        self.assertEqual(typo.matches, ("SD cards",))
        self.assertEqual(typo.inventory_match_method, "cosine_similarity")
        missing = self.new_session().handle("Where is the pen?")
        self.assertEqual(missing.action, "OBJECT_MISSING")
        self.assertEqual(missing.matches, ())

    def test_cosine_similarity_does_not_turn_unrelated_requests_into_inventory_queries(self) -> None:
        for utterance in ("Tell me a joke.", "What is the capital of France?", "Book me a flight."):
            with self.subTest(utterance=utterance):
                result = self.new_session().handle(utterance)
                self.assertEqual(result.action, "UNSUPPORTED_REQUEST")
                self.assertIsNone(result.query_match_method)


if __name__ == "__main__":
    unittest.main()
