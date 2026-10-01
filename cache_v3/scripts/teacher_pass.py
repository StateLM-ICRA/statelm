#!/usr/bin/env python3
"""Run the trained SLM once on every stream payload and cache its raw output.

The cache is keyed by the SHA-256 of the canonical payload JSON, so a payload
maps to exactly one SLM decision and the replay is reproducible.  Writing is
incremental (one JSON line per decision, flushed immediately): if Colab
disconnects, rerun the cell and only missing payloads are computed.

Requires the SLM bundle folder (``slm_runtime.py``, ``adapter/``,
``runtime_config.json``) and a CUDA GPU for reasonable speed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def payload_key(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_bundle_runtime(bundle_dir: Path):
    spec = importlib.util.spec_from_file_location("robot_slm_runtime", bundle_dir / "slm_runtime.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(2 ** 20):
            digest.update(block)
    return digest.hexdigest()


class LiveTeacher:
    """Calls the loaded SLM and records every raw output."""

    def __init__(self, engine, cache_path: Path):
        self.engine = engine
        self.cache_path = cache_path
        self.cache = {r["key"]: r["raw"] for r in read_jsonl(cache_path)}
        self.calls = 0

    def __call__(self, payload: dict[str, Any]) -> str:
        key = payload_key(payload)
        if key in self.cache:
            return self.cache[key]
        started = time.perf_counter()
        raw = self.engine(payload)
        elapsed = (time.perf_counter() - started) * 1000
        self.calls += 1
        self.cache[key] = raw
        with self.cache_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"key": key, "raw": raw, "generation_ms": elapsed}, ensure_ascii=False) + "\n")
        return raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True, help="robot_slm_bundle folder")
    parser.add_argument("--stream", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--expected-adapter-sha256", default=None)
    parser.add_argument("--quantize", action="store_true", help="4-bit load (T4); default bf16 (A100)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.expected_adapter_sha256:
        actual = sha256_file(args.bundle / "adapter" / "adapter_model.safetensors")
        if actual != args.expected_adapter_sha256:
            raise SystemExit(f"Adapter hash {actual} does not match the expected checkpoint")
    runtime = load_bundle_runtime(args.bundle)
    stream = read_jsonl(args.stream)
    existing = {r["key"] for r in read_jsonl(args.cache)}
    todo = [row for row in stream if payload_key(row["input"]) not in existing]
    print(f"{len(stream)} stream payloads, {len(stream) - len(todo)} cached, {len(todo)} to compute")
    if not todo:
        return
    print("Loading the trained SLM (this can take several minutes)...")
    engine = runtime.load_runtime(args.bundle, device=args.device, quantize=args.quantize)
    teacher = LiveTeacher(engine, args.cache)
    started = time.perf_counter()
    for index, row in enumerate(todo, start=1):
        raw = teacher(row["input"])
        if index % 10 == 0 or index == len(todo):
            elapsed = time.perf_counter() - started
            print(f"  {index}/{len(todo)} decisions  ({elapsed / index:.1f} s per call)  last: {raw[:90]}")
    print(f"done: {teacher.calls} SLM calls written to {args.cache}")


if __name__ == "__main__":
    main()
