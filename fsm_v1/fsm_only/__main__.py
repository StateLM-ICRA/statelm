"""Command-line entry point for ``python -m fsm_only``."""

from __future__ import annotations

import argparse
import json
import sys

from .compiler import compile_jsonl
from .errors import FSMOnlyError
from .inventory import Inventory
from .model import CompiledFSM
from .runtime import InventorySession


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m fsm_only", description="Compile and run a pure deterministic inventory FSM/FST")
    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="compile labeled JSONL into a learned FSM artifact")
    compile_parser.add_argument("inputs", nargs="+", help="labeled JSONL input file(s)")
    compile_parser.add_argument("-o", "--output", required=True, help="output model JSON")
    compile_parser.add_argument("--start-state", help="override the labeled controller start state")
    compile_parser.add_argument("--entity-slot", help="slot containing the object or family mention")
    compile_parser.add_argument("--allow-incomplete", action="store_true", help="skip required-behavior acceptance validation")

    chat_parser = subparsers.add_parser("chat", help="run a terminal session with a compiled model")
    chat_parser.add_argument("model", help="compiled model JSON")
    chat_parser.add_argument("inventory", help="inventory JSON or JSONL")
    chat_parser.add_argument("--setting", default="lab", help="inventory setting to load (default: lab); use '' for all")
    chat_parser.add_argument("--location-field", default="location", help="inventory location field")
    chat_parser.add_argument("--json", action="store_true", help="print structured JSON turn results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "compile":
            model = compile_jsonl(
                args.inputs,
                args.output,
                start_state=args.start_state,
                entity_slot=args.entity_slot,
                require_complete=not args.allow_incomplete,
            )
            automaton = model.artifact["query_automaton"]
            print(
                f"compiled {automaton['example_count']} query examples into "
                f"{automaton['state_count_after_minimization']} states "
                f"({automaton['state_count_before_minimization']} before minimization)"
            )
            return 0
        setting = args.setting or None
        session = InventorySession(
            CompiledFSM.load(args.model),
            Inventory.load(args.inventory, location_field=args.location_field, setting=setting),
        )
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            if text == "/reset":
                session.reset()
                print("Session reset.")
                continue
            result = session.handle(text)
            print(json.dumps(result.to_dict(), ensure_ascii=False) if args.json else result.text)
        return 0
    except FSMOnlyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
