"""Pure deterministic learned FSM/FST inventory locator.

Stable API::

    model = compile_jsonl("labels.jsonl", "model.json")
    inventory = Inventory.load("inventory.json", setting="lab")
    session = InventorySession(model, inventory)
    print(session.reply("Where is the blue tape?"))
"""

from .compiler import compile_jsonl, compile_records
from .errors import (
    ControllerError,
    FSMOnlyError,
    InventoryFormatError,
    ModelFormatError,
    TrainingDataError,
)
from .inventory import Inventory, InventoryItem
from .model import CompiledFSM, ControllerStep, QueryMatch
from .runtime import InventorySession, TurnResult

__all__ = [
    "CompiledFSM",
    "ControllerError",
    "ControllerStep",
    "FSMOnlyError",
    "Inventory",
    "InventoryFormatError",
    "InventoryItem",
    "InventorySession",
    "ModelFormatError",
    "QueryMatch",
    "TrainingDataError",
    "TurnResult",
    "compile_jsonl",
    "compile_records",
]
