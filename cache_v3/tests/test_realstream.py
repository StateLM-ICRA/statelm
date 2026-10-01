"""Controller tests.  They use a deterministic test double, never the SLM."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from statelm_realstream import (  # noqa: E402
    PaperAlignedRouter,
    RouterConfig,
    empty_memory,
    extract_review_features,
    extract_text_features,
    make_payload,
    next_memory,
)
from statelm_realstream.testing_teacher import OracleTeacher  # noqa: E402

INVENTORY = json.loads((ROOT / "data" / "inventory_by_setting.json").read_text())["lab"]
BY_ID = {item["id"]: item for item in INVENTORY}


def locate(item_id):
    item = BY_ID[item_id]
    return {
        "action": "locate", "item_family": item["family"], "resolved_item_id": item_id,
        "attributes": dict(item["attributes"]), "missing_attributes": [],
        "failure_detected": False, "failure_type": "none", "recovery_strategy": "none",
    }


class BadTeacher:
    """Returns a decision that binds an item the user never asked for."""

    def __init__(self, item_id):
        self.item_id = item_id
        self.calls = 0

    def __call__(self, payload):
        self.calls += 1
        return locate(self.item_id)


class RealStreamTests(unittest.TestCase):
    def setUp(self):
        self.teacher = OracleTeacher(INVENTORY)
        self.config = RouterConfig(
            merge_similarity=0.70, routing_similarity=0.70,
            admission_threshold=0.70, minimum_observations=2, audit_probability=0.0,
        )
        self.router = PaperAlignedRouter(self.teacher, INVENTORY, config=self.config)

    def request(self, text, history=None, memory=None):
        turns = list(history or []) + [{"role": "user", "text": text}]
        return make_payload(turns, memory or empty_memory(), INVENTORY)

    def review(self, text, proposal, history=None):
        turns = list(history or []) + [{"role": "user", "text": text}]
        return make_payload(turns, empty_memory(), INVENTORY,
                            allowed_actions=["locate", "recover"], proposed_robot_answer=proposal)

    # typing
    def test_review_typing_marks_other_item_and_wrong_drawer(self):
        f = extract_review_features("Where are the gloves?", "The blue tape is in drawer 1.", INVENTORY)
        self.assertIn("<item:other>", f.proposal_typed)
        self.assertIn("<drawer:mismatch>", f.proposal_typed)
        f = extract_review_features("Where are the gloves?", "The gloves are in drawer four.", INVENTORY)
        self.assertIn("<item:same>", f.proposal_typed)
        self.assertIn("<drawer:match>", f.proposal_typed)
        f = extract_review_features("Where are the wires?", "I think the item is in drawer too.", INVENTORY)
        self.assertIn("<drawer:mismatch>", f.proposal_typed)  # wires are in drawer 3
        f = extract_review_features("Where are the wires?", "The wires are in drawer 99.", INVENTORY)
        self.assertIn("<drawer:mismatch>", f.proposal_typed)

    def test_user_typing_keeps_item_slot(self):
        f = extract_text_features("Where's the circuit board?", INVENTORY)
        self.assertEqual(f.typed_text, "where s the <item>")
        self.assertEqual(f.item_ids, ("circuit_board",))

    # no hallucination through the SLM path
    def test_review_of_wrong_item_recovers_requested_item(self):
        result = self.router.route(self.review("Where are the gloves?", "The blue tape is in drawer 1."))
        self.assertEqual(result.decision["action"], "recover")
        self.assertEqual(result.decision["resolved_item_id"], "gloves")
        self.assertIn("gloves", result.response)
        self.assertIn("drawer 4", result.response)
        self.assertNotIn("tape", result.response)

    def test_correction_turn_recovers_requested_item(self):
        history = [{"role": "user", "text": "Where are the gloves?"},
                   {"role": "robot", "text": "The blue tape is listed in drawer 1."}]
        memory = next_memory(empty_memory(), locate("blue_tape"))
        result = self.router.route(self.request("I said gloves.", history=history, memory=memory))
        self.assertEqual(result.decision["action"], "recover")
        self.assertEqual(result.decision["resolved_item_id"], "gloves")
        self.assertIn("drawer 4", result.response)
        self.assertNotIn("tape", result.response)

    def test_bad_teacher_item_is_rejected_by_task_guard(self):
        router = PaperAlignedRouter(BadTeacher("lidocaine" if "lidocaine" in BY_ID else "blue_tape"),
                                    INVENTORY, config=self.config)
        result = router.route(self.request("Can I have a lardocaine?"))
        self.assertIsNone(result.decision["resolved_item_id"])
        self.assertIn(result.decision["action"], {"clarify", "not_found"})
        self.assertIsNotNone(result.reason)
        result = router.route(self.request("Where are the gloves?"))
        self.assertEqual(result.decision["resolved_item_id"], "gloves")  # guard repairs to the user's item
        self.assertIn("drawer 4", result.response)

    def test_search_failure_review_keeps_item_and_true_drawer(self):
        result = self.router.route(self.review("Which drawer is the wires in?", "I think the item is in drawer one."))
        self.assertEqual(result.decision["action"], "recover")
        self.assertEqual(result.decision["failure_type"], "search_failure")
        self.assertEqual(result.decision["resolved_item_id"], "wires")
        self.assertIn("drawer 3", result.response)

    # FSM_0 seeding and generalisation
    def test_seed_request_generalises_to_new_item_same_form(self):
        seeded = self.router.seed_example(self.request("Where are the scissors?"), locate("scissors"), "seed:1")
        self.assertIsNotNone(seeded)
        self.assertEqual(seeded.status, "active")
        result = self.router.route(self.request("Where are the wires?"))
        self.assertEqual(result.source, "fsm")
        self.assertEqual(result.decision["resolved_item_id"], "wires")
        self.assertIn("drawer 3", result.response)
        self.assertEqual(self.teacher.calls, 0)

    def test_seeded_approval_never_approves_wrong_drawer(self):
        self.router.seed_example(self.review("Where are the scissors?", "Item is in drawer five."),
                                 locate("scissors"), "seed:review")
        ok = self.router.route(self.review("Where are the wires?", "Item is in drawer three."))
        self.assertEqual(ok.source, "fsm")
        self.assertEqual(ok.decision["action"], "locate")
        wrong = self.router.route(self.review("Where are the wires?", "Item is in drawer one."))
        self.assertEqual(wrong.decision["action"], "recover")
        self.assertEqual(wrong.decision["resolved_item_id"], "wires")
        self.assertIn("drawer 3", wrong.response)

    def test_cached_recovery_pattern_binds_current_item(self):
        # Learn the "wrong drawer" recovery from two different items.
        self.router.route(self.review("Where are the scissors?", "I think the item is in drawer one."),
                          evidence_id="e1")
        self.router.route(self.review("Where are the batteries?", "I think the item is in drawer one."),
                          evidence_id="e2")
        active = [p for p in self.router.patterns if p.status == "active"]
        self.assertTrue(active)
        result = self.router.route(self.review("Where are the wires?", "I think the item is in drawer one."),
                                   evidence_id="e3")
        self.assertEqual(result.source, "fsm")
        self.assertEqual(result.decision["resolved_item_id"], "wires")
        self.assertEqual(result.decision["failure_type"], "search_failure")
        self.assertIn("drawer 3", result.response)

    def test_unresolved_request_is_never_bound_from_cache(self):
        self.router.seed_example(self.request("Where are the scissors?"), locate("scissors"), "seed:1")
        result = self.router.route(self.request("Where are the sizzlers?"))
        self.assertIsNone(result.decision["resolved_item_id"])
        self.assertEqual(result.source, "slm")

    def test_snapshot_round_trip(self):
        self.router.seed_example(self.request("Where are the scissors?"), locate("scissors"), "seed:1")
        snapshot = json.loads(json.dumps(self.router.snapshot()))
        restored = PaperAlignedRouter.from_snapshot(snapshot, self.teacher, INVENTORY)
        result = restored.route(self.request("Where are the wires?"))
        self.assertEqual(result.source, "fsm")
        self.assertIn("drawer 3", result.response)

    def test_learn_flag_off_does_not_change_cache(self):
        before = len(self.router.patterns)
        self.router.route(self.request("Where are the wires?"), learn=False)
        self.assertEqual(len(self.router.patterns), before)


if __name__ == "__main__":
    unittest.main()
