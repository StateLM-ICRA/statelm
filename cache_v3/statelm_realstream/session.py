"""Interactive session that follows the deployed flow of the robot runtime.

Per user utterance:
  1. dialogue decision (FSM if a cached pattern matches, else SLM);
  2. if the decision is ``locate``, the rendered answer is reviewed before it
     is spoken (pre-send review: FSM or SLM), exactly as ``RobotSession`` in
     the SLM bundle does;
  3. the spoken reply is appended to the history and memory is updated.

``robot(line)`` injects an external robot line (for example a wrong drawer)
after the last user request: it is reviewed as a proposal, and then treated
as spoken so that a following user correction is handled from the history.
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Mapping, Sequence

from .contract import decision_dict, empty_memory, make_payload, next_memory, render_response
from .features import extract_text_features
from .router import PaperAlignedRouter, RouteResult


class CombinedSession:
    def __init__(
        self,
        router: PaperAlignedRouter,
        inventories: Mapping[str, Sequence[Mapping[str, Any]]],
        setting: str = "lab",
        learn: bool = False,
        pre_send_review: bool = True,
    ) -> None:
        self.router = router
        self.inventories = {k: list(v) for k, v in inventories.items()}
        self.setting = setting
        self.learn = learn
        self.pre_send_review = pre_send_review
        self.reset()

    def reset(self) -> None:
        self.history: list[dict[str, str]] = []
        self.memory = empty_memory()
        self.session_id = "chat:" + uuid.uuid4().hex
        self.transcript: list[dict[str, Any]] = []
        self._memory_at_last_user = empty_memory()

    @property
    def inventory(self) -> list[dict[str, Any]]:
        return self.inventories[self.setting]

    def set_setting(self, setting: str) -> None:
        if setting not in self.inventories:
            raise ValueError(f"Unknown setting {setting}; choose one of {sorted(self.inventories)}")
        self.setting = setting
        self.reset()

    @staticmethod
    def _describe(result: RouteResult) -> dict[str, Any]:
        d = result.decision
        return {
            "source": result.source,
            "action": d["action"],
            "item": d["resolved_item_id"],
            "failure_type": d["failure_type"],
            "recovery_strategy": d["recovery_strategy"],
            "similarity": None if result.similarity is None else round(result.similarity, 3),
            "pattern_id": result.pattern_id,
            "reason": result.reason,
            "typed_text": result.typed_text,
            "response": result.response,
        }

    def user(self, text: str) -> dict[str, Any]:
        """One user utterance through the deployed flow."""

        text = text.strip()
        if not text:
            raise ValueError("Empty utterance")
        history = self.history + [{"role": "user", "text": text}]
        memory_before = copy.deepcopy(self.memory)
        self._memory_at_last_user = copy.deepcopy(memory_before)
        payload = make_payload(history, memory_before, self.inventory)
        initial = self.router.route(payload, session_id=self.session_id, learn=self.learn)
        steps = [{"stage": "dialogue_decision", **self._describe(initial)}]
        final = initial
        if self.pre_send_review and initial.decision["action"] == "locate":
            proposal = render_response(initial.decision, self.inventory)
            review_payload = make_payload(history, memory_before, self.inventory,
                                          allowed_actions=["locate", "recover"],
                                          proposed_robot_answer=proposal)
            review = self.router.route(review_payload, session_id=self.session_id + ":review",
                                       learn=self.learn)
            steps.append({"stage": "pre_send_review", "proposal": proposal, **self._describe(review)})
            final = review
        response = final.response
        self.history = history + [{"role": "robot", "text": response}]
        self.memory = next_memory(memory_before, final.decision)
        record = {"user": text, "response": response, "decision": final.decision, "steps": steps,
                  "memory": copy.deepcopy(self.memory)}
        self.transcript.append(record)
        return record

    def robot(self, line: str) -> dict[str, Any]:
        """Inject an external robot line after the last user request and review it."""

        line = line.strip()
        if not line:
            raise ValueError("Empty robot line")
        # The history up to and including the last user turn.
        cut = len(self.history)
        while cut > 0 and self.history[cut - 1]["role"] != "user":
            cut -= 1
        if cut == 0:
            raise ValueError("Say a user request first, then inject the robot line")
        history = self.history[:cut]
        memory_before = copy.deepcopy(self._memory_at_last_user)
        review_payload = make_payload(history, memory_before, self.inventory,
                                      allowed_actions=["locate", "recover"], proposed_robot_answer=line)
        review = self.router.route(review_payload, session_id=self.session_id + ":inject", learn=self.learn)
        described = self._describe(review)
        # Treat the injected line as spoken (a failure that went out), so that
        # a following correction is judged against it.
        features = extract_text_features(line, self.inventory)
        spoken_item = features.item_ids[0] if len(features.item_ids) == 1 else None
        by_id = {item["id"]: item for item in self.inventory}
        spoken_decision = {
            "action": "locate" if spoken_item else "clarify",
            "item_family": by_id[spoken_item]["family"] if spoken_item else None,
            "resolved_item_id": spoken_item,
            "attributes": dict(by_id[spoken_item]["attributes"]) if spoken_item else {},
            "missing_attributes": [],
            "failure_detected": False,
            "failure_type": "none",
            "recovery_strategy": "none",
        }
        self.history = history + [{"role": "robot", "text": line}]
        self.memory = next_memory(memory_before, spoken_decision)
        record = {"injected_robot_line": line, "review": described,
                  "what_statelm_would_have_said": review.response,
                  "memory": copy.deepcopy(self.memory)}
        self.transcript.append(record)
        return record


def format_record(record: Mapping[str, Any]) -> str:
    lines = []
    if "user" in record:
        lines.append(f"You:   {record['user']}")
        for step in record["steps"]:
            tag = f"[{step['stage']}: source={step['source']} action={step['action']}"
            if step["item"]:
                tag += f" item={step['item']}"
            if step["failure_type"] != "none":
                tag += f" failure={step['failure_type']} strategy={step['recovery_strategy']}"
            if step["similarity"] is not None:
                tag += f" cosine={step['similarity']}"
            if step.get("proposal"):
                tag += f" proposal={step['proposal']!r}"
            if step.get("reason"):
                tag += f" note={step['reason']}"
            lines.append("       " + tag + "]")
        lines.append(f"Robot: {record['response']}")
    else:
        lines.append(f"Robot (injected): {record['injected_robot_line']}")
        r = record["review"]
        tag = f"[pre_send_review of the injected line: source={r['source']} action={r['action']}"
        if r["item"]:
            tag += f" item={r['item']}"
        if r["failure_type"] != "none":
            tag += f" failure={r['failure_type']} strategy={r['recovery_strategy']}"
        if r["similarity"] is not None:
            tag += f" cosine={r['similarity']}"
        lines.append("       " + tag + "]")
        lines.append(f"       StateLM would have said instead: {record['what_statelm_would_have_said']}")
    return "\n".join(lines)
