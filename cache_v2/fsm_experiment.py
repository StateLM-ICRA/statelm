"""Run a request stream through the progressive FSM router and plot convergence."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

GROUPS = ("routine", "risky", "unlearnable")
LEARNABLE = {"routine", "risky"}


def payload_key(payload):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MemoTeacher:
    """Greedy SLM calls are deterministic, so identical payloads are answered once.

    Raw outputs are appended to a JSONL file so a restarted runtime resumes quickly.
    """

    def __init__(self, engine, cache_path=None):
        self.engine = engine
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        self.calls = 0
        self.seconds = 0.0
        if self.cache_path and self.cache_path.is_file():
            for line in self.cache_path.read_text().splitlines():
                if line.strip():
                    record = json.loads(line)
                    self.cache[record["key"]] = record["raw"]

    def __call__(self, payload):
        key = payload_key(payload)
        if key not in self.cache:
            started = time.perf_counter()
            raw = self.engine(payload)
            self.seconds += time.perf_counter() - started
            self.calls += 1
            self.cache[key] = raw
            if self.cache_path:
                with self.cache_path.open("a") as handle:
                    handle.write(json.dumps({"key": key, "raw": raw}, ensure_ascii=False) + "\n")
        return self.cache[key]


def slm_decision(teacher, validate_raw, payload):
    """Validated SLM decision as a dict, or None if the SLM output is rejected."""
    try:
        decision = validate_raw(teacher(payload), payload)
        return decision.model_dump() if hasattr(decision, "model_dump") else dict(decision)
    except Exception:
        return None


def run_stream(rows, router, teacher, validate_raw, decisions_equal, extract_bundle,
               progress_every=100, log=print):
    """Route every row, compare against the SLM, and track per-pattern trust."""
    patterns = sorted({r["pattern_id"] for r in rows if r["pattern_group"] in GROUPS})
    group_of = {r["pattern_id"]: r["pattern_group"] for r in rows}
    keys = []
    for row in rows:
        exact, typed = router._keys(row["input"], extract_bundle(row["input"], router.policy))
        keys.append((exact, typed))
    typed_keys = {p: set() for p in patterns}
    exact_keys = {p: set() for p in patterns}
    seen = {p: 0 for p in patterns}
    promoted_after = {}
    logs, trusted_rows = [], []
    started = time.perf_counter()

    def is_active(level, key):
        entry = router._entry_for_key(level, key)
        return entry is not None and entry.status == "active"

    for index, row in enumerate(rows):
        payload, pattern, group = row["input"], row["pattern_id"], row["pattern_group"]
        exact_key, typed_key = keys[index]
        if pattern in seen:
            seen[pattern] += 1
            typed_keys[pattern].add(typed_key)
            exact_keys[pattern].add(exact_key)

        trace = fsm_decision = error = None
        try:
            raw, trace = router.predict(payload, evidence_id=f"request:{row['id']}")
            routed = json.loads(raw)
        except Exception as exc:  # the SLM produced an invalid decision
            routed, error = None, f"{type(exc).__name__}: {exc}"
        if trace is not None and trace.trace_id in router.traces:
            router.discard_trace(trace.trace_id, reason="offline_stream")

        primary = trace.primary if trace is not None else "slm"
        if primary == "fsm":
            fsm_decision = routed
        reference = routed if primary == "slm" else slm_decision(teacher, validate_raw, payload)
        fsm_matches = bool(fsm_decision is not None and reference is not None
                           and decisions_equal(fsm_decision, reference))
        gold_ok = routed is not None and decisions_equal(routed, row["target"])

        trusted = {}
        for p in patterns:
            ok = any(is_active("typed", k) for k in typed_keys[p]) or any(
                is_active("exact", k) for k in exact_keys[p])
            trusted[p] = ok
            if ok and p not in promoted_after:
                promoted_after[p] = seen[p]
        trusted_rows.append(trusted)

        logs.append({
            "step": index + 1,
            "id": row["id"],
            "batch": row["batch"],
            "pattern_id": pattern,
            "pattern_group": group,
            "scenario": row["scenario"],
            "route": trace.route if trace is not None else "slm_invalid",
            "primary": primary,
            "teacher_called": trace.teacher_called if trace is not None else True,
            "audited": trace.audited if trace is not None else False,
            "fsm_answered": fsm_decision is not None,
            "fsm_matches_slm": fsm_matches,
            "slm_valid": reference is not None,
            "served_matches_gold": gold_ok,
            "error": error,
        })
        if progress_every and (index + 1) % progress_every == 0:
            recent = logs[-progress_every:]
            log(f"{index + 1}/{len(rows)} requests | FSM share (last {progress_every}) "
                f"{sum(r['primary'] == 'fsm' for r in recent) / len(recent):.0%} | "
                f"new SLM calls {teacher.calls} | {time.perf_counter() - started:.0f}s")
    return logs, trusted_rows, promoted_after, group_of


def self_consistency(engine, teacher, validate_raw, decisions_equal, rows, sample=30, seed=7):
    """Re-ask the SLM, uncached, for a sample of already-answered payloads."""
    import random

    rng = random.Random(seed)
    unique = {}
    for row in rows:
        key = payload_key(row["input"])
        if key in teacher.cache:
            unique.setdefault(key, row["input"])
    chosen = rng.sample(sorted(unique), min(sample, len(unique)))
    agree = 0
    for key in chosen:
        payload = unique[key]
        try:
            first = validate_raw(teacher.cache[key], payload)
            second = validate_raw(engine(payload), payload)
            agree += bool(decisions_equal(first, second))
        except Exception:
            pass
    return agree / len(chosen) if chosen else None, len(chosen)


def summarize(logs, trusted_rows, promoted_after, group_of, rows, consistency):
    import statistics

    n_batches = max(r["batch"] for r in rows)
    batch_ends = [max(l["step"] for l in logs if l["batch"] == b) for b in range(1, n_batches + 1)]
    last = [l for l in logs if l["batch"] == n_batches]
    unlearnable_share = sum(l["pattern_group"] not in LEARNABLE for l in last) / len(last)
    groups = {}
    for group in GROUPS:
        members = sorted(p for p, g in group_of.items() if g == group)
        after = [promoted_after[p] for p in members if p in promoted_after]
        end1 = trusted_rows[batch_ends[0] - 1]
        groups[group] = {
            "patterns": len(members),
            "promoted": len(after),
            "promoted_by_end_of_batch_1": sum(end1[p] for p in members),
            "median_observations_to_promotion": statistics.median(after) if after else None,
        }
    return {
        "requests": len(logs),
        "batches": n_batches,
        "batch_ends": batch_ends,
        "slm_floor": unlearnable_share,
        "fsm_ceiling": 1 - unlearnable_share,
        "slm_self_consistency": consistency,
        "final_batch_fsm_share": sum(l["primary"] == "fsm" for l in last) / len(last),
        "final_batch_fsm_matches_slm": sum(l["fsm_matches_slm"] for l in last) / len(last),
        "fsm_served_agreement": (
            sum(l["fsm_matches_slm"] for l in logs if l["fsm_answered"])
            / max(1, sum(l["fsm_answered"] for l in logs))
        ),
        "served_matches_gold": sum(l["served_matches_gold"] for l in logs) / len(logs),
        "slm_invalid_outputs": sum(not l["slm_valid"] for l in logs),
        "groups": groups,
    }


def make_figure(logs, trusted_rows, group_of, summary, window=75):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.DataFrame(logs)
    steps = df["step"].to_numpy()
    slm_c, fsm_c = "#D9822B", "#3A68AE"
    purple, dark_g, light_g = "#7B4FB5", "#3C8D4B", "#93C99A"
    roll = lambda s: s.astype(float).rolling(window, min_periods=window).mean() * 100

    fig, axes = plt.subplots(3, 1, figsize=(12, 13.5), sharex=True)
    fig.suptitle(
        "FSM convergence toward the SLM depends on pattern repetition and SLM consistency",
        x=0.01, ha="left", fontsize=15, fontweight="bold",
    )
    ends = summary["batch_ends"]
    starts = [0] + ends[:-1]
    for ax in axes:
        for end in ends[:-1]:
            ax.axvline(end, color="0.45", linestyle=":", linewidth=1.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 105)
        ax.set_xlim(0, len(df))

    # Panel A
    ax = axes[0]
    ax.plot(steps, roll(df["primary"] != "fsm"), color=slm_c, lw=2.2, label="Routed to SLM")
    ax.plot(steps, roll(df["primary"] == "fsm"), color=fsm_c, lw=2.2, label="Routed to FSM")
    floor = summary["slm_floor"] * 100
    ax.axhline(floor, color=slm_c, ls="--", lw=1.3,
               label=f"SLM floor ≈ {floor:.0f}% (novel + not generalizable)")
    ax.set_title("A.  SLM routing falls and FSM routing rises as patterns repeat",
                 loc="left", fontsize=13, fontweight="bold", pad=22)
    ax.set_ylabel(f"Share of requests (%)\nrolling {window}")
    for i, (a, b) in enumerate(zip(starts, ends), 1):
        ax.text((a + b) / 2, 1.02, f"batch {i}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color="0.4", style="italic", fontsize=11)
    g = summary["groups"]
    converged = all(g[k]["promoted_by_end_of_batch_1"] == g[k]["patterns"] for k in ("routine", "risky"))
    note = (f"End of batch 1: {g['routine']['promoted_by_end_of_batch_1']}/{g['routine']['patterns']} routine"
            f" and {g['risky']['promoted_by_end_of_batch_1']}/{g['risky']['patterns']} risky\n"
            f"patterns promoted — {'converged' if converged else 'not fully converged'}")
    fsm_roll = roll(df["primary"] == "fsm")
    y_end = fsm_roll.iloc[ends[0] - 1]
    if pd.notna(y_end):
        ax.annotate(note, xy=(ends[0], y_end), xytext=(ends[0] + 0.04 * len(df), max(8, y_end - 25)),
                    fontsize=10.5, arrowprops=dict(arrowstyle="->", color="0.35"),
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.legend(loc="center right", fontsize=11, framealpha=0.85, edgecolor="none")

    # Panel B
    ax = axes[1]
    trusted = pd.DataFrame(trusted_rows)
    colors = {"routine": fsm_c, "risky": purple, "unlearnable": slm_c}
    for group in ("routine", "risky", "unlearnable"):
        members = [p for p, gg in group_of.items() if gg == group and p in trusted]
        share = trusted[members].mean(axis=1) * 100
        info = g[group]
        if group == "unlearnable":
            text = (f"Not generalizable (n={info['patterns']}) — "
                    + ("never promoted" if info["promoted"] == 0 else f"{info['promoted']} promoted"))
        else:
            med = info["median_observations_to_promotion"]
            text = (f"{group.capitalize()} (n={info['patterns']}) — "
                    + (f"promoted after ≈{med:.0f} consistent obs" if med is not None else "not promoted yet"))
        ax.step(steps, share, where="post", color=colors[group], lw=2.2, label=text)
    ax.set_title("B.  Patterns become trusted only after repeated, consistent SLM answers",
                 loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel("Patterns trusted by FSM (%)")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.85, edgecolor="none")

    # Panel C
    ax = axes[2]
    learn = df["pattern_group"].isin(LEARNABLE)
    stable = df["fsm_matches_slm"].where(learn)
    stable_roll = stable.dropna().astype(float).rolling(window, min_periods=window).mean() * 100
    ax.plot(df.loc[learn, "step"], stable_roll, color=dark_g, lw=2.6,
            label="FSM matches SLM — repeated, stable patterns")
    ax.plot(steps, roll(df["fsm_matches_slm"]), color=light_g, lw=2.2,
            label="FSM matches SLM — all requests")
    served = df[df["fsm_answered"]]
    if len(served) >= 10:
        served_roll = served["fsm_matches_slm"].astype(float).rolling(
            window, min_periods=min(window, 10)).mean() * 100
        ax.plot(served["step"], served_roll, color="0.3", ls=":", lw=1.4,
                label="Only requests the FSM answered")
    consistency = summary["slm_self_consistency"]
    if consistency is not None:
        ax.axhline(consistency * 100, color=slm_c, ls="--", lw=1.4,
                   label=f"SLM self-consistency ≈ {consistency * 100:.0f}% (target)")
    ceiling = summary["fsm_ceiling"] * 100
    ax.axhline(ceiling, color=dark_g, ls=":", lw=1.2, alpha=0.8)
    ax.text(len(df) * 0.01, ceiling + 1.5,
            f"ceiling ≈ {ceiling:.0f}%: novel + not-generalizable requests can't be learned",
            color=dark_g, fontsize=10.5)
    ax.set_title("C.  FSM answers converge toward the SLM (starts at 0: FSM has no answers yet)",
                 loc="left", fontsize=13, fontweight="bold")
    ax.set_ylabel(f"FSM answer = SLM answer (%)\nrolling {window}")
    ax.set_xlabel("Requests processed", fontsize=12)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.85, edgecolor="none")

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig
