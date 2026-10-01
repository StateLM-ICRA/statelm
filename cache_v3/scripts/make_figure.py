#!/usr/bin/env python3
"""Three-panel Fig. 2 from a replay's turn_log.jsonl and metrics.json.

Abscissa of every panel: deployment turn index t (1..T) over the real stream.
Panel A ordinate: share of the last W turns served by each route (%).
Panel B ordinate: number of FSM patterns (active = executable, candidate =
stored but not yet admitted).
Panel C ordinate: cumulative rates (%): FSM-served decisions that equal the
SLM's checked decision; served decisions whose action equals the reference
action; review turns whose served failure type equals the annotated type.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#d9d8d4"
BLUE = "#2a78d6"    # FSM route / active patterns
ORANGE = "#eb6834"  # SLM route
AQUA = "#1baf7a"    # agreement
VIOLET = "#4a3aa7"  # candidates / reference accuracy


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rolling(flags, window):
    out = []
    for index in range(len(flags)):
        chunk = flags[max(0, index - window + 1): index + 1]
        out.append(100.0 * sum(chunk) / len(chunk))
    return out


def cumulative(values):
    out, num, den = [], 0, 0
    for value in values:
        if value is not None:
            num += int(bool(value))
            den += 1
        out.append(100.0 * num / den if den else float("nan"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True, help="folder with turn_log.jsonl and metrics.json")
    parser.add_argument("--out", type=Path, required=True, help="output image path (.png or .pdf)")
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    logs = read_jsonl(args.replay / "turn_log.jsonl")
    metrics = json.loads((args.replay / "metrics.json").read_text())
    t = [log["position"] for log in logs]
    fsm = rolling([log["source"] == "fsm" for log in logs], args.window)
    slm = rolling([log["source"] != "fsm" for log in logs], args.window)
    active = [log["active_patterns"] for log in logs]
    candidates = [log["candidate_patterns"] for log in logs]
    agree = cumulative([log["fsm_matches_checked_slm"] for log in logs])
    action_ok = cumulative([log["served_action_ok"] for log in logs])
    failure_ok = cumulative([log["served_failure_type_ok"] if log["kind"] == "review" else None for log in logs])
    n_real = sum(1 for log in logs if log["origin"] == "real_failure_trial")
    n_aug = sum(1 for log in logs if log["origin"] != "real_failure_trial")
    n_participants = len({log["participant_id"] for log in logs if log["participant_id"]})
    seed_patterns = metrics["seed"]["seed_patterns"]
    cfg = metrics["config"]

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK,
        "ytick.color": INK, "text.color": INK, "axes.titlecolor": INK, "legend.frameon": False,
    })
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 10.5), sharex=True, facecolor=SURFACE)
    title = args.title or (
        f"FSM growth on the real deployment stream: {n_real} turns from real failure trials "
        f"({n_participants} participants) plus {n_aug} augmented requests"
    )
    fig.suptitle(title, fontsize=11.5, fontweight="bold", color=INK, y=0.995)

    ax = axes[0]
    ax.set_facecolor(SURFACE)
    ax.plot(t, slm, color=ORANGE, linewidth=2, label="Routed to SLM")
    ax.plot(t, fsm, color=BLUE, linewidth=2, label="Served by the FSM (no SLM call)")
    ax.set_ylim(0, 100)
    ax.set_ylabel(f"Share of turns (%)\nrolling window of {args.window} turns")
    ax.set_title("A. Serving routes during deployment", loc="left", fontsize=10.5)
    offset = 3.0 if abs(fsm[-1] - slm[-1]) < 8 else 0.0
    ax.text(t[-1], fsm[-1] + offset, f" {fsm[-1]:.0f}%", color=BLUE, va="center", fontsize=9)
    ax.text(t[-1], slm[-1] - offset, f" {slm[-1]:.0f}%", color=ORANGE, va="center", fontsize=9)
    ax.set_xlim(0, t[-1] * 1.06)
    ax.legend(loc="upper center", ncol=2)

    ax = axes[1]
    ax.set_facecolor(SURFACE)
    ax.plot(t, active, color=BLUE, linewidth=2, label="Active patterns (executable)")
    ax.plot(t, candidates, color=VIOLET, linewidth=2, label="Candidates (stored, not yet admitted)")
    ax.axhline(seed_patterns, color=INK2, linewidth=1, linestyle=":", label=f"FSM_0 seed patterns ({seed_patterns})")
    ax.set_ylabel("Number of patterns")
    ax.set_title(
        f"B. Pattern learning (admission after {cfg['minimum_observations']} distinct observations, "
        f"reliability >= {cfg['admission_threshold']})", loc="left", fontsize=10.5)
    ax.legend(loc="upper left")

    ax = axes[2]
    ax.set_facecolor(SURFACE)
    ax.plot(t, agree, color=AQUA, linewidth=2, label="FSM-served decision equals the SLM's checked decision")
    ax.plot(t, action_ok, color=VIOLET, linewidth=2, linestyle="--", label="Served action equals the reference action")
    ax.plot(t, failure_ok, color=ORANGE, linewidth=2, linestyle="-.", label="Review turns: served failure type equals the annotated type")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Cumulative rate (%)")
    ax.set_xlabel("Deployment turn index t (real trials in participant order; request turn then review turn per trial)")
    ax.set_title("C. Correctness of what is served", loc="left", fontsize=10.5)
    ax.legend(loc="lower right")

    for ax in axes:
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    summary = metrics["summary"]
    foot = (
        f"Embedding {cfg['embedder']}; tau_z={cfg['merge_similarity']}, tau_r={cfg['routing_similarity']}, "
        f"tau_c={cfg['admission_threshold']}. Every SLM decision is the trained SLM's own output (cached once, no audits). "
        f"Overall FSM share {100*summary['fsm_share_overall']:.1f}%, last window {100*summary['fsm_share_last_window']:.1f}%; "
        f"FSM-served agreement with the SLM {100*(summary['fsm_served_exact_agreement_with_checked_slm'] or 0):.1f}%; "
        f"wrong item served: {summary['wrong_item_served']}."
    )
    fig.text(0.01, 0.005, foot, fontsize=7.5, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    if args.out.suffix.lower() != ".pdf":
        fig.savefig(args.out.with_suffix(".pdf"), facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
