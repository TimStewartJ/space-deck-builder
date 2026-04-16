"""Post-hoc analysis of collected replay data.

Reads gzipped JSONL replay files produced by ReplayCollector and
generates aggregate statistics and charts about the agent's behavior.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass

from src.analysis.replay_collector import ReplayCollector, GameReplay, DecisionRecord


def _game_phase(turn: int) -> str:
    """Classify a turn number into a game phase."""
    if turn <= 5:
        return "early"
    elif turn <= 15:
        return "mid"
    else:
        return "late"


PHASES = ["early", "mid", "late"]


@dataclass
class AnalysisResult:
    """Container for all analysis outputs."""
    num_games: int
    num_decisions: int
    win_rate: float
    avg_game_length: float
    avg_game_length_wins: float
    avg_game_length_losses: float
    buy_table: dict         # card_name -> {phase -> {bought, affordable, rate}}
    action_dist: dict       # phase -> {action_type -> count}
    economy_curves: dict    # turn -> {avg_trade, avg_combat, avg_health, count}
    entropy_by_phase: dict  # phase -> {mean, min, max}
    value_accuracy: dict    # {wins_mean_value, losses_mean_value}


def analyze_replays(
    path: str,
    output_dir: str | None = None,
) -> AnalysisResult:
    """Run full analysis on a replay file and print results.

    Args:
        path: Path to the gzipped JSONL replay file.
        output_dir: Optional directory for saving charts. If None, charts
            are saved to ``analysis/`` next to the replay file.

    Returns:
        AnalysisResult with all computed statistics.
    """
    meta, replays = ReplayCollector.load(path)
    card_names = meta["card_names"]
    num_cards = len(card_names)

    if not replays:
        print("No replays found.")
        return AnalysisResult(
            num_games=0, num_decisions=0, win_rate=0, avg_game_length=0,
            avg_game_length_wins=0, avg_game_length_losses=0,
            buy_table={}, action_dist={}, economy_curves={},
            entropy_by_phase={}, value_accuracy={},
        )

    # ---------- Aggregate accumulators ----------
    wins = sum(1 for r in replays if r.winner == "PPO")
    total_decisions = sum(len(r.decisions) for r in replays)
    win_lengths = [r.total_turns for r in replays if r.winner == "PPO"]
    loss_lengths = [r.total_turns for r in replays if r.winner != "PPO"]

    # Buy analysis: per card, per phase — bought count and affordable count
    buy_data: dict[int, dict[str, dict[str, int]]] = defaultdict(
        lambda: {p: {"bought": 0, "affordable": 0} for p in PHASES}
    )

    # Action type distribution per phase
    action_counts: dict[str, dict[str, int]] = {p: defaultdict(int) for p in PHASES}

    # Economy curves: per turn
    econ: dict[int, dict[str, list]] = defaultdict(
        lambda: {"trade": [], "combat": [], "health": []}
    )

    # Entropy per phase
    entropy_data: dict[str, list[float]] = {p: [] for p in PHASES}

    # Value estimates for wins vs losses
    value_wins: list[float] = []
    value_losses: list[float] = []

    for replay in replays:
        is_win = replay.winner == "PPO"
        for d in replay.decisions:
            phase = _game_phase(d.turn)

            # Action distribution
            action_counts[phase][d.action_type] += 1

            # Economy curves
            econ[d.turn]["trade"].append(d.trade)
            econ[d.turn]["combat"].append(d.combat)
            econ[d.turn]["health"].append(d.player_health)

            # Entropy
            entropy_data[phase].append(d.policy_entropy)

            # Value estimates
            if is_win:
                value_wins.append(d.value_estimate)
            else:
                value_losses.append(d.value_estimate)

            # Buy analysis: which cards were affordable and which was bought
            for cid in d.buyable_card_ids:
                if 0 <= cid < num_cards:
                    buy_data[cid][phase]["affordable"] += 1
            if d.action_type == "BUY_CARD" and d.action_card_id is not None:
                cid = d.action_card_id
                if 0 <= cid < num_cards:
                    buy_data[cid][phase]["bought"] += 1

    # ---------- Build result structures ----------

    # Buy priority table
    buy_table = {}
    for cid in sorted(buy_data.keys()):
        if cid >= num_cards:
            continue
        name = card_names[cid]
        entry = {}
        total_bought = 0
        total_affordable = 0
        for phase in PHASES:
            b = buy_data[cid][phase]["bought"]
            a = buy_data[cid][phase]["affordable"]
            total_bought += b
            total_affordable += a
            entry[phase] = {
                "bought": b,
                "affordable": a,
                "rate": b / a if a > 0 else 0.0,
            }
        entry["overall"] = {
            "bought": total_bought,
            "affordable": total_affordable,
            "rate": total_bought / total_affordable if total_affordable > 0 else 0.0,
        }
        buy_table[name] = entry

    # Action distribution (as percentages)
    action_dist = {}
    for phase in PHASES:
        total = sum(action_counts[phase].values())
        action_dist[phase] = {
            k: {"count": v, "pct": v / total * 100 if total > 0 else 0}
            for k, v in sorted(action_counts[phase].items())
        }

    # Economy curves
    economy_curves = {}
    for turn in sorted(econ.keys()):
        d = econ[turn]
        economy_curves[turn] = {
            "avg_trade": sum(d["trade"]) / len(d["trade"]),
            "avg_combat": sum(d["combat"]) / len(d["combat"]),
            "avg_health": sum(d["health"]) / len(d["health"]),
            "count": len(d["trade"]),
        }

    # Entropy by phase
    entropy_by_phase = {}
    for phase in PHASES:
        vals = entropy_data[phase]
        if vals:
            entropy_by_phase[phase] = {
                "mean": sum(vals) / len(vals),
                "min": min(vals),
                "max": max(vals),
            }
        else:
            entropy_by_phase[phase] = {"mean": 0, "min": 0, "max": 0}

    # Value accuracy
    value_accuracy = {
        "wins_mean_value": sum(value_wins) / len(value_wins) if value_wins else 0,
        "losses_mean_value": sum(value_losses) / len(value_losses) if value_losses else 0,
    }

    result = AnalysisResult(
        num_games=len(replays),
        num_decisions=total_decisions,
        win_rate=wins / len(replays) if replays else 0,
        avg_game_length=sum(r.total_turns for r in replays) / len(replays),
        avg_game_length_wins=sum(win_lengths) / len(win_lengths) if win_lengths else 0,
        avg_game_length_losses=sum(loss_lengths) / len(loss_lengths) if loss_lengths else 0,
        buy_table=buy_table,
        action_dist=action_dist,
        economy_curves=economy_curves,
        entropy_by_phase=entropy_by_phase,
        value_accuracy=value_accuracy,
    )

    _print_report(result)
    _save_charts(result, output_dir or os.path.join(os.path.dirname(path), "analysis"))

    return result


def _print_report(r: AnalysisResult) -> None:
    """Print a human-readable analysis summary to stdout."""
    print("\n" + "=" * 70)
    print("REPLAY ANALYSIS REPORT")
    print("=" * 70)

    print(f"\nGames: {r.num_games}  |  Decisions: {r.num_decisions:,}  |  "
          f"Win rate: {r.win_rate:.1%}")
    print(f"Avg game length: {r.avg_game_length:.1f} turns "
          f"(wins: {r.avg_game_length_wins:.1f}, losses: {r.avg_game_length_losses:.1f})")

    # Value estimate accuracy
    v = r.value_accuracy
    print(f"\nValue estimates — wins avg: {v['wins_mean_value']:.3f}, "
          f"losses avg: {v['losses_mean_value']:.3f}")

    # Entropy by phase
    print(f"\nPolicy entropy by phase:")
    print(f"  {'Phase':<8} {'Mean':>8} {'Min':>8} {'Max':>8}")
    for phase in PHASES:
        e = r.entropy_by_phase[phase]
        print(f"  {phase:<8} {e['mean']:>8.3f} {e['min']:>8.3f} {e['max']:>8.3f}")

    # Action distribution
    print(f"\nAction distribution (%):")
    all_types = sorted(set(
        t for phase_data in r.action_dist.values() for t in phase_data
    ))
    header = f"  {'Phase':<8}" + "".join(f" {t[:12]:>12}" for t in all_types)
    print(header)
    for phase in PHASES:
        row = f"  {phase:<8}"
        for t in all_types:
            pct = r.action_dist[phase].get(t, {}).get("pct", 0)
            row += f" {pct:>11.1f}%"
        print(row)

    # Buy priority table — top 20 by overall buy rate
    print(f"\nBuy priority (top 20 by buy rate when affordable):")
    sorted_cards = sorted(
        r.buy_table.items(),
        key=lambda x: x[1]["overall"]["rate"],
        reverse=True,
    )[:20]
    print(f"  {'Card':<25} {'Early':>8} {'Mid':>8} {'Late':>8} {'Overall':>8}  {'Bought':>7}")
    for name, data in sorted_cards:
        overall = data["overall"]
        if overall["affordable"] == 0:
            continue
        row = f"  {name:<25}"
        for phase in PHASES:
            rate = data[phase]["rate"]
            row += f" {rate:>7.1%}"
        row += f" {overall['rate']:>7.1%}"
        row += f"  {overall['bought']:>7}"
        print(row)

    print("\n" + "=" * 70)


def _save_charts(r: AnalysisResult, output_dir: str) -> None:
    """Save matplotlib charts to the output directory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping chart generation.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Economy curves
    turns = sorted(r.economy_curves.keys())
    if turns:
        trades = [r.economy_curves[t]["avg_trade"] for t in turns]
        combats = [r.economy_curves[t]["avg_combat"] for t in turns]
        healths = [r.economy_curves[t]["avg_health"] for t in turns]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        ax1.plot(turns, trades, label="Avg Trade", color="gold", linewidth=2)
        ax1.plot(turns, combats, label="Avg Combat", color="red", linewidth=2)
        ax1.set_xlabel("Turn")
        ax1.set_ylabel("Resources")
        ax1.set_title("Economy Curves — Trade & Combat per Turn")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(turns, healths, label="Avg Health", color="green", linewidth=2)
        ax2.set_xlabel("Turn")
        ax2.set_ylabel("Health")
        ax2.set_title("Health over Time")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = os.path.join(output_dir, "economy_curves.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved: {chart_path}")

    # Chart 2: Action distribution by phase (stacked bar)
    all_types = sorted(set(
        t for phase_data in r.action_dist.values() for t in phase_data
    ))
    if all_types:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = range(len(PHASES))
        bottom = [0.0] * len(PHASES)

        colors = plt.cm.Set3(range(len(all_types)))
        for tidx, action_type in enumerate(all_types):
            values = [
                r.action_dist[phase].get(action_type, {}).get("pct", 0)
                for phase in PHASES
            ]
            ax.bar(x, values, bottom=bottom, label=action_type[:15],
                   color=colors[tidx % len(colors)])
            bottom = [b + v for b, v in zip(bottom, values)]

        ax.set_xticks(x)
        ax.set_xticklabels(PHASES)
        ax.set_ylabel("% of Decisions")
        ax.set_title("Action Type Distribution by Game Phase")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "action_distribution.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved: {chart_path}")

    # Chart 3: Buy rate heatmap (top 15 cards)
    sorted_cards = sorted(
        r.buy_table.items(),
        key=lambda x: x[1]["overall"]["rate"],
        reverse=True,
    )[:15]
    if sorted_cards:
        fig, ax = plt.subplots(figsize=(10, 8))
        card_labels = [name for name, _ in sorted_cards]
        data_matrix = [
            [entry[phase]["rate"] * 100 for phase in PHASES]
            for _, entry in sorted_cards
        ]

        im = ax.imshow(data_matrix, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(PHASES)))
        ax.set_xticklabels(PHASES)
        ax.set_yticks(range(len(card_labels)))
        ax.set_yticklabels(card_labels)
        ax.set_title("Buy Rate When Affordable (%)")

        # Annotate cells
        for i in range(len(card_labels)):
            for j in range(len(PHASES)):
                val = data_matrix[i][j]
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        color="white" if val > 50 else "black", fontsize=9)

        plt.colorbar(im, ax=ax, label="Buy Rate %")
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "buy_rate_heatmap.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved: {chart_path}")

    # Chart 4: Policy entropy over game phases
    phase_entropies = [r.entropy_by_phase[p]["mean"] for p in PHASES]
    if any(e > 0 for e in phase_entropies):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(PHASES, phase_entropies, color=["#4CAF50", "#FF9800", "#F44336"])
        ax.set_ylabel("Mean Policy Entropy")
        ax.set_title("Policy Entropy by Game Phase")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        chart_path = os.path.join(output_dir, "entropy_by_phase.png")
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved: {chart_path}")
