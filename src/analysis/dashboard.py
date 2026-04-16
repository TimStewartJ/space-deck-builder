"""Interactive HTML dashboard for replay analysis.

Generates a self-contained HTML file with Plotly.js charts organized
into tabs: Overview, Economy, Decisions, Cards, Combat, and Failures.
No server needed — opens directly in any browser.
"""

from __future__ import annotations

import json
import html
import os
from collections import defaultdict, Counter

from src.analysis.replay_collector import ReplayCollector, GameReplay, DecisionRecord


def _game_phase(turn: int) -> str:
    if turn <= 5:
        return "early"
    elif turn <= 15:
        return "mid"
    return "late"


PHASES = ["early", "mid", "late"]


def generate_dashboard(
    replay_path: str,
    output_path: str | None = None,
    model_info: str = "",
) -> str:
    """Generate an interactive HTML dashboard from replay data.

    Args:
        replay_path: Path to gzipped JSONL replay file.
        output_path: Where to write the HTML. Defaults to same dir as replay.
        model_info: Optional model description string for the header.

    Returns:
        The output path written.
    """
    meta, replays = ReplayCollector.load(replay_path)
    card_names = meta["card_names"]
    num_cards = len(card_names)

    if not replays:
        print("No replays found — skipping dashboard generation.")
        return None

    # Load card metadata for faction-aware analysis
    card_factions = _load_card_factions(card_names)

    stats = _compute_stats(replays, card_names, num_cards, card_factions)
    html_content = _build_html(stats, card_names, model_info, replay_path)

    if output_path is None:
        base = os.path.splitext(os.path.splitext(replay_path)[0])[0]
        output_path = base + "_dashboard.html"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return output_path


def _load_card_factions(card_names: list[str]) -> dict[str, str]:
    """Load faction for each card name from the card registry.

    Returns a dict mapping card_name → faction display string
    (e.g. "Blob", "Machine Cult", "Unaligned").
    """
    try:
        from src.config import DataConfig
        from src.cards.factions import Faction, FACTION_NAMES
        data_cfg = DataConfig()
        cards = data_cfg.load_cards()
        faction_map = {}
        seen = set()
        for c in cards:
            if c.name in seen:
                continue
            seen.add(c.name)
            factions = [FACTION_NAMES[f] for f in Faction if f in c.faction and f in FACTION_NAMES]
            faction_map[c.name] = factions[0] if len(factions) == 1 else ("Multi" if factions else "Unaligned")
        # Starter cards
        for name in card_names:
            if name not in faction_map:
                faction_map[name] = "Unaligned"
        return faction_map
    except Exception:
        return {name: "Unknown" for name in card_names}


def _compute_stats(replays: list[GameReplay], card_names: list[str], num_cards: int,
                   card_factions: dict[str, str] | None = None) -> dict:
    """Compute all dashboard metrics from replay data."""
    stats = {}

    # ── Overview ────────────────────────────────────────────────────
    total_games = len(replays)
    wins = [r for r in replays if r.winner == "PPO"]
    losses = [r for r in replays if r.winner != "PPO"]
    win_lengths = [r.total_turns for r in wins]
    loss_lengths = [r.total_turns for r in losses]
    all_lengths = [r.total_turns for r in replays]
    total_decisions = sum(len(r.decisions) for r in replays)

    opp_stats = defaultdict(lambda: {"wins": 0, "total": 0, "lengths": []})
    for r in replays:
        opp_stats[r.opponent_type]["total"] += 1
        opp_stats[r.opponent_type]["lengths"].append(r.total_turns)
        if r.winner == "PPO":
            opp_stats[r.opponent_type]["wins"] += 1

    stats["overview"] = {
        "total_games": total_games,
        "total_decisions": total_decisions,
        "win_rate": len(wins) / total_games if total_games else 0,
        "avg_length": sum(all_lengths) / len(all_lengths) if all_lengths else 0,
        "avg_length_wins": sum(win_lengths) / len(win_lengths) if win_lengths else 0,
        "avg_length_losses": sum(loss_lengths) / len(loss_lengths) if loss_lengths else 0,
        "win_lengths": win_lengths,
        "loss_lengths": loss_lengths,
        "opponent_stats": {k: dict(v) for k, v in opp_stats.items()},
    }

    # ── Economy curves with percentile bands ────────────────────────
    econ_all = defaultdict(lambda: {"trade": [], "combat": [], "health": []})
    econ_wins = defaultdict(lambda: {"trade": [], "combat": [], "health": []})
    econ_losses = defaultdict(lambda: {"trade": [], "combat": [], "health": []})

    for r in replays:
        is_win = r.winner == "PPO"
        for d in r.decisions:
            econ_all[d.turn]["trade"].append(d.trade)
            econ_all[d.turn]["combat"].append(d.combat)
            econ_all[d.turn]["health"].append(d.player_health)
            target = econ_wins if is_win else econ_losses
            target[d.turn]["trade"].append(d.trade)
            target[d.turn]["combat"].append(d.combat)
            target[d.turn]["health"].append(d.player_health)

    def _summarize_econ(econ_data):
        result = {}
        for turn in sorted(econ_data.keys()):
            d = econ_data[turn]
            n = len(d["trade"])
            if n == 0:
                continue
            summary = {}
            for key in ["trade", "combat", "health"]:
                vals = sorted(d[key])
                summary[key] = {
                    "mean": sum(vals) / n,
                    "median": vals[n // 2],
                    "p10": vals[max(0, int(n * 0.1))],
                    "p90": vals[min(n - 1, int(n * 0.9))],
                    "count": n,
                }
            result[turn] = summary
        return result

    stats["economy"] = {
        "all": _summarize_econ(econ_all),
        "wins": _summarize_econ(econ_wins),
        "losses": _summarize_econ(econ_losses),
    }

    # ── Action distribution ─────────────────────────────────────────
    action_counts = {p: defaultdict(int) for p in PHASES}
    action_counts_wins = {p: defaultdict(int) for p in PHASES}
    action_counts_losses = {p: defaultdict(int) for p in PHASES}

    for r in replays:
        is_win = r.winner == "PPO"
        for d in r.decisions:
            phase = _game_phase(d.turn)
            action_counts[phase][d.action_type] += 1
            target = action_counts_wins if is_win else action_counts_losses
            target[phase][d.action_type] += 1

    def _action_pcts(counts):
        result = {}
        for phase in PHASES:
            total = sum(counts[phase].values())
            result[phase] = {
                k: round(v / total * 100, 2) if total > 0 else 0
                for k, v in sorted(counts[phase].items())
            }
        return result

    stats["actions"] = {
        "all": _action_pcts(action_counts),
        "wins": _action_pcts(action_counts_wins),
        "losses": _action_pcts(action_counts_losses),
    }

    # ── Entropy ─────────────────────────────────────────────────────
    entropy_by_turn = defaultdict(list)
    entropy_by_turn_wins = defaultdict(list)
    entropy_by_turn_losses = defaultdict(list)
    entropy_by_phase = {p: [] for p in PHASES}

    for r in replays:
        is_win = r.winner == "PPO"
        for d in r.decisions:
            entropy_by_turn[d.turn].append(d.policy_entropy)
            (entropy_by_turn_wins if is_win else entropy_by_turn_losses)[d.turn].append(d.policy_entropy)
            entropy_by_phase[_game_phase(d.turn)].append(d.policy_entropy)

    stats["entropy"] = {
        "by_turn": {
            t: {"mean": sum(v) / len(v), "count": len(v)}
            for t, v in sorted(entropy_by_turn.items())
        },
        "by_turn_wins": {
            t: {"mean": sum(v) / len(v)}
            for t, v in sorted(entropy_by_turn_wins.items()) if v
        },
        "by_turn_losses": {
            t: {"mean": sum(v) / len(v)}
            for t, v in sorted(entropy_by_turn_losses.items()) if v
        },
        "by_phase": {
            p: {
                "mean": sum(v) / len(v) if v else 0,
                "min": min(v) if v else 0,
                "max": max(v) if v else 0,
            }
            for p, v in entropy_by_phase.items()
        },
    }

    # ── Value estimates ─────────────────────────────────────────────
    value_trajectories = []
    value_wins_all = []
    value_losses_all = []

    for r in replays:
        is_win = r.winner == "PPO"
        traj = {"game_id": r.game_id, "winner": r.winner, "opponent": r.opponent_type, "values": []}
        for d in r.decisions:
            traj["values"].append({"turn": d.turn, "value": round(d.value_estimate, 4)})
            (value_wins_all if is_win else value_losses_all).append(d.value_estimate)
        value_trajectories.append(traj)

    stats["values"] = {
        "trajectories": value_trajectories[:50],
        "wins_mean": sum(value_wins_all) / len(value_wins_all) if value_wins_all else 0,
        "losses_mean": sum(value_losses_all) / len(value_losses_all) if value_losses_all else 0,
    }

    # ── Buy analysis ────────────────────────────────────────────────
    buy_data = defaultdict(lambda: {p: {"bought": 0, "affordable": 0} for p in PHASES})
    buy_timing = defaultdict(list)
    deck_wins = defaultdict(list)
    deck_losses = defaultdict(list)
    trade_efficiency = defaultdict(lambda: {"available": [], "spent": []})

    for r in replays:
        is_win = r.winner == "PPO"
        game_buys = Counter()
        prev_trade = 0
        for d in r.decisions:
            phase = _game_phase(d.turn)
            for cid in d.buyable_card_ids:
                if 0 <= cid < num_cards:
                    buy_data[cid][phase]["affordable"] += 1
            if d.action_type == "BUY_CARD" and d.action_card_id is not None:
                cid = d.action_card_id
                if 0 <= cid < num_cards:
                    buy_data[cid][phase]["bought"] += 1
                    buy_timing[cid].append(d.turn)
                    game_buys[cid] += 1
            if d.action_type == "END_TURN":
                trade_efficiency[d.turn]["available"].append(prev_trade + d.trade)
                trade_efficiency[d.turn]["spent"].append(prev_trade)
            prev_trade = d.trade

        target = deck_wins if is_win else deck_losses
        for cid, count in game_buys.items():
            target[cid].append(count)

    buy_table = {}
    for cid in sorted(buy_data.keys()):
        if cid >= num_cards:
            continue
        total_b = sum(buy_data[cid][p]["bought"] for p in PHASES)
        total_a = sum(buy_data[cid][p]["affordable"] for p in PHASES)
        if total_a == 0:
            continue
        entry = {"name": card_names[cid], "total_bought": total_b}
        for p in PHASES:
            a = buy_data[cid][p]["affordable"]
            b = buy_data[cid][p]["bought"]
            entry[p] = round(b / a * 100, 1) if a > 0 else 0
        entry["overall"] = round(total_b / total_a * 100, 1)
        buy_table[card_names[cid]] = entry

    buy_timing_summary = {}
    for cid, turns in buy_timing.items():
        if cid < num_cards and turns:
            buy_timing_summary[card_names[cid]] = {
                "mean_turn": round(sum(turns) / len(turns), 1),
                "count": len(turns),
            }

    num_wins = len(wins) or 1
    num_losses = len(losses) or 1
    deck_comp = {}
    for cid in set(list(deck_wins.keys()) + list(deck_losses.keys())):
        if cid >= num_cards:
            continue
        w = deck_wins.get(cid, [])
        l = deck_losses.get(cid, [])
        deck_comp[card_names[cid]] = {
            "avg_wins": round(sum(w) / num_wins, 2),
            "avg_losses": round(sum(l) / num_losses, 2),
        }

    cooccurrence = defaultdict(int)
    for r in replays:
        if r.winner != "PPO":
            continue
        bought_set = set()
        for d in r.decisions:
            if d.action_type == "BUY_CARD" and d.action_card_id is not None and d.action_card_id < num_cards:
                bought_set.add(card_names[d.action_card_id])
        bought_list = sorted(bought_set)
        for i, a in enumerate(bought_list):
            for b in bought_list[i + 1:]:
                cooccurrence[(a, b)] += 1

    top_pairs = sorted(cooccurrence.items(), key=lambda x: x[1], reverse=True)[:20]

    stats["cards"] = {
        "buy_table": buy_table,
        "buy_timing": buy_timing_summary,
        "deck_composition": deck_comp,
        "co_occurrence": [{"pair": list(pair), "count": cnt} for pair, cnt in top_pairs],
        "trade_efficiency": {
            str(t): {
                "avg_available": round(sum(v["available"]) / len(v["available"]), 2) if v["available"] else 0,
                "avg_spent": round(sum(v["spent"]) / len(v["spent"]), 2) if v["spent"] else 0,
            }
            for t, v in sorted(trade_efficiency.items()) if v["available"]
        },
    }

    # ── Combat analysis ─────────────────────────────────────────────
    combat_per_turn = defaultdict(lambda: {"generated": [], "attacks": 0, "base_attacks": 0})
    base_play_counts = defaultdict(int)
    attack_timing = {"player": [], "base": []}

    for r in replays:
        for d in r.decisions:
            if d.combat > 0:
                combat_per_turn[d.turn]["generated"].append(d.combat)
            if d.action_type == "ATTACK_PLAYER":
                combat_per_turn[d.turn]["attacks"] += 1
                attack_timing["player"].append(d.turn)
            elif d.action_type == "ATTACK_BASE":
                combat_per_turn[d.turn]["base_attacks"] += 1
                attack_timing["base"].append(d.turn)
            for bid in d.bases_in_play_ids:
                if bid < num_cards:
                    base_play_counts[card_names[bid]] += 1

    stats["combat"] = {
        "per_turn": {
            str(t): {
                "avg_combat": round(sum(v["generated"]) / len(v["generated"]), 2) if v["generated"] else 0,
                "attacks": v["attacks"],
                "base_attacks": v["base_attacks"],
            }
            for t, v in sorted(combat_per_turn.items())
        },
        "base_frequency": dict(sorted(base_play_counts.items(), key=lambda x: x[1], reverse=True)[:15]),
        "attack_timing_player": dict(sorted(Counter(attack_timing["player"]).items())),
        "attack_timing_base": dict(sorted(Counter(attack_timing["base"]).items())),
    }

    # ── Failure analysis ────────────────────────────────────────────
    turning_points = []
    health_diff_wins = defaultdict(list)
    health_diff_losses = defaultdict(list)

    for r in replays:
        is_win = r.winner == "PPO"
        prev_val = 0
        for d in r.decisions:
            diff = d.player_health - d.opp_health
            (health_diff_wins if is_win else health_diff_losses)[d.turn].append(diff)
            if not is_win and prev_val >= 0 and d.value_estimate < 0:
                turning_points.append(d.turn)
            prev_val = d.value_estimate

    missed_buys_losses = defaultdict(lambda: {"affordable": 0, "bought": 0})
    missed_buys_wins = defaultdict(lambda: {"affordable": 0, "bought": 0})
    for r in replays:
        is_win = r.winner == "PPO"
        target = missed_buys_wins if is_win else missed_buys_losses
        for d in r.decisions:
            for cid in d.buyable_card_ids:
                if 0 <= cid < num_cards:
                    target[card_names[cid]]["affordable"] += 1
            if d.action_type == "BUY_CARD" and d.action_card_id is not None and d.action_card_id < num_cards:
                target[card_names[d.action_card_id]]["bought"] += 1

    buy_rate_delta = {}
    for name in set(list(missed_buys_wins.keys()) + list(missed_buys_losses.keys())):
        w = missed_buys_wins.get(name, {"affordable": 0, "bought": 0})
        l = missed_buys_losses.get(name, {"affordable": 0, "bought": 0})
        wr = w["bought"] / w["affordable"] if w["affordable"] > 0 else 0
        lr = l["bought"] / l["affordable"] if l["affordable"] > 0 else 0
        if w["affordable"] > 5 or l["affordable"] > 5:
            buy_rate_delta[name] = {
                "win_rate": round(wr * 100, 1),
                "loss_rate": round(lr * 100, 1),
                "delta": round((wr - lr) * 100, 1),
            }

    stats["failures"] = {
        "turning_points": dict(sorted(Counter(turning_points).items())),
        "health_diff_wins": {
            str(t): round(sum(v) / len(v), 1)
            for t, v in sorted(health_diff_wins.items())
        },
        "health_diff_losses": {
            str(t): round(sum(v) / len(v), 1)
            for t, v in sorted(health_diff_losses.items())
        },
        "buy_rate_delta": dict(sorted(
            buy_rate_delta.items(), key=lambda x: x[1]["delta"], reverse=True
        )[:20]),
    }

    # ── Strategy analysis (unbiased) ────────────────────────────────

    if card_factions is None:
        card_factions = {name: "Unknown" for name in card_names}

    faction_labels = ["Blob", "Machine Cult", "Star Empire", "Trade Federation", "Unaligned"]

    # --- 2. Outcome-conditioned win rate lift ---
    # Per card: win rate in games where it was bought vs. not bought
    card_bought_in = defaultdict(lambda: {"bought_wins": 0, "bought_total": 0,
                                          "not_bought_wins": 0, "not_bought_total": 0})
    per_game_buys = []
    for r in replays:
        is_win = r.winner == "PPO"
        bought_ids = set()
        for d in r.decisions:
            if d.action_type == "BUY_CARD" and d.action_card_id is not None and 0 <= d.action_card_id < num_cards:
                bought_ids.add(d.action_card_id)
        per_game_buys.append((bought_ids, is_win, r))
        # Track which cards were available at any point in the game
        available_ids = set()
        for d in r.decisions:
            for cid in d.buyable_card_ids:
                if 0 <= cid < num_cards:
                    available_ids.add(cid)
        for cid in available_ids:
            if cid in bought_ids:
                card_bought_in[cid]["bought_total"] += 1
                if is_win:
                    card_bought_in[cid]["bought_wins"] += 1
            else:
                card_bought_in[cid]["not_bought_total"] += 1
                if is_win:
                    card_bought_in[cid]["not_bought_wins"] += 1

    win_rate_lift = {}
    for cid, data in card_bought_in.items():
        if cid >= num_cards:
            continue
        bt = data["bought_total"]
        nbt = data["not_bought_total"]
        if bt < 3 and nbt < 3:
            continue
        bought_wr = data["bought_wins"] / bt if bt > 0 else 0
        not_bought_wr = data["not_bought_wins"] / nbt if nbt > 0 else 0
        win_rate_lift[card_names[cid]] = {
            "bought_wr": round(bought_wr * 100, 1),
            "not_bought_wr": round(not_bought_wr * 100, 1),
            "lift": round((bought_wr - not_bought_wr) * 100, 1),
            "buy_freq": round(bt / (bt + nbt) * 100, 1),
            "sample": bt + nbt,
        }

    # --- 3. Value estimate reactions on buy ---
    value_deltas = defaultdict(list)
    for r in replays:
        prev_val = None
        for d in r.decisions:
            if d.action_type == "BUY_CARD" and d.action_card_id is not None and 0 <= d.action_card_id < num_cards:
                if prev_val is not None:
                    delta = d.value_estimate - prev_val
                    value_deltas[card_names[d.action_card_id]].append(delta)
            prev_val = d.value_estimate

    value_reactions = {}
    for name, deltas in value_deltas.items():
        if len(deltas) >= 3:
            value_reactions[name] = {
                "mean_delta": round(sum(deltas) / len(deltas), 4),
                "count": len(deltas),
            }

    # --- 4. Emergent archetype discovery (unsupervised k-means) ---
    # Build faction purchase vectors per game
    game_vectors = []
    game_meta = []
    for bought_ids, is_win, r in per_game_buys:
        vec = {f: 0 for f in faction_labels}
        total = 0
        for cid in bought_ids:
            if cid < num_cards:
                faction = card_factions.get(card_names[cid], "Unaligned")
                if faction in vec:
                    vec[faction] += 1
                else:
                    vec["Unaligned"] += 1
                total += 1
        if total > 0:
            fvec = [vec[f] / total for f in faction_labels]
        else:
            fvec = [0.0] * len(faction_labels)
        game_vectors.append(fvec)
        game_meta.append({"game_id": r.game_id, "win": is_win, "opponent": r.opponent_type, "total_bought": total})

    # Simple k-means (pure Python, no sklearn needed)
    archetypes = _kmeans_cluster(game_vectors, game_meta, faction_labels, k=5, max_iter=50)

    # --- 5. Contrastive card analysis ---
    # For top 15 most-bought cards: what else was bought when this card was/wasn't
    all_buy_counts = Counter()
    for bought_ids, _, _ in per_game_buys:
        for cid in bought_ids:
            if cid < num_cards:
                all_buy_counts[card_names[cid]] += 1

    top_cards = [name for name, _ in all_buy_counts.most_common(15)]
    contrastive = {}
    for focal_card in top_cards:
        focal_idx = card_names.index(focal_card) if focal_card in card_names else -1
        if focal_idx < 0:
            continue
        with_focal = Counter()
        without_focal = Counter()
        with_count = 0
        without_count = 0
        for bought_ids, _, _ in per_game_buys:
            bought_names = {card_names[cid] for cid in bought_ids if cid < num_cards}
            if focal_card in bought_names:
                with_count += 1
                for name in bought_names:
                    if name != focal_card:
                        with_focal[name] += 1
            else:
                without_count += 1
                for name in bought_names:
                    without_focal[name] += 1
        # Normalize to rates
        top_with = []
        for name, cnt in with_focal.most_common(8):
            rate_with = cnt / with_count if with_count > 0 else 0
            rate_without = without_focal.get(name, 0) / without_count if without_count > 0 else 0
            top_with.append({"card": name, "rate_with": round(rate_with * 100, 1),
                             "rate_without": round(rate_without * 100, 1),
                             "delta": round((rate_with - rate_without) * 100, 1)})
        contrastive[focal_card] = {
            "games_with": with_count,
            "games_without": without_count,
            "companions": sorted(top_with, key=lambda x: x["delta"], reverse=True)[:6],
        }

    stats["strategy"] = {
        "win_rate_lift": win_rate_lift,
        "value_reactions": dict(sorted(value_reactions.items(), key=lambda x: x[1]["mean_delta"], reverse=True)),
        "archetypes": archetypes,
        "contrastive": contrastive,
        "faction_labels": faction_labels,
    }

    return stats


def _kmeans_cluster(vectors: list[list[float]], meta: list[dict],
                    faction_labels: list[str], k: int = 5, max_iter: int = 50) -> dict:
    """Simple k-means clustering for faction purchase vectors.

    Returns cluster summaries with centroids, sizes, win rates, and labels.
    """
    import random as _rand
    if len(vectors) < k:
        k = max(1, len(vectors))

    n = len(vectors)
    dim = len(vectors[0]) if vectors else 0

    # Initialize centroids with k-means++ style (pick spread-out initial points)
    centroids = [list(vectors[_rand.randint(0, n - 1)])]
    for _ in range(k - 1):
        dists = []
        for v in vectors:
            min_d = min(sum((a - b) ** 2 for a, b in zip(v, c)) for c in centroids)
            dists.append(min_d)
        total = sum(dists)
        if total == 0:
            centroids.append(list(vectors[_rand.randint(0, n - 1)]))
            continue
        r = _rand.random() * total
        cumulative = 0
        for i, d in enumerate(dists):
            cumulative += d
            if cumulative >= r:
                centroids.append(list(vectors[i]))
                break

    assignments = [0] * n
    for _ in range(max_iter):
        # Assign
        changed = False
        for i, v in enumerate(vectors):
            best_c = min(range(k), key=lambda c: sum((a - b) ** 2 for a, b in zip(v, centroids[c])))
            if assignments[i] != best_c:
                changed = True
                assignments[i] = best_c
        if not changed:
            break
        # Update centroids
        for c in range(k):
            members = [vectors[i] for i in range(n) if assignments[i] == c]
            if members:
                centroids[c] = [sum(m[d] for m in members) / len(members) for d in range(dim)]

    # Build cluster summaries
    clusters = []
    for c in range(k):
        member_indices = [i for i in range(n) if assignments[i] == c]
        if not member_indices:
            continue
        wins = sum(1 for i in member_indices if meta[i]["win"])
        size = len(member_indices)
        centroid = centroids[c]

        # Label by dominant faction(s)
        sorted_factions = sorted(zip(faction_labels, centroid), key=lambda x: x[1], reverse=True)
        dominant = [f for f, v in sorted_factions if v > 0.15]
        label = " + ".join(dominant[:2]) if dominant else "Mixed"

        clusters.append({
            "label": label,
            "size": size,
            "win_rate": round(wins / size * 100, 1),
            "centroid": {faction_labels[d]: round(centroid[d] * 100, 1) for d in range(dim)},
            "games": [meta[i]["game_id"] for i in member_indices[:10]],
        })

    # Sort by size descending
    clusters.sort(key=lambda c: c["size"], reverse=True)
    return {"clusters": clusters, "game_assignments": assignments}


# ── HTML template ───────────────────────────────────────────────────

_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #e2e8f0; }
.header { background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 24px 32px; border-bottom: 1px solid #334155; }
.header h1 { font-size: 24px; font-weight: 700; color: #f8fafc; }
.header .subtitle { color: #94a3b8; font-size: 14px; margin-top: 4px; }
.kpi-row { display: flex; gap: 16px; padding: 20px 32px; flex-wrap: wrap; }
.kpi { background: #1e293b; border-radius: 12px; padding: 20px 24px; flex: 1; min-width: 160px; border: 1px solid #334155; }
.kpi .label { color: #94a3b8; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; }
.kpi .value { font-size: 28px; font-weight: 700; color: #f8fafc; margin-top: 4px; }
.kpi .detail { color: #64748b; font-size: 12px; margin-top: 2px; }
.tabs { display: flex; gap: 0; padding: 0 32px; border-bottom: 1px solid #334155; background: #1e293b; }
.tab { padding: 12px 20px; cursor: pointer; color: #94a3b8; font-size: 14px; font-weight: 500; border-bottom: 2px solid transparent; transition: all 0.2s; }
.tab:hover { color: #e2e8f0; }
.tab.active { color: #60a5fa; border-bottom-color: #60a5fa; }
.tab-content { display: none; padding: 24px 32px; }
.tab-content.active { display: block; }
.chart-row { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
.chart-box { background: #1e293b; border-radius: 12px; padding: 16px; flex: 1; min-width: 400px; border: 1px solid #334155; }
.chart-box.full { min-width: 100%; }
.chart-box h3 { color: #f8fafc; font-size: 15px; margin-bottom: 12px; font-weight: 600; }
.chart-box .note { color: #64748b; font-size: 12px; margin-bottom: 8px; }
.glossary-toggle { color: #60a5fa; font-size: 13px; cursor: pointer; padding: 8px 32px; display: inline-block; }
.glossary-toggle:hover { color: #93c5fd; }
.glossary { display: none; padding: 12px 32px 20px; }
.glossary.open { display: block; }
.glossary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 12px; }
.glossary-item { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 12px 16px; }
.glossary-item .term { color: #60a5fa; font-weight: 600; font-size: 13px; }
.glossary-item .def { color: #94a3b8; font-size: 12px; margin-top: 4px; line-height: 1.5; }
"""


def _build_html(stats: dict, card_names: list[str], model_info: str, replay_path: str) -> str:
    """Build the complete HTML dashboard string."""
    # Escape </script> sequences so the JSON cannot break out of its
    # containing <script> tag during HTML parsing.
    stats_json = json.dumps(stats, default=str).replace("</", r"<\/")
    escaped_info = html.escape(model_info or replay_path)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Star Realms AI — Analysis Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>{_CSS}</style>
</head>
<body>
<div class="header">
  <h1>🚀 Star Realms AI — Analysis Dashboard</h1>
  <div class="subtitle">{escaped_info} · Generated from replay data</div>
</div>
<div id="kpis" class="kpi-row"></div>
<div class="glossary-toggle" onclick="this.nextElementSibling.classList.toggle('open')">📖 Glossary — click to expand</div>
<div class="glossary">
<div class="glossary-grid">
  <div class="glossary-item"><div class="term">Early / Mid / Late (Game Phases)</div><div class="def">Turns 1–5 = early, 6–15 = mid, 16+ = late. Used to track how the agent's strategy evolves over the course of a game.</div></div>
  <div class="glossary-item"><div class="term">Policy Entropy</div><div class="def">Measures how uncertain the agent is about its next action. High entropy = many viable options being considered. Low entropy = agent is very confident in one action. Entropy near 0 may indicate the policy has collapsed.</div></div>
  <div class="glossary-item"><div class="term">Value Estimate</div><div class="def">The critic network's prediction of how likely the agent is to win from the current state. Positive = expects to win, negative = expects to lose. Range is unbounded but typically –1 to +1.</div></div>
  <div class="glossary-item"><div class="term">Buy Rate (when affordable)</div><div class="def">How often the agent buys a card when it has enough trade to do so. 100% = always buys when it can afford it. Measures card preference independent of economy strength.</div></div>
  <div class="glossary-item"><div class="term">Trade / Combat</div><div class="def">Trade is currency for buying cards from the trade row. Combat is damage for attacking opponents or destroying bases. Both reset to 0 at end of turn.</div></div>
  <div class="glossary-item"><div class="term">Authority (Health)</div><div class="def">Each player starts with 50 authority. Reduced by opponent combat. A player at 0 or below loses. Some cards gain authority (healing).</div></div>
  <div class="glossary-item"><div class="term">Health Differential</div><div class="def">Player's health minus opponent's health at each decision point. Positive = ahead, negative = behind. Shows when games diverge.</div></div>
  <div class="glossary-item"><div class="term">Turning Point</div><div class="def">The turn in a lost game where the agent's value estimate first goes negative — the moment the critic thinks the game is lost.</div></div>
  <div class="glossary-item"><div class="term">Buy Rate Delta</div><div class="def">Difference in buy rate between winning and losing games. A large positive delta means the agent buys that card more in wins — it may be a key card to prioritize.</div></div>
  <div class="glossary-item"><div class="term">Card Co-occurrence</div><div class="def">How often two cards are both purchased in the same winning game. High co-occurrence suggests synergy between cards.</div></div>
  <div class="glossary-item"><div class="term">P10 / P90 Range</div><div class="def">The 10th and 90th percentile values. The shaded band on economy curves shows where 80% of games fall, filtering out extreme outliers.</div></div>
  <div class="glossary-item"><div class="term">Base / Outpost</div><div class="def">Bases stay in play across turns (unlike ships). Outposts are bases that must be destroyed before the opponent can attack you or your other bases directly.</div></div>
</div>
</div>
<div class="tabs">
  <div class="tab active" onclick="switchTab('overview', this)">Overview</div>
  <div class="tab" onclick="switchTab('economy', this)">Economy</div>
  <div class="tab" onclick="switchTab('decisions', this)">Decisions</div>
  <div class="tab" onclick="switchTab('cards', this)">Cards</div>
  <div class="tab" onclick="switchTab('combat', this)">Combat</div>
  <div class="tab" onclick="switchTab('failures', this)">Failures</div>
  <div class="tab" onclick="switchTab('strategy', this)">Strategy</div>
</div>
<div id="tab-overview" class="tab-content active"></div>
<div id="tab-economy" class="tab-content"></div>
<div id="tab-decisions" class="tab-content"></div>
<div id="tab-cards" class="tab-content"></div>
<div id="tab-combat" class="tab-content"></div>
<div id="tab-failures" class="tab-content"></div>
<div id="tab-strategy" class="tab-content"></div>
<script id="dashboard-data" type="application/json">{stats_json}</script>
<script>
{_DASHBOARD_JS}
</script>
</body>
</html>"""


# JavaScript for the dashboard — kept as a plain string to avoid
# f-string brace conflicts with JS object literals.
_DASHBOARD_JS = r"""
const S = JSON.parse(document.getElementById('dashboard-data').textContent);
const plotBg = '#1e293b';
const plotPaper = '#1e293b';
const plotFont = { color: '#e2e8f0', family: '-apple-system, sans-serif' };
const plotGrid = { color: '#334155' };
const plotLayout = { paper_bgcolor: plotPaper, plot_bgcolor: plotBg, font: plotFont, margin: {l:50,r:20,t:40,b:40}, xaxis: {gridcolor: plotGrid}, yaxis: {gridcolor: plotGrid} };
const cardLayout = { paper_bgcolor: plotPaper, plot_bgcolor: plotBg, font: plotFont, margin: {l:160,r:20,t:40,b:40}, xaxis: {gridcolor: plotGrid}, yaxis: {gridcolor: plotGrid} };
const pairLayout = { paper_bgcolor: plotPaper, plot_bgcolor: plotBg, font: plotFont, margin: {l:280,r:20,t:40,b:40}, xaxis: {gridcolor: plotGrid}, yaxis: {gridcolor: plotGrid} };
const plotConfig = { responsive: true, displayModeBar: false };

function switchTab(name, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  el.classList.add('active');
  if (!document.getElementById('tab-' + name).dataset.rendered) {
    renderTab(name);
    document.getElementById('tab-' + name).dataset.rendered = '1';
  }
}

// KPIs
const ov = S.overview;
document.getElementById('kpis').innerHTML =
  '<div class="kpi"><div class="label">Win Rate</div><div class="value">' + (ov.win_rate*100).toFixed(1) + '%</div><div class="detail">' + Math.round(ov.win_rate*ov.total_games) + ' / ' + ov.total_games + ' games</div></div>' +
  '<div class="kpi"><div class="label">Avg Game Length</div><div class="value">' + ov.avg_length.toFixed(1) + '</div><div class="detail">W: ' + ov.avg_length_wins.toFixed(1) + ' / L: ' + ov.avg_length_losses.toFixed(1) + '</div></div>' +
  '<div class="kpi"><div class="label">Total Decisions</div><div class="value">' + ov.total_decisions.toLocaleString() + '</div><div class="detail">' + (ov.total_decisions/ov.total_games).toFixed(0) + ' per game</div></div>' +
  '<div class="kpi"><div class="label">Value Accuracy</div><div class="value">' + S.values.wins_mean.toFixed(3) + '</div><div class="detail">Wins avg / Losses: ' + S.values.losses_mean.toFixed(3) + '</div></div>';

function renderTab(name) {
  var el = document.getElementById('tab-' + name);
  switch(name) {
    case 'overview': renderOverview(el); break;
    case 'economy': renderEconomy(el); break;
    case 'decisions': renderDecisions(el); break;
    case 'cards': renderCards(el); break;
    case 'combat': renderCombat(el); break;
    case 'failures': renderFailures(el); break;
    case 'strategy': renderStrategy(el); break;
  }
}

function renderOverview(el) {
  el.innerHTML = '<div class="chart-row"><div class="chart-box"><h3>Win Rate by Opponent</h3><div id="ov-opp"></div></div><div class="chart-box"><h3>Game Length Distribution</h3><div id="ov-lengths"></div></div></div>';
  var opps = Object.entries(ov.opponent_stats);
  if (opps.length > 0) {
    var names = opps.map(function(e){return e[0]});
    var rates = opps.map(function(e){return e[1].total > 0 ? (e[1].wins/e[1].total*100) : 0});
    var counts = opps.map(function(e){return e[1].wins+'/'+e[1].total});
    Plotly.newPlot('ov-opp', [{x:names, y:rates, type:'bar', marker:{color:rates.map(function(r){return r>=50?'#4ade80':'#f87171'})}, text:counts, textposition:'auto'}], Object.assign({}, plotLayout, {yaxis:{gridcolor:plotGrid, title:'Win Rate %', range:[0,100]}}), plotConfig);
  }
  var traces = [];
  if (ov.win_lengths.length) traces.push({x:ov.win_lengths, type:'histogram', name:'Wins', marker:{color:'#4ade80', opacity:0.7}});
  if (ov.loss_lengths.length) traces.push({x:ov.loss_lengths, type:'histogram', name:'Losses', marker:{color:'#f87171', opacity:0.7}});
  Plotly.newPlot('ov-lengths', traces, Object.assign({}, plotLayout, {barmode:'overlay', xaxis:{gridcolor:plotGrid, title:'Turns'}, yaxis:{gridcolor:plotGrid, title:'Games'}}), plotConfig);
}

function renderEconomy(el) {
  el.innerHTML = '<div class="chart-row"><div class="chart-box full"><h3>Trade & Combat per Turn</h3><p class="note">Solid = mean, dashed = median, shaded = P10-P90 range</p><div id="ec-resources"></div></div></div><div class="chart-row"><div class="chart-box full"><h3>Health over Time — Wins vs Losses</h3><div id="ec-health"></div></div></div>';
  var econ = S.economy.all;
  var turns = Object.keys(econ).map(Number).sort(function(a,b){return a-b});
  function mkTrace(key, color, label) {
    var vals = turns.map(function(t){return econ[t] && econ[t][key] ? econ[t][key] : {}});
    return [
      {x:turns, y:vals.map(function(v){return v.mean||0}), name:'Mean '+label, line:{color:color, width:2}, type:'scatter'},
      {x:turns, y:vals.map(function(v){return v.median||0}), name:'Median '+label, line:{color:color, width:1.5, dash:'dash'}, type:'scatter'},
      {x:turns.concat(turns.slice().reverse()), y:vals.map(function(v){return v.p90||0}).concat(vals.slice().reverse().map(function(v){return v.p10||0})), fill:'toself', fillcolor:color+'15', line:{width:0}, showlegend:false, name:label+' range', type:'scatter'}
    ];
  }
  Plotly.newPlot('ec-resources', mkTrace('trade','#eab308','Trade').concat(mkTrace('combat','#ef4444','Combat')), Object.assign({}, plotLayout, {xaxis:{gridcolor:plotGrid, title:'Turn'}, yaxis:{gridcolor:plotGrid, title:'Resources'}}), plotConfig);
  var ew = S.economy.wins; var el2 = S.economy.losses;
  var wT = Object.keys(ew).map(Number).sort(function(a,b){return a-b});
  var lT = Object.keys(el2).map(Number).sort(function(a,b){return a-b});
  Plotly.newPlot('ec-health', [
    {x:wT, y:wT.map(function(t){return ew[t] && ew[t].health ? ew[t].health.mean : 0}), name:'Wins (mean)', line:{color:'#4ade80', width:2}},
    {x:wT, y:wT.map(function(t){return ew[t] && ew[t].health ? ew[t].health.median : 0}), name:'Wins (median)', line:{color:'#4ade80', width:1.5, dash:'dash'}},
    {x:lT, y:lT.map(function(t){return el2[t] && el2[t].health ? el2[t].health.mean : 0}), name:'Losses (mean)', line:{color:'#f87171', width:2}},
    {x:lT, y:lT.map(function(t){return el2[t] && el2[t].health ? el2[t].health.median : 0}), name:'Losses (median)', line:{color:'#f87171', width:1.5, dash:'dash'}}
  ], Object.assign({}, plotLayout, {xaxis:{gridcolor:plotGrid, title:'Turn'}, yaxis:{gridcolor:plotGrid, title:'Health'}}), plotConfig);
}

function renderDecisions(el) {
  el.innerHTML = '<div class="chart-row"><div class="chart-box"><h3>Action Distribution by Phase</h3><div id="dc-actions"></div></div><div class="chart-box"><h3>Entropy by Phase</h3><div id="dc-entropy-phase"></div></div></div><div class="chart-row"><div class="chart-box full"><h3>Policy Entropy over Time</h3><p class="note">Wins vs Losses</p><div id="dc-entropy-turn"></div></div></div><div class="chart-row"><div class="chart-box full"><h3>Value Estimate Trajectories</h3><p class="note">Up to 50 games — green = wins, red = losses. Hover for details.</p><div id="dc-values"></div></div></div>';
  var ad = S.actions.all;
  var allTypes = []; Object.values(ad).forEach(function(p){Object.keys(p).forEach(function(t){if(allTypes.indexOf(t)<0)allTypes.push(t)});}); allTypes.sort();
  var colors = ['#60a5fa','#f87171','#4ade80','#eab308','#a78bfa','#fb923c','#94a3b8','#2dd4bf','#e879f9','#f472b6'];
  var traces = allTypes.map(function(t,i){return {x:['Early','Mid','Late'], y:['early','mid','late'].map(function(p){return ad[p] && ad[p][t] ? ad[p][t] : 0}), name:t, type:'bar', marker:{color:colors[i%colors.length]}}});
  Plotly.newPlot('dc-actions', traces, Object.assign({}, plotLayout, {barmode:'stack', yaxis:{gridcolor:plotGrid, title:'% of Decisions'}}), plotConfig);
  var ep = S.entropy.by_phase;
  Plotly.newPlot('dc-entropy-phase', [{x:['Early','Mid','Late'], y:['early','mid','late'].map(function(p){return ep[p]?ep[p].mean:0}), type:'bar', marker:{color:['#4ade80','#eab308','#f87171']}, text:['early','mid','late'].map(function(p){return (ep[p]?ep[p].mean:0).toFixed(3)}), textposition:'auto'}], Object.assign({}, plotLayout, {yaxis:{gridcolor:plotGrid, title:'Mean Entropy'}}), plotConfig);
  var etw = S.entropy.by_turn_wins; var etl = S.entropy.by_turn_losses;
  var etwT = Object.keys(etw).map(Number).sort(function(a,b){return a-b});
  var etlT = Object.keys(etl).map(Number).sort(function(a,b){return a-b});
  Plotly.newPlot('dc-entropy-turn', [{x:etwT, y:etwT.map(function(t){return etw[t]?etw[t].mean:0}), name:'Wins', line:{color:'#4ade80', width:2}}, {x:etlT, y:etlT.map(function(t){return etl[t]?etl[t].mean:0}), name:'Losses', line:{color:'#f87171', width:2}}], Object.assign({}, plotLayout, {xaxis:{gridcolor:plotGrid, title:'Turn'}, yaxis:{gridcolor:plotGrid, title:'Mean Entropy'}}), plotConfig);
  var vt = S.values.trajectories;
  var vtTraces = vt.map(function(g){return {x:g.values.map(function(v){return v.turn}), y:g.values.map(function(v){return v.value}), name:'Game '+g.game_id, type:'scatter', mode:'lines', line:{color:g.winner==='PPO'?'rgba(74,222,128,0.25)':'rgba(248,113,113,0.25)', width:1}, showlegend:false, hovertemplate:'Game '+g.game_id+'<br>Turn %{x}<br>Value: %{y:.3f}<extra>'+g.opponent+'</extra>'}});
  Plotly.newPlot('dc-values', vtTraces, Object.assign({}, plotLayout, {xaxis:{gridcolor:plotGrid, title:'Turn'}, yaxis:{gridcolor:plotGrid, title:'Value Estimate'}, shapes:[{type:'line',x0:0,x1:50,y0:0,y1:0,line:{color:'#64748b',width:1,dash:'dot'}}]}), plotConfig);
}

function renderCards(el) {
  el.innerHTML = '<div class="chart-row"><div class="chart-box full"><h3>Buy Rate Heatmap (when affordable)</h3><div id="cd-heatmap"></div></div></div><div class="chart-row"><div class="chart-box"><h3>Card Acquisition Timing</h3><p class="note">Average turn when each card is bought</p><div id="cd-timing"></div></div><div class="chart-box"><h3>Deck Composition: Wins vs Losses</h3><div id="cd-deck"></div></div></div><div class="chart-row"><div class="chart-box full"><h3>Card Co-occurrence in Wins</h3><p class="note">Top 20 card pairs most frequently bought together in winning games</p><div id="cd-cooccurrence"></div></div></div>';
  var bt = S.cards.buy_table;
  var sorted = Object.entries(bt).sort(function(a,b){return b[1].overall-a[1].overall}).slice(0,20);
  var cardLabels = sorted.map(function(e){return e[0]});
  var z = sorted.map(function(e){return ['early','mid','late'].map(function(p){return e[1][p]})});
  Plotly.newPlot('cd-heatmap', [{z:z, x:['Early','Mid','Late'], y:cardLabels, type:'heatmap', colorscale:[[0,'#1e293b'],[0.5,'#eab308'],[1,'#ef4444']], zmin:0, zmax:100, text:z.map(function(r){return r.map(function(v){return v+'%'})}), texttemplate:'%{text}', textfont:{size:11}, colorbar:{title:'Buy Rate %', ticksuffix:'%'}}], Object.assign({}, cardLayout, {yaxis:{gridcolor:plotGrid, autorange:'reversed'}, height:Math.max(400, sorted.length*28)}), plotConfig);
  var timing = Object.entries(S.cards.buy_timing).sort(function(a,b){return a[1].mean_turn-b[1].mean_turn}).slice(0,20);
  Plotly.newPlot('cd-timing', [{y:timing.map(function(e){return e[0]}), x:timing.map(function(e){return e[1].mean_turn}), type:'bar', orientation:'h', marker:{color:timing.map(function(e){return e[1].mean_turn}), colorscale:[[0,'#4ade80'],[0.5,'#eab308'],[1,'#f87171']]}, text:timing.map(function(e){return 'Turn '+e[1].mean_turn+' ('+e[1].count+'x)'}), textposition:'auto'}], Object.assign({}, cardLayout, {xaxis:{gridcolor:plotGrid, title:'Avg Turn Bought'}, yaxis:{gridcolor:plotGrid, autorange:'reversed'}, height:Math.max(400, timing.length*28)}), plotConfig);
  var dc = S.cards.deck_composition;
  var dcSorted = Object.entries(dc).filter(function(e){return e[1].avg_wins>0.05||e[1].avg_losses>0.05}).sort(function(a,b){return (b[1].avg_wins-b[1].avg_losses)-(a[1].avg_wins-a[1].avg_losses)}).slice(0,20);
  Plotly.newPlot('cd-deck', [{y:dcSorted.map(function(e){return e[0]}), x:dcSorted.map(function(e){return e[1].avg_wins}), name:'Wins', type:'bar', orientation:'h', marker:{color:'#4ade80'}}, {y:dcSorted.map(function(e){return e[0]}), x:dcSorted.map(function(e){return e[1].avg_losses}), name:'Losses', type:'bar', orientation:'h', marker:{color:'#f87171'}}], Object.assign({}, cardLayout, {barmode:'group', xaxis:{gridcolor:plotGrid, title:'Avg Cards Bought per Game'}, yaxis:{gridcolor:plotGrid, autorange:'reversed'}, height:Math.max(400, dcSorted.length*28)}), plotConfig);
  var co = S.cards.co_occurrence;
  if (co.length > 0) { Plotly.newPlot('cd-cooccurrence', [{y:co.map(function(c){return c.pair.join(' + ')}), x:co.map(function(c){return c.count}), type:'bar', orientation:'h', marker:{color:'#60a5fa'}, text:co.map(function(c){return c.count+' wins'}), textposition:'auto'}], Object.assign({}, pairLayout, {xaxis:{gridcolor:plotGrid, title:'Games Won Together'}, yaxis:{gridcolor:plotGrid, autorange:'reversed'}, height:Math.max(400, co.length*28)}), plotConfig); }
}

function renderCombat(el) {
  el.innerHTML = '<div class="chart-row"><div class="chart-box full"><h3>Combat Generated per Turn</h3><div id="cb-combat"></div></div></div><div class="chart-row"><div class="chart-box"><h3>Attack Timing</h3><p class="note">When the agent attacks players vs bases</p><div id="cb-timing"></div></div><div class="chart-box"><h3>Most Played Bases</h3><p class="note">Total decision-steps each base was in play</p><div id="cb-bases"></div></div></div>';
  var cp = S.combat.per_turn;
  var cpTurns = Object.keys(cp).map(Number).sort(function(a,b){return a-b});
  Plotly.newPlot('cb-combat', [{x:cpTurns, y:cpTurns.map(function(t){return cp[t]?cp[t].avg_combat:0}), name:'Avg Combat', type:'scatter', line:{color:'#ef4444', width:2}}], Object.assign({}, plotLayout, {xaxis:{gridcolor:plotGrid, title:'Turn'}, yaxis:{gridcolor:plotGrid, title:'Avg Combat Available'}}), plotConfig);
  var atP = S.combat.attack_timing_player; var atB = S.combat.attack_timing_base;
  var allT = Object.keys(Object.assign({}, atP, atB)).map(Number).sort(function(a,b){return a-b});
  Plotly.newPlot('cb-timing', [{x:allT, y:allT.map(function(t){return atP[t]||0}), name:'Attack Player', type:'bar', marker:{color:'#f87171'}}, {x:allT, y:allT.map(function(t){return atB[t]||0}), name:'Attack Base', type:'bar', marker:{color:'#eab308'}}], Object.assign({}, plotLayout, {barmode:'stack', xaxis:{gridcolor:plotGrid, title:'Turn'}, yaxis:{gridcolor:plotGrid, title:'Attacks'}}), plotConfig);
  var bf = S.combat.base_frequency;
  var bfSorted = Object.entries(bf).sort(function(a,b){return b[1]-a[1]});
  Plotly.newPlot('cb-bases', [{y:bfSorted.map(function(e){return e[0]}), x:bfSorted.map(function(e){return e[1]}), type:'bar', orientation:'h', marker:{color:'#60a5fa'}, text:bfSorted.map(function(e){return e[1]}), textposition:'auto'}], Object.assign({}, cardLayout, {yaxis:{gridcolor:plotGrid, autorange:'reversed'}, height:Math.max(300, bfSorted.length*28)}), plotConfig);
}

function renderFailures(el) {
  el.innerHTML = '<div class="chart-row"><div class="chart-box full"><h3>Health Differential over Time</h3><p class="note">Player health minus opponent health — wins vs losses</p><div id="fl-health"></div></div></div><div class="chart-row"><div class="chart-box"><h3>Turning Points in Losses</h3><p class="note">Turn where value estimate first goes negative</p><div id="fl-turning"></div></div><div class="chart-box"><h3>Buy Rate Delta: Wins vs Losses</h3><p class="note">Cards bought more often in wins vs losses</p><div id="fl-buydelta"></div></div></div>';
  var hdw = S.failures.health_diff_wins; var hdl = S.failures.health_diff_losses;
  var wT = Object.keys(hdw).map(Number).sort(function(a,b){return a-b});
  var lT = Object.keys(hdl).map(Number).sort(function(a,b){return a-b});
  Plotly.newPlot('fl-health', [{x:wT, y:wT.map(function(t){return hdw[t]}), name:'Wins', line:{color:'#4ade80', width:2}}, {x:lT, y:lT.map(function(t){return hdl[t]}), name:'Losses', line:{color:'#f87171', width:2}}, {type:'scatter', x:[0,50], y:[0,0], mode:'lines', line:{color:'#64748b', width:1, dash:'dot'}, showlegend:false}], Object.assign({}, plotLayout, {xaxis:{gridcolor:plotGrid, title:'Turn'}, yaxis:{gridcolor:plotGrid, title:'Health Differential'}}), plotConfig);
  var tp = S.failures.turning_points;
  var tpTurns = Object.keys(tp).map(Number).sort(function(a,b){return a-b});
  if (tpTurns.length > 0) { Plotly.newPlot('fl-turning', [{x:tpTurns, y:tpTurns.map(function(t){return tp[t]}), type:'bar', marker:{color:'#f87171'}}], Object.assign({}, plotLayout, {xaxis:{gridcolor:plotGrid, title:'Turn'}, yaxis:{gridcolor:plotGrid, title:'Games Turning Negative'}}), plotConfig); }
  var bd = S.failures.buy_rate_delta;
  var bdSorted = Object.entries(bd).sort(function(a,b){return b[1].delta-a[1].delta}).slice(0,15);
  if (bdSorted.length > 0) { Plotly.newPlot('fl-buydelta', [{y:bdSorted.map(function(e){return e[0]}), x:bdSorted.map(function(e){return e[1].win_rate}), name:'Win Buy Rate', type:'bar', orientation:'h', marker:{color:'#4ade80'}}, {y:bdSorted.map(function(e){return e[0]}), x:bdSorted.map(function(e){return e[1].loss_rate}), name:'Loss Buy Rate', type:'bar', orientation:'h', marker:{color:'#f87171'}}], Object.assign({}, cardLayout, {barmode:'group', xaxis:{gridcolor:plotGrid, title:'Buy Rate %'}, yaxis:{gridcolor:plotGrid, autorange:'reversed'}, height:Math.max(300, bdSorted.length*35)}), plotConfig); }
}

function renderStrategy(el) {
  el.innerHTML = '<div class="chart-row"><div class="chart-box full"><h3>Outcome-Conditioned Win Rate Lift</h3><p class="note">X = how often the card is bought, Y = win rate when bought minus win rate when not bought. Upper-right = frequently bought and correlated with winning.</p><div id="st-lift"></div></div></div>' +
    '<div class="chart-row"><div class="chart-box full"><h3>Value Estimate Reaction on Buy</h3><p class="note">Average change in the model\'s own value estimate immediately after buying each card. Positive = the model thinks this buy improved its position.</p><div id="st-value"></div></div></div>' +
    '<div class="chart-row"><div class="chart-box full"><h3>Emergent Deck Archetypes</h3><p class="note">Games clustered by faction purchase mix (unsupervised). Labels are assigned after clustering based on dominant factions — not predefined.</p><div id="st-archetypes"></div></div></div>' +
    '<div class="chart-row"><div class="chart-box full"><h3>Archetype Faction Profiles</h3><div id="st-archetype-profiles"></div></div></div>' +
    '<div class="chart-row"><div class="chart-box full"><h3>Contrastive Card Analysis</h3><p class="note">When the agent buys a given card, what else does it tend to buy more/less often? Delta = co-purchase rate when bought minus when not bought.</p><div id="st-contrastive"></div></div></div>';

  var st = S.strategy;

  // Win rate lift scatter
  var wrl = st.win_rate_lift;
  var wrlEntries = Object.entries(wrl);
  if (wrlEntries.length > 0) {
    Plotly.newPlot('st-lift', [{
      x: wrlEntries.map(function(e){return e[1].buy_freq}),
      y: wrlEntries.map(function(e){return e[1].lift}),
      text: wrlEntries.map(function(e){return e[0] + '<br>Bought WR: ' + e[1].bought_wr + '%<br>Not Bought WR: ' + e[1].not_bought_wr + '%<br>n=' + e[1].sample}),
      mode: 'markers+text', type: 'scatter',
      textposition: 'top center',
      textfont: {size: 10, color: '#94a3b8'},
      marker: {
        size: wrlEntries.map(function(e){return Math.max(8, Math.min(30, e[1].sample / 5))}),
        color: wrlEntries.map(function(e){return e[1].lift}),
        colorscale: [[0,'#f87171'],[0.5,'#94a3b8'],[1,'#4ade80']],
        cmid: 0, colorbar: {title: 'Win Rate Lift %'},
      },
      hovertemplate: '%{text}<extra></extra>',
    }], Object.assign({}, plotLayout, {
      xaxis: {gridcolor: plotGrid, title: 'Buy Frequency %'},
      yaxis: {gridcolor: plotGrid, title: 'Win Rate Lift (bought - not bought) %'},
      shapes: [{type:'line',x0:0,x1:100,y0:0,y1:0,line:{color:'#64748b',width:1,dash:'dot'}}],
      height: 500,
    }), plotConfig);
  }

  // Value reactions bar chart
  var vr = st.value_reactions;
  var vrEntries = Object.entries(vr).sort(function(a,b){return b[1].mean_delta - a[1].mean_delta});
  if (vrEntries.length > 0) {
    Plotly.newPlot('st-value', [{
      y: vrEntries.map(function(e){return e[0]}),
      x: vrEntries.map(function(e){return e[1].mean_delta}),
      type: 'bar', orientation: 'h',
      marker: {color: vrEntries.map(function(e){return e[1].mean_delta >= 0 ? '#4ade80' : '#f87171'})},
      text: vrEntries.map(function(e){return (e[1].mean_delta >= 0 ? '+' : '') + e[1].mean_delta.toFixed(4) + ' (' + e[1].count + 'x)'}),
      textposition: 'auto',
    }], Object.assign({}, cardLayout, {
      xaxis: {gridcolor: plotGrid, title: 'Avg Value Estimate Change'},
      yaxis: {gridcolor: plotGrid, autorange: 'reversed'},
      shapes: [{type:'line',x0:0,x1:0,y0:-0.5,y1:vrEntries.length-0.5,line:{color:'#64748b',width:1,dash:'dot'}}],
      height: Math.max(400, vrEntries.length * 28),
    }), plotConfig);
  }

  // Archetype clusters
  var arch = st.archetypes.clusters;
  if (arch && arch.length > 0) {
    var labels = arch.map(function(c){return c.label + ' (' + c.size + ' games)'});
    var winRates = arch.map(function(c){return c.win_rate});
    var sizes = arch.map(function(c){return c.size});
    Plotly.newPlot('st-archetypes', [{
      x: labels, y: winRates, type: 'bar',
      marker: {color: winRates.map(function(r){return r >= 50 ? '#4ade80' : '#f87171'})},
      text: winRates.map(function(r){return r + '%'}), textposition: 'auto',
    }], Object.assign({}, plotLayout, {
      yaxis: {gridcolor: plotGrid, title: 'Win Rate %', range: [0, 100]},
      xaxis: {gridcolor: plotGrid},
    }), plotConfig);

    // Archetype faction profiles (stacked bar)
    var fLabels = st.faction_labels;
    var fColors = {'Blob': '#4ade80', 'Machine Cult': '#f87171', 'Star Empire': '#eab308', 'Trade Federation': '#60a5fa', 'Unaligned': '#94a3b8'};
    var profileTraces = fLabels.map(function(f) {
      return {
        x: arch.map(function(c){return c.label}),
        y: arch.map(function(c){return c.centroid[f] || 0}),
        name: f, type: 'bar',
        marker: {color: fColors[f] || '#94a3b8'},
      };
    });
    Plotly.newPlot('st-archetype-profiles', profileTraces, Object.assign({}, plotLayout, {
      barmode: 'stack',
      yaxis: {gridcolor: plotGrid, title: 'Faction Mix %'},
      xaxis: {gridcolor: plotGrid},
    }), plotConfig);
  }

  // Contrastive card analysis — top 6 cards as subplots
  var cont = st.contrastive;
  var contCards = Object.keys(cont).slice(0, 6);
  if (contCards.length > 0) {
    var contHtml = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:16px;">';
    contCards.forEach(function(card, idx) {
      contHtml += '<div style="background:#0f172a;border:1px solid #334155;border-radius:8px;padding:12px;"><h4 style="color:#f8fafc;font-size:13px;margin-bottom:8px;">When buying ' + card + ' (' + cont[card].games_with + ' games)</h4><div id="st-cont-' + idx + '"></div></div>';
    });
    contHtml += '</div>';
    document.getElementById('st-contrastive').innerHTML = contHtml;

    contCards.forEach(function(card, idx) {
      var companions = cont[card].companions;
      if (companions.length === 0) return;
      Plotly.newPlot('st-cont-' + idx, [
        {y: companions.map(function(c){return c.card}), x: companions.map(function(c){return c.rate_with}), name: 'With ' + card, type: 'bar', orientation: 'h', marker: {color: '#60a5fa'}},
        {y: companions.map(function(c){return c.card}), x: companions.map(function(c){return c.rate_without}), name: 'Without', type: 'bar', orientation: 'h', marker: {color: '#64748b'}},
      ], Object.assign({}, cardLayout, {
        barmode: 'group', height: Math.max(200, companions.length * 32),
        xaxis: {gridcolor: plotGrid, title: 'Co-purchase Rate %'},
        yaxis: {gridcolor: plotGrid, autorange: 'reversed'},
        margin: {l: 140, r: 10, t: 10, b: 30},
      }), plotConfig);
    });
  }
}

// Render the overview tab on load
renderTab('overview');
"""
