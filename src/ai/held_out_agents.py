"""Held-out scripted opponents for measuring checkpoint strength.

Six styles of one rule-based player, which scores every purchase from the
card's own effect tree with phase-dependent weights:

- ``balanced``: economy and thinning early, damage and draw later;
- ``aggro``: damage first, always attacks the player, burns scrap abilities
  to finish;
- ``economy``: trade first, then the most expensive cards available;
- ``scrapper``: Machine Cult thinning above everything else;
- ``fortress``: outposts, authority, and a slow win;
- ``blobrush``: cheap Blob damage with Blob allies.

They are held out: no agent trains against them and they are not tuned
against any checkpoint (docs/decisions/0013-strength-yardstick.md). Keep them
out of every training opponent registry, and do not change their decision
rules: the league's port check reproduced published counts with them under
rules version 1. Those counts belong to the old rules. Under the current rules
(docs/decisions/0016-rulebook-ability-timing.md) an OR ability gives one
option, which ``_choose_or`` picks.
"""
from __future__ import annotations

from collections import defaultdict
from types import MappingProxyType

from src.ai.agent import Agent
from src.cards.effects import CardEffectType as T
from src.engine.actions import ActionType as A, CardSource, get_available_actions

STARTERS = ("Scout", "Viper")

HELD_OUT_STYLES = ("balanced", "aggro", "economy", "scrapper", "fortress", "blobrush")

# Phase weights are (early, mid, late): a player's turns 0-3, 4-8, and 9+.
_STYLE_PARAMETERS = {
    "balanced": dict(
        trade=(3.0, 1.6, 0.3), combat=(1.0, 1.8, 2.4), draw=(3.0, 3.2, 3.2),
        thin=(4.0, 2.5, 0.5), heal=(0.3, 0.5, 0.9), discard=(1.5, 1.8, 1.8),
        destroy=(1.0, 1.5, 1.5), special=(1.0, 1.0, 1.0), cost=(0.5, 0.6, 0.6),
        base_def=(0.25, 0.35, 0.35), outpost_def=(0.45, 0.6, 0.7),
        ally=0.7, synergy=0.5, faction_bonus={}, threshold=(2.5, 3.5, 5.0),
        explorer_until=3, base_kill_min_cost=5, burn_below=0,
        explorer_scrap_turn=7, deny_cost=7,
    ),
    "aggro": dict(
        trade=(2.0, 0.6, 0.0), combat=(2.4, 2.8, 3.0), draw=(2.5, 2.8, 2.8),
        thin=(1.5, 1.0, 0.2), heal=(0.0, 0.1, 0.3), discard=(2.0, 2.2, 2.2),
        destroy=(0.8, 1.0, 1.0), special=(0.8, 0.8, 0.8), cost=(0.3, 0.3, 0.3),
        base_def=(0.05, 0.05, 0.05), outpost_def=(0.1, 0.1, 0.1),
        ally=0.8, synergy=0.6, faction_bonus={}, threshold=(2.0, 3.0, 4.0),
        explorer_until=2, base_kill_min_cost=99, burn_below=18,
        explorer_scrap_turn=5, deny_cost=99,
    ),
    "economy": dict(
        trade=(4.0, 2.6, 0.6), combat=(0.6, 1.4, 2.0), draw=(3.0, 3.0, 3.0),
        thin=(3.0, 2.0, 0.5), heal=(0.4, 0.6, 0.8), discard=(1.0, 1.4, 1.4),
        destroy=(1.0, 1.5, 1.5), special=(1.2, 1.2, 1.2), cost=(0.9, 1.1, 1.1),
        base_def=(0.3, 0.4, 0.4), outpost_def=(0.4, 0.6, 0.6),
        ally=0.6, synergy=0.4, faction_bonus={}, threshold=(2.5, 3.5, 5.0),
        explorer_until=4, base_kill_min_cost=4, burn_below=0,
        explorer_scrap_turn=8, deny_cost=7,
    ),
    "scrapper": dict(
        trade=(2.6, 1.6, 0.3), combat=(1.0, 1.8, 2.4), draw=(3.0, 3.2, 3.2),
        thin=(7.0, 5.0, 1.0), heal=(0.3, 0.5, 0.8), discard=(1.2, 1.5, 1.5),
        destroy=(1.0, 1.5, 1.5), special=(1.0, 1.0, 1.0), cost=(0.5, 0.6, 0.6),
        base_def=(0.25, 0.35, 0.35), outpost_def=(0.5, 0.65, 0.7),
        ally=0.7, synergy=0.5, faction_bonus={2: 2.5}, threshold=(2.5, 3.5, 5.0),
        explorer_until=2, base_kill_min_cost=5, burn_below=0,
        explorer_scrap_turn=6, deny_cost=7,
    ),
    "fortress": dict(
        trade=(3.0, 1.8, 0.5), combat=(0.8, 1.5, 2.0), draw=(3.0, 3.0, 3.0),
        thin=(3.0, 2.0, 0.5), heal=(1.2, 1.5, 1.6), discard=(1.2, 1.5, 1.5),
        destroy=(1.0, 1.2, 1.2), special=(1.0, 1.0, 1.0), cost=(0.5, 0.6, 0.6),
        base_def=(0.7, 0.9, 0.9), outpost_def=(1.2, 1.5, 1.5),
        ally=0.7, synergy=0.5, faction_bonus={8: 1.5}, threshold=(2.5, 3.5, 5.0),
        explorer_until=3, base_kill_min_cost=3, burn_below=0,
        explorer_scrap_turn=9, deny_cost=7,
    ),
    "blobrush": dict(
        trade=(1.8, 0.6, 0.0), combat=(2.4, 2.8, 3.0), draw=(2.5, 2.8, 2.8),
        thin=(1.0, 0.8, 0.2), heal=(0.0, 0.1, 0.2), discard=(1.5, 1.8, 1.8),
        destroy=(1.0, 1.2, 1.2), special=(1.0, 1.0, 1.0), cost=(0.3, 0.3, 0.3),
        base_def=(0.05, 0.1, 0.1), outpost_def=(0.1, 0.1, 0.1),
        ally=1.0, synergy=0.8, faction_bonus={1: 3.0}, threshold=(2.0, 3.0, 4.0),
        explorer_until=2, base_kill_min_cost=99, burn_below=15,
        explorer_scrap_turn=5, deny_cost=99,
    ),
}
STYLE_PARAMETERS = MappingProxyType({
    style: MappingProxyType(dict(
        values, faction_bonus=MappingProxyType(dict(values["faction_bonus"])),
    ))
    for style, values in _STYLE_PARAMETERS.items()
})

_FEATURE_CACHE: dict[str, dict] = {}


def card_features(card) -> dict:
    """Summarize a card's printed effect tree as weighted feature totals.

    Effects are bucketed by kind (trade, combat, draw, thinning, ...) and by
    whether they are unconditional, ally-gated, or paid for by scrapping the
    card; OR choices count at 0.6. Cost, defense, type, and faction bits are
    included.
    """
    feats = defaultdict(float)

    def leaf(effect, ally, scrap, weight):
        prefix = "scrap_" if scrap else ("ally_" if ally else "")
        kind = effect.effect_type
        value = effect.value or 0
        if kind == T.COMBAT:
            feats[prefix + "combat"] += weight * value
        elif kind == T.TRADE:
            feats[prefix + "trade"] += weight * value
        elif kind == T.HEAL:
            feats[prefix + "heal"] += weight * value
        elif kind == T.DRAW:
            feats[prefix + "draw"] += weight * value
        elif kind == T.DRAW_PER_FACTION:
            feats[prefix + "draw"] += weight * 2.0
        elif kind == T.CONDITIONAL_DRAW:
            feats[prefix + "draw"] += weight * 0.8
        elif kind == T.SCRAP:
            if effect.card_targets and "trade" in effect.card_targets:
                feats[prefix + "special"] += weight * 0.3
            else:
                feats[prefix + "thin"] += weight * 1.0
        elif kind == T.SCRAP_FROM_HAND_DISCARD:
            feats[prefix + "thin"] += weight * 2.0
            feats[prefix + "draw"] += weight * 1.5
        elif kind == T.TARGET_DISCARD:
            feats[prefix + "discard"] += weight * value
        elif kind == T.DESTROY_BASE:
            feats[prefix + "destroy"] += weight * 1.0
        elif kind == T.ACQUIRE_SHIP_TOP:
            feats[prefix + "special"] += weight * 4.0
        elif kind == T.NEXT_SHIP_TOP:
            feats[prefix + "special"] += weight * 1.0
        elif kind == T.COPY_SHIP:
            feats[prefix + "special"] += weight * 3.0
        elif kind == T.PASSIVE_SHIP_COMBAT:
            feats[prefix + "special"] += weight * 5.0
        elif kind == T.WILDCARD_ALLY:
            feats[prefix + "special"] += weight * 3.0
        elif kind == T.DISCARD_DRAW:
            feats[prefix + "special"] += weight * 1.5

    def walk(effect, ally=False, scrap=False, weight=1.0):
        ally = ally or bool(effect.faction_requirement)
        scrap = scrap or bool(effect.is_scrap_effect)
        if effect.child_effects:
            child_weight = weight * (0.6 if effect.is_or_effect else 1.0)
            for child in effect.child_effects:
                walk(child, ally, scrap, child_weight)
        else:
            leaf(effect, ally, scrap, weight)

    for effect in card.effects:
        walk(effect)
    feats["cost"] = float(card.cost or 0)
    feats["defense"] = float(card.defense or 0)
    feats["is_base"] = 1.0 if card.is_base() else 0.0
    feats["is_outpost"] = 1.0 if card.is_outpost() else 0.0
    feats["faction"] = int(card.faction)
    return dict(feats)


def _features(card) -> dict:
    cached = _FEATURE_CACHE.get(card.name)
    if cached is None:
        cached = _FEATURE_CACHE[card.name] = card_features(card)
    return cached


class HeldOutAgent(Agent):
    """One held-out scripted player; create a new instance for every game."""

    def __init__(self, name: str, style: str):
        if style not in STYLE_PARAMETERS:
            raise ValueError(
                f"Unknown held-out style {style!r}. Valid: {', '.join(HELD_OUT_STYLES)}"
            )
        super().__init__(name)
        self.style = style
        self.params = STYLE_PARAMETERS[style]
        self._turn = None
        self._seen: set[int] = set()
        self._steps = 0
        self._done_buying = False

    # ---- helpers ---------------------------------------------------------
    def _sync_turn(self, game):
        marker = (id(game), game.stats.total_turns)
        if marker != self._turn:
            self._turn = marker
            self._seen = set()
            self._steps = 0
            self._done_buying = False

    @staticmethod
    def _my_turn_index(game):
        return game.stats.total_turns // 2

    def _phase(self, game):
        turn = self._my_turn_index(game)
        return 0 if turn <= 3 else (1 if turn <= 8 else 2)

    @staticmethod
    def _owned(player):
        return (player.hand + player.deck + player.discard_pile
                + player.played_cards
                + [b for b in player.bases if b not in player.played_cards])

    def _junk_rank(self, card, game):
        """Lower is scrapped or discarded first; None means keep."""
        late = self._phase(game) >= 1
        if card.name == "Viper":
            return 1 if late else 0
        if card.name == "Scout":
            return 0 if late else 1
        if card.name == "Explorer":
            return 2 if self._phase(game) == 2 else None
        return None

    def _buy_score(self, card, game, player):
        params = self.params
        phase = self._phase(game)
        f = _features(card)
        if card.name == "Explorer":
            if self._my_turn_index(game) > params["explorer_until"]:
                return -1.0
            return params["trade"][phase] * 2.0 - 2.5
        owned = self._owned(player)
        same = sum(1 for c in owned if int(c.faction) & int(card.faction))
        ally_scale = params["ally"] * min(1.0, 0.25 + 0.25 * same)
        score = 0.0
        for key in ("trade", "combat", "draw", "thin", "heal", "discard", "destroy", "special"):
            weight = params[key][phase]
            score += weight * f.get(key, 0.0)
            score += weight * ally_scale * f.get("ally_" + key, 0.0)
            score += weight * 0.25 * f.get("scrap_" + key, 0.0)
        score += params["cost"][phase] * f["cost"]
        if f["is_outpost"]:
            score += params["outpost_def"][phase] * f["defense"]
        elif f["is_base"]:
            score += params["base_def"][phase] * f["defense"]
        score += params["synergy"] * min(same, 4)
        score += params["faction_bonus"].get(int(card.faction), 0.0)
        return score

    @staticmethod
    def _scrap_combat_effects(effect_actions):
        out = []
        for action in effect_actions:
            effect = action.card_effect
            if effect is None or not effect.is_scrap_effect:
                continue
            gain = 0
            stack = [effect]
            while stack:
                node = stack.pop()
                if node.child_effects:
                    stack.extend(node.child_effects)
                elif node.effect_type == T.COMBAT:
                    gain += node.value
            if gain > 0:
                out.append((gain, action))
        return sorted(out, key=lambda item: -item[0])

    @staticmethod
    def _unapplied_leaves(effect):
        out = []
        stack = [effect]
        while stack:
            node = stack.pop()
            if node.child_effects:
                stack.extend(node.child_effects)
            elif not node.applied:
                out.append(node)
        return out

    def _has_junk(self, game, player, zones):
        cards = []
        if "hand" in zones:
            cards += player.hand
        if "discard" in zones:
            cards += player.discard_pile
        return any(self._junk_rank(c, game) is not None for c in cards)

    # ---- decision --------------------------------------------------------
    def make_decision(self, game):
        params = self.params
        self._sync_turn(game)
        player = game.current_player
        opponent = game.get_opponent(player)
        actions = get_available_actions(game, player)
        if len(actions) == 1:
            return actions[0]
        self._steps += 1
        pending = player.get_current_pending_set()
        if pending:
            return self._pending(game, player, opponent, actions, pending)
        end_turn = next(a for a in actions if a.type == A.END_TURN)
        if self._steps > 300:
            return end_turn

        plays = [a for a in actions if a.type == A.PLAY_CARD]
        effects = [a for a in actions if a.type == A.APPLY_EFFECT]

        # 1. purchased cards first: bases, then drawers, then by cost;
        #    Stealth Needle waits until another ship is in play.
        def play_priority(action):
            card = action.card
            f = _features(card)
            return (f["is_base"] * 100 + (f.get("draw", 0) > 0) * 50 + f["cost"])

        strong = [a for a in plays if a.card.name not in STARTERS
                  and a.card.name != "Stealth Needle"]
        if strong:
            return max(strong, key=play_priority)

        # 2. free effects (everything except scrapping the source card)
        choice = self._free_effect(game, player, opponent, effects)
        if choice is not None:
            return choice

        # 3. starters, then Stealth Needle
        starters = [a for a in plays if a.card.name in STARTERS]
        if starters:
            return starters[0]
        if plays:
            return plays[0]

        # 4. lethal, using scrap abilities for combat when needed
        outposts = [b for b in opponent.bases if b.is_outpost()]
        need = opponent.health + sum(b.defense or 0 for b in outposts)
        burners = self._scrap_combat_effects(effects)
        potential = player.combat + sum(gain for gain, _ in burners)
        if player.combat < need <= potential and burners:
            return burners[0][1]
        lethal = player.combat >= need

        # 5. buys
        if not lethal and not self._done_buying:
            buys = [a for a in actions if a.type == A.BUY_CARD]
            if buys:
                scored = [(self._buy_score(a.card, game, player), a) for a in buys]
                best_score, best = max(scored, key=lambda item: item[0])
                if best_score >= params["threshold"][self._phase(game)]:
                    return best
            self._done_buying = True

        # 6. scrap abilities outside lethal: late Explorers, and burn mode
        blocked = bool(outposts) and player.combat < sum(b.defense or 0 for b in outposts)
        for gain, action in burners:
            if action.card.name == "Explorer" and (
                self._my_turn_index(game) >= params["explorer_scrap_turn"]
            ):
                return action
            if (params["burn_below"] and opponent.health <= params["burn_below"]
                    and not blocked and not action.card.is_base()):
                return action

        # 7. attack
        attack = self._attack(game, player, opponent, actions)
        if attack is not None:
            return attack
        return end_turn

    def _free_effect(self, game, player, opponent, effects):
        params = self.params
        for action in effects:
            effect = action.card_effect
            if effect is None or effect.is_scrap_effect or id(effect) in self._seen:
                continue
            card = action.card
            parent = None
            for top in card.effects:
                if top.is_or_effect and top.child_effects and effect in top.child_effects:
                    parent = top
                    break
            if parent is not None:
                pick = self._choose_or(game, player, opponent, parent)
                for child in parent.child_effects:
                    self._seen.add(id(child))
                for candidate in effects:
                    if candidate.card_effect is pick:
                        return candidate
                continue
            self._seen.add(id(effect))
            leaves = self._unapplied_leaves(effect)
            if not leaves:
                continue
            wanted = False
            for leaf in leaves:
                kind = leaf.effect_type
                if kind in (T.COMBAT, T.TRADE, T.HEAL, T.DRAW, T.TARGET_DISCARD,
                            T.DRAW_PER_FACTION, T.CONDITIONAL_DRAW,
                            T.ACQUIRE_SHIP_TOP, T.NEXT_SHIP_TOP):
                    wanted = True
                elif kind == T.DESTROY_BASE:
                    wanted = wanted or bool(opponent.bases)
                elif kind in (T.SCRAP, T.SCRAP_FROM_HAND_DISCARD):
                    targets = leaf.card_targets or ["hand", "discard"]
                    if "trade" in targets:
                        wanted = wanted or any(
                            c.cost >= params["deny_cost"] and c.cost > player.trade
                            for c in game.trade_row
                        )
                    elif leaf.is_mandatory:
                        # Machine Base: only with something worth losing
                        if not self._has_junk(game, player, ["hand"]):
                            wanted = False
                            break
                        wanted = True
                    else:
                        wanted = wanted or self._has_junk(game, player, targets)
            if wanted:
                return action
        return None

    def _choose_or(self, game, player, opponent, parent):
        """The one option of an OR ability this player takes.

        Recycling Station: discard and draw while the hand holds junk, else
        trade. Blob World: draw when at least two Blob cards were played this
        turn, else combat. Trade or combat (Patrol Mech): combat when it
        reaches the opponent's authority plus outpost defense; otherwise trade
        in the early phase (also the middle phase except for aggro), combat
        later. Authority or trade (Trading Post, Barter World): trade in the
        early and middle phases, then authority. Authority or combat (Defense
        Center): with no outpost in the way or when it is lethal, combat
        (fortress: authority); otherwise authority.
        """
        phase = self._phase(game)
        style = self.style
        children = parent.child_effects
        kinds = {child.effect_type: child for child in children}
        outposts = [b for b in opponent.bases if b.is_outpost()]
        need = opponent.health + sum(b.defense or 0 for b in outposts)
        if T.DISCARD_DRAW in kinds:
            if self._has_junk(game, player, ["hand"]):
                return kinds[T.DISCARD_DRAW]
            return kinds[T.TRADE]
        if T.DRAW_PER_FACTION in kinds:
            blobs = sum(1 for c in player.played_cards if int(c.effective_faction) & 1)
            return kinds[T.DRAW_PER_FACTION] if blobs >= 2 else kinds[T.COMBAT]
        if T.COMBAT in kinds and T.TRADE in kinds:
            if player.combat + kinds[T.COMBAT].value >= need:
                return kinds[T.COMBAT]
            return kinds[T.TRADE] if phase <= (1 if style != "aggro" else 0) else kinds[T.COMBAT]
        if T.HEAL in kinds and T.TRADE in kinds:
            return kinds[T.TRADE] if phase <= 1 else kinds[T.HEAL]
        if T.HEAL in kinds and T.COMBAT in kinds:
            if not outposts or player.combat + kinds[T.COMBAT].value >= need:
                return kinds[T.COMBAT] if style != "fortress" else kinds[T.HEAL]
            return kinds[T.HEAL]
        return children[0]

    def _attack(self, game, player, opponent, actions):
        params = self.params
        if player.combat <= 0:
            return None
        attack_player = next((a for a in actions if a.type == A.ATTACK_PLAYER), None)
        base_attacks = [a for a in actions if a.type == A.ATTACK_BASE]
        by_index = {}
        for base in opponent.bases:
            by_index.setdefault(base.index, base)
        if attack_player is not None and player.combat >= opponent.health:
            return attack_player
        outposts = [b for b in opponent.bases if b.is_outpost()]
        if outposts:
            if not base_attacks:
                return None
            total = sum(b.defense or 0 for b in outposts)
            if player.combat >= total:
                return base_attacks[0]
            return max(base_attacks, key=lambda a: by_index[a.card_id].cost)
        worth = [
            a for a in base_attacks
            if by_index[a.card_id].cost >= params["base_kill_min_cost"]
        ]
        if worth:
            return max(worth, key=lambda a: by_index[a.card_id].cost)
        return attack_player

    def _pending(self, game, player, opponent, actions, pending):
        params = self.params
        skip = next((a for a in actions if a.type == A.SKIP_DECISION), None)
        kinds = {a.type for a in actions}
        if A.DESTROY_BASE in kinds:
            options = [a for a in actions if a.type == A.DESTROY_BASE]
            return max(
                options,
                key=lambda a: ((a.card.is_outpost() if a.card else 0), (a.card.cost if a.card else 0)),
            )
        if A.SCRAP_CARD in kinds:
            options = [a for a in actions if a.type == A.SCRAP_CARD]
            own = [a for a in options if a.card_source in (CardSource.HAND, CardSource.DISCARD)]
            if own:
                ranked = []
                for a in own:
                    rank = self._junk_rank(a.card, game)
                    if rank is not None:
                        # discard pile first: it costs nothing this turn
                        ranked.append((0 if a.card_source == CardSource.DISCARD else 1, rank, a))
                if ranked:
                    return min(ranked, key=lambda item: item[:2])[2]
                if skip is not None:
                    return skip
                return min(own, key=lambda a: a.card.cost)
            trade = [a for a in options if a.card_source == CardSource.TRADE]
            deny = [a for a in trade
                    if a.card.cost >= params["deny_cost"] and a.card.cost > player.trade]
            if deny:
                return max(deny, key=lambda a: a.card.cost)
            return skip if skip is not None else trade[0]
        if A.DISCARD_CARDS in kinds:
            options = [a for a in actions if a.type == A.DISCARD_CARDS]
            if pending.mandatory:
                def keep_value(a):
                    rank = {"Viper": 0, "Scout": 1, "Explorer": 2}.get(a.card.name)
                    return (0, rank) if rank is not None else (1, a.card.cost)
                return min(options, key=keep_value)
            junk = [a for a in options if a.card.name in STARTERS]
            if junk:
                return junk[0]
            return skip if skip is not None else options[0]
        if A.BUY_CARD in kinds:
            options = [a for a in actions if a.type == A.BUY_CARD]
            return max(options, key=lambda a: (a.card.cost, self._buy_score(a.card, game, player)))
        if A.APPLY_EFFECT in kinds:
            options = [a for a in actions if a.type == A.APPLY_EFFECT]
            return max(options, key=lambda a: a.card.cost if a.card else 0)
        return skip if skip is not None else actions[0]


def held_out_agent_factory(style: str):
    """Return a ``name -> agent`` factory for one held-out style."""
    if style not in STYLE_PARAMETERS:
        raise ValueError(
            f"Unknown held-out style {style!r}. Valid: {', '.join(HELD_OUT_STYLES)}"
        )

    def factory(name: str) -> HeldOutAgent:
        return HeldOutAgent(name, style)

    return factory
