from src.engine.game import Game
from src.engine.actions import ActionType, Action, get_available_actions
from src.cards.effects import CardEffectType
from src.ai.agent import Agent


class HeuristicAgent(Agent):
    """Rule-based agent using core Star Realms strategy heuristics.

    Decision priority: play all cards → apply effects (draw first) →
    attack (lethal check, outposts, bases, player) → buy (scored) → end turn.
    """

    def __init__(self, name: str = "Heuristic", cli_interface=None):
        super().__init__(name, cli_interface)

    def make_decision(self, game_state: Game) -> Action:
        player = game_state.current_player
        actions = get_available_actions(game_state, player)

        if len(actions) == 1:
            return actions[0]

        # Pending decisions take a separate path
        pending = player.get_current_pending_set()
        if pending:
            return self._handle_pending(game_state, player, actions, pending)

        # Priority 1: always play cards from hand
        play_actions = [a for a in actions if a.type == ActionType.PLAY_CARD]
        if play_actions:
            return self._pick_best_play(play_actions, player)

        # Priority 2: apply available effects (draw effects first)
        effect_actions = [a for a in actions if a.type == ActionType.APPLY_EFFECT]
        if effect_actions:
            return self._pick_best_effect(effect_actions)

        # Priority 3: attack (with lethal detection)
        attack_action = self._pick_attack(actions, game_state, player)
        if attack_action:
            return attack_action

        # Priority 4: buy cards
        buy_action = self._pick_buy(actions, game_state, player)
        if buy_action:
            return buy_action

        # Priority 5: end turn
        return self._get_end_turn(actions)

    # ── Pending decisions ──────────────────────────────────────────────

    def _handle_pending(self, game_state: Game, player, actions, pending):
        """Handle forced or optional pending decisions (scrap, discard, destroy)."""
        action_types = {a.type for a in actions}

        # Destroy base: always destroy if we can
        if ActionType.DESTROY_BASE in action_types:
            destroys = [a for a in actions if a.type == ActionType.DESTROY_BASE]
            if destroys:
                return destroys[0]

        # Scrap decisions
        if ActionType.SCRAP_CARD in action_types:
            return self._handle_scrap_decision(game_state, player, actions, pending)

        # Discard from hand (forced by opponent effect)
        if ActionType.DISCARD_CARDS in action_types:
            return self._handle_discard_decision(player, actions)

        # Skip optional decisions we don't want to take
        skip = [a for a in actions if a.type == ActionType.SKIP_DECISION]
        if skip:
            return skip[0]

        return actions[0]

    def _handle_scrap_decision(self, game_state, player, actions, pending):
        """Choose what to scrap. Phase-aware: Vipers first early, Scouts later."""
        scrap_actions = [a for a in actions if a.type == ActionType.SCRAP_CARD]
        skip_actions = [a for a in actions if a.type == ActionType.SKIP_DECISION]

        if not scrap_actions:
            return skip_actions[0] if skip_actions else actions[0]

        total_deck_size = len(player.hand) + len(player.deck) + len(player.discard_pile) + len(player.played_cards)
        is_early_game = total_deck_size <= 14

        # Categorize scrap targets by priority
        scrap_vipers = [a for a in scrap_actions if a.card_id == "Viper"]
        scrap_scouts = [a for a in scrap_actions if a.card_id == "Scout"]
        scrap_explorers = [a for a in scrap_actions if a.card_id == "Explorer"]

        # Early game: scrap Vipers first (preserve economy for buying)
        if is_early_game:
            if scrap_vipers:
                return scrap_vipers[0]
            if scrap_scouts:
                return scrap_scouts[0]
            if scrap_explorers:
                return scrap_explorers[0]
        else:
            # Late game: scrap Scouts first (deck thinning, economy less critical)
            if scrap_scouts:
                return scrap_scouts[0]
            if scrap_vipers:
                return scrap_vipers[0]
            if scrap_explorers:
                return scrap_explorers[0]

        # Scrap from trade row if available (deny opponent good cards)
        trade_scraps = [a for a in scrap_actions if a.card_source == "trade"]
        if trade_scraps:
            # Scrap the most expensive trade row card to deny it
            trade_scraps.sort(key=lambda a: self._card_cost(a, game_state), reverse=True)
            best = trade_scraps[0]
            if self._card_cost(best, game_state) >= 5:
                return best

        # Skip if nothing worth scrapping
        if skip_actions:
            return skip_actions[0]
        return scrap_actions[0]

    def _handle_discard_decision(self, player, actions):
        """Discard the lowest-value card when forced."""
        discard_actions = [a for a in actions if a.type == ActionType.DISCARD_CARDS]
        if not discard_actions:
            return actions[0]
        # Discard the cheapest card (Scouts/Vipers first)
        discard_actions.sort(key=lambda a: self._card_value(a.card_id))
        return discard_actions[0]

    # ── Play cards ─────────────────────────────────────────────────────

    def _pick_best_play(self, play_actions, player):
        """Play cards that generate draw effects first for maximum information."""
        # Prefer cards with draw effects (play them first for more options)
        def play_priority(action):
            if not action.card:
                return 0
            draw_value = 0
            for effect in action.card.effects:
                if effect.effect_type == CardEffectType.DRAW:
                    draw_value += effect.value * 10
            # Slight preference for higher-cost cards
            return draw_value + (action.card.cost or 0)

        play_actions.sort(key=play_priority, reverse=True)
        return play_actions[0]

    # ── Apply effects ──────────────────────────────────────────────────

    def _pick_best_effect(self, effect_actions):
        """Apply effects in strategic order: draw > trade > combat > heal."""
        def effect_priority(action):
            if not action.card_effect:
                return 0
            etype = action.card_effect.effect_type
            value = action.card_effect.value or 0
            if etype == CardEffectType.DRAW:
                return 100 + value
            if etype == CardEffectType.TRADE:
                return 80 + value
            if etype == CardEffectType.COMBAT:
                return 60 + value
            if etype == CardEffectType.TARGET_DISCARD:
                return 55
            if etype == CardEffectType.DESTROY_BASE:
                return 50
            if etype == CardEffectType.HEAL:
                return 40 + value
            if etype == CardEffectType.SCRAP:
                return 30
            return 10

        effect_actions.sort(key=effect_priority, reverse=True)
        return effect_actions[0]

    # ── Attack ─────────────────────────────────────────────────────────

    def _pick_attack(self, actions, game_state, player):
        """Attack with lethal detection: kill player if possible, else smart targeting."""
        attack_player = [a for a in actions if a.type == ActionType.ATTACK_PLAYER]
        attack_base = [a for a in actions if a.type == ActionType.ATTACK_BASE]

        if not attack_player and not attack_base:
            return None

        # Lethal check: if we can kill the opponent, do it immediately
        if attack_player:
            opponent = game_state.get_opponent(player)
            if opponent and player.combat >= opponent.health:
                return attack_player[0]

        # Attack outposts first (required by game rules to unlock other targets)
        outpost_attacks = []
        if attack_base:
            opponent = game_state.get_opponent(player)
            if opponent:
                outpost_names = {b.name for b in opponent.bases if b.is_outpost()}
                outpost_attacks = [a for a in attack_base if a.card_id in outpost_names]

        if outpost_attacks:
            # Prefer destroying the cheapest outpost first (efficient combat use)
            outpost_attacks.sort(key=lambda a: self._base_defense(a, game_state))
            return outpost_attacks[0]

        # Attack non-outpost bases that are worth removing
        if attack_base:
            # Sort by defense (destroy cheapest first to preserve combat for player)
            attack_base.sort(key=lambda a: self._base_defense(a, game_state))
            return attack_base[0]

        # Direct damage to player
        if attack_player:
            return attack_player[0]

        return None

    # ── Buy cards ──────────────────────────────────────────────────────

    def _pick_buy(self, actions, game_state, player):
        """Buy the highest-value card using a scoring function with faction synergy."""
        buy_actions = [a for a in actions if a.type == ActionType.BUY_CARD]
        if not buy_actions:
            return None

        # Count faction distribution in player's full deck for synergy scoring
        faction_counts = self._count_factions(player)

        # Score and sort
        scored = []
        for action in buy_actions:
            score = self._score_card_for_buy(action, player, faction_counts, game_state)
            scored.append((score, action))

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_action = scored[0]

        # Skip buying if best option is weak (e.g., don't buy Explorer late game for nothing)
        if best_score <= 1:
            return None

        return best_action

    def _score_card_for_buy(self, action, player, faction_counts, game_state):
        """Score a card for purchase. Higher = more desirable."""
        card = action.card
        if not card:
            return 0

        score = 0.0

        # Base score: cost is a rough proxy for power (higher cost = better card)
        score += card.cost * 2

        # Explorer penalty — only buy as fallback, low inherent value
        if card.name == "Explorer":
            total_deck_size = (len(player.hand) + len(player.deck) +
                               len(player.discard_pile) + len(player.played_cards))
            if total_deck_size > 14:
                return 0  # Don't buy Explorers in late game
            return 2  # Marginal early game buy

        # Faction synergy: bonus if we already have cards of this faction
        if card.faction and isinstance(card.faction, str):
            existing = faction_counts.get(card.faction.lower(), 0)
            score += existing * 1.5  # Each existing ally card adds synergy value

        # Effect bonuses
        for effect in card.effects:
            if effect.effect_type == CardEffectType.DRAW:
                score += effect.value * 4  # Draw is king
            elif effect.effect_type == CardEffectType.COMBAT:
                score += effect.value * 1.5  # Combat is strong (aggro wins)
            elif effect.effect_type == CardEffectType.SCRAP and not effect.is_scrap_effect:
                score += 3  # Scrap ability (deck thinning) is very valuable
            elif effect.effect_type == CardEffectType.TARGET_DISCARD:
                score += 2  # Disruption
            elif effect.effect_type == CardEffectType.DESTROY_BASE:
                score += 2

        # Base/outpost bonus — persistent value across turns
        if card.card_type == "outpost":
            score += 4 + (card.defense or 0) * 0.5
        elif card.card_type == "base":
            score += 3 + (card.defense or 0) * 0.5

        return score

    def _count_factions(self, player):
        """Count faction occurrences across the player's full card pool."""
        counts = {}
        for zone in [player.hand, player.deck, player.discard_pile, player.played_cards, player.bases]:
            for card in zone:
                if card.faction and isinstance(card.faction, str):
                    faction = card.faction.lower()
                    counts[faction] = counts.get(faction, 0) + 1
        return counts

    # ── Helpers ─────────────────────────────────────────────────────────

    def _get_end_turn(self, actions):
        """Return END_TURN action, or last action as fallback."""
        for a in actions:
            if a.type == ActionType.END_TURN:
                return a
        return actions[-1]

    def _card_value(self, card_id):
        """Rough card value for discard priority (lower = discard first)."""
        if card_id == "Scout":
            return 1
        if card_id == "Viper":
            return 2
        if card_id == "Explorer":
            return 3
        return 10  # Keep purchased cards

    def _card_cost(self, action, game_state):
        """Get cost of a card from an action, looking it up from trade row if needed."""
        if action.card and hasattr(action.card, 'cost'):
            return action.card.cost or 0
        for card in game_state.trade_row:
            if card.name == action.card_id:
                return card.cost or 0
        return 0

    def _base_defense(self, action, game_state):
        """Get defense value of a base target."""
        opponent = game_state.get_opponent(game_state.current_player)
        if opponent:
            for base in opponent.bases:
                if base.name == action.card_id:
                    return base.defense or 0
        return 0
