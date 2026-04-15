from typing import List
from src.ai.agent import Agent
from src.cards.effects import CardEffectType
from src.utils import logger as _logger
from src.utils.logger import log
from src.cards.card import Card
from src.config import GameConfig
from src.engine.player import Player
from src.engine.actions import ActionType, Action
from src.engine.game_stats import GameStats


def _pending_action_matches(pending: Action, executed: Action) -> bool:
    """Check if an executed action semantically matches a pending action.

    Compares only the fields relevant for pending action identity
    (type, card_id, card_source, target_id), ignoring object references
    like card= that may differ between legacy and new-style callers.
    """
    return (
        pending.type == executed.type
        and pending.card_id == executed.card_id
        and pending.card_source == executed.card_source
        and pending.target_id == executed.target_id
    )


class Game:
    def __init__(self, cards=None, card_names: list[str] | None = None,
                 game_config: GameConfig | None = None,
                 card_index_map: dict[str, int] | None = None):
        self.config = game_config or GameConfig()
        self.players: List[Player] = []
        self.current_turn = 0
        self.is_game_over = False
        self.trade_deck = [card.clone() for card in cards] if cards else []
        self.card_names = card_names if card_names else []
        self.card_index_map = card_index_map
        self.trade_row: List[Card] = []
        self.explorer_pile: List[Card] = []
        self.is_running = False
        self.current_player: Player = Player("none", Agent("none"),
                                             starting_health=self.config.starting_health,
                                             hand_size=self.config.hand_size)
        self.stats = GameStats()
        self.first_player_name = None

    def start_game(self):
        if not self.trade_deck:
            raise ValueError("Cannot start game without cards")
        self.setup_players()
        self.setup_trade_row()
        self.setup_explorer_pile()
        self.is_game_over = False
        self.is_running = True
        self.current_player = self.players[self.current_turn]

    def setup_trade_row(self):
        self.shuffle_trade_deck()
        self.fill_trade_row()

    def setup_explorer_pile(self):
        from src.cards.card import Card
        from src.cards.effects import Effect, CardEffectType

        explorer_idx = self.card_index_map["Explorer"] if self.card_index_map else len(self.card_names) - 1
        for _ in range(self.config.explorer_count):
            explorer_card = Card(
                "Explorer",
                explorer_idx,
                2,
                [
                    Effect(CardEffectType.TRADE, 2),
                    Effect(CardEffectType.COMBAT, 2, is_scrap_effect=True),
                ],
                "ship",
            )
            self.explorer_pile.append(explorer_card)
    def fill_trade_row(self):
        while len(self.trade_row) < self.config.trade_row_size and self.trade_deck:
            card = self.trade_deck.pop()
            self.trade_row.append(card)
            if not _logger.disabled:
                log(f"Added {card.name} to trade row", v=True)

    def shuffle_trade_deck(self):
        import random
        random.shuffle(self.trade_deck)

    def setup_players(self):
        # Shuffle players to randomize turn order
        import random
        random.shuffle(self.players)

        is_first_player = True

        for player in self.players:
            # Initialize statistics for this player
            self.stats.add_player(player.name)
            # Initialize each player's starting deck with 8 Scouts and 2 Vipers
            starting_deck = self.create_starting_deck()
            player.deck.extend(starting_deck)
            player.shuffle_deck()
            # Draw initial hand
            starting_hand_size = (self.config.first_player_hand_size
                                  if is_first_player else self.config.hand_size)
            for _ in range(starting_hand_size):
                player.draw_card()
                self.stats.record_card_draw(player.name)
            if is_first_player:
                self.first_player_name = player.name
                is_first_player = False

    def create_starting_deck(self):
        from src.cards.effects import Effect
        from src.cards.card import Card
        starting_deck = []
        scout_idx = self.card_index_map["Scout"] if self.card_index_map else len(self.card_names) - 2
        viper_idx = self.card_index_map["Viper"] if self.card_index_map else len(self.card_names) - 1
        for _ in range(self.config.num_scouts):
            starting_deck.append(Card("Scout", scout_idx, 0, [Effect(CardEffectType.TRADE, 1)], "ship"))
        for _ in range(self.config.num_vipers):
            starting_deck.append(Card("Viper", viper_idx, 0, [Effect(CardEffectType.COMBAT, 1)], "ship"))
        return starting_deck
    
    def step(self, action: Action | None = None):
        # Remember who acted
        actor = self.current_player
        # Execute the action & advance
        self.next_step(action)
        done = self.is_game_over
        return done

    def needs_decision(self) -> bool:
        """Check if the game is waiting for a player decision."""
        return not self.is_game_over

    def get_decision_context(self) -> tuple['Player', list[Action]] | None:
        """Return (player, available_actions) if a decision is needed, else None."""
        if self.is_game_over:
            return None
        from src.engine.actions import get_available_actions
        available = get_available_actions(self, self.current_player)
        return (self.current_player, available)

    def apply_decision(self, action: Action) -> bool:
        """Apply an externally-chosen action. Returns True if game is over."""
        self.next_step(action)
        return self.is_game_over

    def next_step(self, action: Action | None = None):
        if self.is_game_over:
            return
        
        turn_ended = False

        if action is None:
            # Get player decision (through UI or AI)
            action = self.current_player.make_decision(self)

        if not _logger.disabled:
            log(f"Getting action for {self.current_player.name}", v=True)

        if action:
            turn_ended = self.execute_action(action)
        
        if turn_ended:
            if not _logger.disabled:
                log(f"Ended turn for {self.current_player.name}", v=True)

            # Tell player turn is over
            self.current_player.reset_resources()
            self.current_player.end_turn()

            # Increment turn counter and move onto next player
            self.stats.total_turns += 1
            if self.stats.total_turns > self.config.turn_cap:
                if not _logger.disabled:
                    log(f"Game has exceeded {self.config.turn_cap} turns, ending game.", v=True)
                self.end_game()
            self.current_turn = (self.current_turn + 1) % len(self.players)
            self.current_player = self.players[self.current_turn]

        return action
        
    
    def execute_action(self, action: Action):
        """Execute a player's action and update game state.
        
        Uses direct card references (action.card) for O(1) resolution
        when available, falling back to name-based scanning for legacy
        callers that don't set card refs.
        """
        _log_enabled = not _logger.disabled
        if _log_enabled:
            log(f"{self.current_player.name} executing action: {action}", v=True)

        # Handle pending action sets — use identity first, then semantic match
        pending_set = self.current_player.get_current_pending_set()
        pending_set_completed = False
        pending_skipped = False
        pending_action_matched = False
        if pending_set:
            if action.type == ActionType.SKIP_DECISION and not pending_set.mandatory:
                pending_skipped = True
            else:
                # Remove by identity first (fast path for action context resolvers),
                # then by semantic match (type + card_id + source/target) for legacy
                for idx, pending_action in enumerate(pending_set.actions):
                    if pending_action is action or _pending_action_matches(pending_action, action):
                        pending_set.actions.pop(idx)
                        pending_set.decisions_left -= 1
                        pending_action_matched = True
                        break
                if pending_set.decisions_left <= 0:
                    pending_set_completed = True

        if action.type == ActionType.END_TURN:
            return True

        elif action.type == ActionType.PLAY_CARD:
            # Use direct card reference when available
            card = action.card
            if card is not None and card in self.current_player.hand:
                self.current_player.play_card(card)
            else:
                # Fallback: scan by name for legacy callers
                card = None
                for c in self.current_player.hand:
                    if c.name == action.card_id:
                        card = c
                        self.current_player.play_card(c)
                        break
            if card is not None:
                self.stats.record_card_play(self.current_player.name)
                if _log_enabled:
                    log(f"{self.current_player.name} played {card.name}", v=True)
                # Apply the newly played card's own auto-applicable effects,
                # plus any other cards in play with unapplied simple effects
                # (e.g. bases whose effects reset each turn).
                self._apply_auto_effects(card)

        elif action.type == ActionType.APPLY_EFFECT and action.card_effect is not None:
            # Resolve source card for scrap-from-play effects
            source_card = action.card
            if source_card is None:
                for c in self.current_player.played_cards + self.current_player.bases:
                    if c.name == action.card_id:
                        source_card = c
                        break
            action.card_effect.apply(self, self.current_player, source_card)
            if _log_enabled:
                log(f"{self.current_player.name} applied effect: {action.card_effect}", v=True)

        elif action.type == ActionType.BUY_CARD:
            card = action.card
            if card is not None:
                # Direct reference path — check if it's an explorer or trade row card
                if self.explorer_pile and card is self.explorer_pile[-1]:
                    self.explorer_pile.pop()
                    self.current_player.trade -= card.cost
                    self.current_player.discard_pile.append(card)
                    self.stats.record_card_buy(self.current_player.name)
                    if _log_enabled:
                        log(f"{self.current_player.name} bought {card.name} for {card.cost} trade", v=True)
                    return False
                # Find in trade row by identity
                for i, tr_card in enumerate(self.trade_row):
                    if tr_card is card:
                        self.current_player.trade -= card.cost
                        self.current_player.discard_pile.append(card)
                        self.stats.record_card_buy(self.current_player.name)
                        if _log_enabled:
                            log(f"{self.current_player.name} bought {card.name} for {card.cost} trade", v=True)
                        self.trade_row.pop(i)
                        if self.trade_deck:
                            new_card = self.trade_deck.pop()
                            self.trade_row.append(new_card)
                            self.stats.trade_row_refreshes += 1
                            if _log_enabled:
                                log(f"Added {new_card.name} to trade row", v=True)
                        return False
            # Fallback: name-based scan for legacy callers
            if action.card_id == "Explorer":
                if self.explorer_pile:
                    card = self.explorer_pile.pop()
                    self.current_player.trade -= card.cost
                    self.current_player.discard_pile.append(card)
                    self.stats.record_card_buy(self.current_player.name)
                    if _log_enabled:
                        log(f"{self.current_player.name} bought {card.name} for {card.cost} trade", v=True)
                    return False
            for i, card in enumerate(self.trade_row):
                if card.name == action.card_id and self.current_player.trade >= card.cost:
                    self.current_player.trade -= card.cost
                    self.current_player.discard_pile.append(card)
                    self.stats.record_card_buy(self.current_player.name)
                    if _log_enabled:
                        log(f"{self.current_player.name} bought {card.name} for {card.cost} trade", v=True)
                    self.trade_row.pop(i)
                    if self.trade_deck:
                        new_card = self.trade_deck.pop()
                        self.trade_row.append(new_card)
                        self.stats.trade_row_refreshes += 1
                        if _log_enabled:
                            log(f"Added {new_card.name} to trade row", v=True)
                    break

        elif action.type == ActionType.ATTACK_BASE:
            # Use direct card reference when available
            target_base = action.card
            if target_base is not None:
                for player in self.players:
                    if player is not self.current_player and target_base in player.bases:
                        if target_base.defense and self.current_player.combat >= target_base.defense:
                            self.current_player.combat -= target_base.defense
                            player.bases.remove(target_base)
                            player.discard_pile.append(target_base)
                            player.invalidate_faction_cache()
                            self.stats.record_base_destroy(self.current_player.name)
                            if _log_enabled:
                                log(f"{self.current_player.name} destroyed {player.name}'s {target_base.name}", v=True)
                        break
            else:
                # Fallback: name-based scan
                for player in self.players:
                    if player != self.current_player:
                        for base in player.bases:
                            if base.name == action.target_id and base.defense and self.current_player.combat >= base.defense:
                                self.current_player.combat -= base.defense
                                player.bases.remove(base)
                                player.discard_pile.append(base)
                                player.invalidate_faction_cache()
                                self.stats.record_base_destroy(self.current_player.name)
                                if _log_enabled:
                                    log(f"{self.current_player.name} destroyed {player.name}'s {base.name}", v=True)
                                break

        elif action.type == ActionType.DESTROY_BASE:
            # Use direct card reference when available
            target_base = action.card
            if target_base is not None:
                for player in self.players:
                    if player is not self.current_player and target_base in player.bases:
                        self.stats.record_base_destroy(self.current_player.name)
                        player.bases.remove(target_base)
                        player.discard_pile.append(target_base)
                        player.invalidate_faction_cache()
                        if _log_enabled:
                            log(f"{self.current_player.name} destroyed {player.name}'s {target_base.name}", v=True)
                        break
            else:
                # Fallback: name-based scan
                for player in self.players:
                    if player != self.current_player:
                        for base in player.bases:
                            if base.name == action.target_id:
                                self.stats.record_base_destroy(self.current_player.name)
                                player.bases.remove(base)
                                player.discard_pile.append(base)
                                player.invalidate_faction_cache()
                                if _log_enabled:
                                    log(f"{self.current_player.name} destroyed {player.name}'s {base.name}", v=True)
                                break

        elif action.type == ActionType.ATTACK_PLAYER:
            # Find target by target_id, fall back to sole opponent in 2-player
            target = None
            if action.target_id:
                for player in self.players:
                    if player is not self.current_player and player.name == action.target_id:
                        target = player
                        break
            if target is None:
                target = self.get_opponent(self.current_player)
            if target is not None:
                if not any(b.is_outpost() for b in target.bases):
                    damage = self.current_player.combat
                    target.health -= damage
                    self.current_player.combat = 0
                    self.stats.record_damage(self.current_player.name, damage)
                    if _log_enabled:
                        log(f"{self.current_player.name} attacked {target.name} for {damage} damage", v=True)
                    if target.health <= 0:
                        if _log_enabled:
                            log(f"{target.name} has been defeated!", v=True)
                        self.is_game_over = True
                        self.stats.end_game(self.current_player.name)

        elif action.type == ActionType.SCRAP_CARD and action.card_source is not None:
            self.stats.record_card_scrap(self.current_player.name, action.card_source)
            target_card = action.card
            if action.card_source == 'hand':
                if target_card is not None and target_card in self.current_player.hand:
                    self.current_player.hand.remove(target_card)
                else:
                    for card in self.current_player.hand:
                        if card.name == action.card_id:
                            target_card = card
                            self.current_player.hand.remove(card)
                            break
                if _log_enabled and target_card:
                    log(f"{self.current_player.name} scrapped {target_card.name} from hand", v=True)
            elif action.card_source == 'discard':
                if target_card is not None and target_card in self.current_player.discard_pile:
                    self.current_player.discard_pile.remove(target_card)
                else:
                    for card in self.current_player.discard_pile:
                        if card.name == action.card_id:
                            target_card = card
                            self.current_player.discard_pile.remove(card)
                            break
                if _log_enabled and target_card:
                    log(f"{self.current_player.name} scrapped {target_card.name} from discard pile", v=True)
            elif action.card_source == 'trade':
                if target_card is not None:
                    for i, card in enumerate(self.trade_row):
                        if card is target_card:
                            self.trade_row.pop(i)
                            break
                else:
                    for i, card in enumerate(self.trade_row):
                        if card.name == action.card_id:
                            target_card = card
                            self.trade_row.pop(i)
                            break
                if _log_enabled and target_card:
                    log(f"{self.current_player.name} scrapped {target_card.name} from trade row", v=True)
                # Refill trade row after all scraps from the pending set are done
                if not pending_set or pending_set_completed:
                    self.fill_trade_row()
            # Return explorers to the explorer pile
            if target_card and target_card.name == "Explorer":
                self.explorer_pile.append(target_card)

        elif action.type == ActionType.DISCARD_CARDS:
            # Determine target: opponent's hand if forced discard, else current player
            if action.card_source == "opponent":
                opponent = self.get_opponent(self.current_player)
                if opponent is None:
                    return False
                target_hand = opponent.hand
                target_discard = opponent.discard_pile
                self.stats.record_cards_discarded_from_hand(opponent.name, 1)
            else:
                target_hand = self.current_player.hand
                target_discard = self.current_player.discard_pile
                self.stats.record_cards_discarded_from_hand(self.current_player.name, 1)
            target_card = action.card
            if target_card is not None and target_card in target_hand:
                target_hand.remove(target_card)
                target_discard.append(target_card)
            else:
                for card in target_hand:
                    if card.name == action.card_id:
                        target_card = card
                        target_hand.remove(card)
                        target_discard.append(card)
                        break
            if _log_enabled and target_card:
                log(f"{self.current_player.name} discarded {target_card.name} from {'opponent' if action.card_source == 'opponent' else 'hand'}", v=True)

        # Finalize pending set after action execution so that card moves
        # (discard, scrap) happen before any completion effects (draw-on-complete).
        # Only count actions that actually matched and consumed a pending action.
        if pending_set and (pending_skipped or pending_set_completed):
            if pending_action_matched:
                pending_set.resolved_count += 1
            self._complete_pending_set(pending_set)
        elif pending_set and pending_action_matched and not pending_set_completed:
            pending_set.resolved_count += 1

        return False  # Turn continues

    # Set of effect types that can be auto-applied without player choice
    _AUTO_APPLY_TYPES = frozenset((
        CardEffectType.COMBAT,
        CardEffectType.TRADE,
        CardEffectType.HEAL,
        CardEffectType.DRAW,
        CardEffectType.TARGET_DISCARD,
        CardEffectType.PARENT,
    ))

    def _apply_auto_effects(self, played_card):
        """Apply simple auto-applicable effects on the played card and any
        other cards in play with unapplied effects (e.g. bases whose effects
        reset each turn).

        Skips scrap effects (must be explicitly chosen), OR effects
        (require a choice), and ally-gated effects (surfaced as APPLY_EFFECT).
        The applied guard in Effect.apply() prevents double-application.
        """
        for pc in self.current_player.played_cards:
            for effect in pc.effects:
                if (
                    effect.effect_type in self._AUTO_APPLY_TYPES
                    and not effect.applied
                    and not effect.is_or_effect
                    and not effect.is_scrap_effect
                    and not effect.faction_requirement
                ):
                    effect.apply(self, self.current_player, pc)

    def play(self):
        self.start_game()
        while not self.is_game_over:
            self.next_step()
        return self.get_winner()

    def end_game(self):
        self.is_game_over = True
        self.is_running = False
        # Logic to determine the winner and end the game
        pass

    def _complete_pending_set(self, pending_set):
        """Advance past a completed/skipped pending set and run completion effects.

        Called AFTER the final action in the set has executed, so card moves
        (discard, scrap) have already happened before any draw-on-complete.
        """
        if pending_set.on_complete_draw and pending_set.resolved_count > 0:
            for _ in range(pending_set.resolved_count):
                self.current_player.draw_card()
                self.stats.record_card_draw(self.current_player.name)
        self.current_player.advance_pending_set()

    def add_player(self, name: str, agent: 'Agent'):
        if len(self.players) >= 4:
            raise ValueError("Maximum number of players reached")
        from src.engine.player import Player
        player = Player(name, agent,
                        starting_health=self.config.starting_health,
                        hand_size=self.config.hand_size)
        self.players.append(player)
        return player
        
    def get_opponent(self, player: Player):
        """Get the opponent of the given player"""
        for p in self.players:
            if p != player:
                return p
        return None
        
    def get_winner(self):
        if self.is_game_over:
            # Determine winner based on remaining health
            winner = max(self.players, key=lambda p: p.health)
            return winner.name
        return None
