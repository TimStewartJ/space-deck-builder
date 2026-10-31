"""Acting-player information projection and seeded hidden-state sampling."""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from src.ai.agent import Agent
from src.cards.card import Card
from src.engine.actions import Action, CardSource, PendingActionSet
from src.engine.game import Game
from src.engine.player import Player


class InformationSetError(ValueError):
    """Raised when public knowledge cannot describe a legal game state."""


@dataclass(frozen=True, slots=True)
class InformationSet:
    """A complete acting-player view with no true hidden allocation or order."""

    viewer_index: int
    player_hand_sizes: tuple[int, int]
    player_deck_sizes: tuple[int, int]
    public_owned_counts: tuple[tuple[tuple[int, int], ...], ...]
    public_known_hand_counts: tuple[tuple[tuple[int, int], ...], ...]
    public_trade_supply_counts: tuple[tuple[int, int], ...]
    fingerprint: str
    _canonical_game: Game = field(repr=False, compare=False)

    @property
    def canonical_game(self) -> Game:
        """The canonical public view this information set was built from.

        Hidden hand/deck allocations, deck order, and trade-deck order hold
        information-independent canonical arrangements. Treat the view as
        read-only; use :func:`determinize` for a playable sampled world.
        """
        return self._canonical_game


def _counter_items(counter: Counter[int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(index), int(count))
        for index, count in sorted(counter.items())
        if count
    )


def _physical_play_area(player: Player) -> list[Card]:
    cards = list(player.played_cards)
    played_ids = {id(card) for card in cards}
    cards.extend(
        base for base in player.bases if id(base) not in played_ids
    )
    return cards


def _physical_owned_cards(player: Player) -> list[Card]:
    return [
        *player.hand,
        *player.deck,
        *player.discard_pile,
        *_physical_play_area(player),
    ]


def _subtract_cards(
    counts: Counter[int],
    cards: Iterable[Card],
    *,
    context: str,
) -> Counter[int]:
    remaining = counts.copy()
    for card in cards:
        remaining[card.index] -= 1
        if remaining[card.index] < 0:
            raise InformationSetError(
                f"{context} contains more {card.name} cards than public "
                "ownership permits"
            )
        if remaining[card.index] == 0:
            del remaining[card.index]
    return remaining


def _subtract_instances(
    cards: Iterable[Card],
    removed: Iterable[Card],
    *,
    context: str,
) -> list[Card]:
    card_list = list(cards)
    remaining = {id(card): card for card in card_list}
    if len(remaining) != len(card_list):
        raise InformationSetError(
            f"{context} public ownership contains duplicate instances"
        )
    for card in removed:
        if remaining.pop(id(card), None) is None:
            raise InformationSetError(
                f"{context} contains a card outside public ownership"
            )
    return list(remaining.values())


def _make_card(game: Game, index: int) -> Card:
    template = game.card_templates.get(index)
    if template is None:
        raise InformationSetError(
            f"no printed card template is registered for index {index}"
        )
    return template.clone()


def _expand_counts(game: Game, counts: Counter[int]) -> list[Card]:
    return [
        _make_card(game, index)
        for index, count in sorted(counts.items())
        for _ in range(count)
    ]


def _clone_without_controllers(game: Game) -> Game:
    agent_memo = {
        id(player.agent): player.agent
        for player in game.players
        if player.agent is not None
    }
    clone = copy.deepcopy(game, agent_memo)
    for player in clone.players:
        player.agent = Agent(player.name)
        player.rng = None
    clone.rng = None
    return clone


def _effect_signature(effect) -> dict:
    return {
        "type": effect.effect_type.value,
        "value": int(effect.value),
        "applied": bool(effect.applied),
        "faction_requirement": effect.faction_requirement,
        "is_scrap": bool(effect.is_scrap_effect),
        "is_ally": bool(effect.is_ally_effect),
        "is_or": bool(effect.is_or_effect),
        "mandatory": bool(effect.is_mandatory),
        "children": [
            _effect_signature(child)
            for child in effect.child_effects or ()
        ],
    }


def _card_signature(card: Card) -> dict:
    return {
        "index": int(card.index),
        "effective_rule": int(card.effective_rule_card_index),
        "effective_faction": int(card.effective_faction),
        "copying": bool(card._is_copying),
        "effects": [_effect_signature(effect) for effect in card.effects],
    }


def _canonical_card_order(cards: Iterable[Card]) -> list[Card]:
    return sorted(
        cards,
        key=lambda card: json.dumps(
            _card_signature(card),
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def _effect_is_pristine(effect) -> bool:
    return (
        not effect.applied
        and all(
            _effect_is_pristine(child)
            for child in effect.child_effects or ()
        )
    )


def _card_is_pristine(card: Card) -> bool:
    return (
        not card._is_copying
        and card.effective_rule_card_index == card.index
        and card.effective_faction == card.faction
        and all(_effect_is_pristine(effect) for effect in card.effects)
    )


def _additional_value_signature(value):
    if isinstance(value, Card):
        return {"card": _card_signature(value)}
    if isinstance(value, dict):
        return {
            str(key): _additional_value_signature(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _action_signature(action: Action) -> dict:
    return {
        "type": action.type.value,
        "card_id": action.card_id,
        "card": (
            _card_signature(action.card)
            if action.card is not None
            else None
        ),
        "target_id": action.target_id,
        "source": (
            action.card_source.value if action.card_source is not None else None
        ),
        "effect": (
            _effect_signature(action.card_effect)
            if action.card_effect is not None
            else None
        ),
        "additional": _additional_value_signature(
            action.additional_params or {}
        ),
    }


def _pending_signature(pending: PendingActionSet) -> dict:
    return {
        "actions": [_action_signature(action) for action in pending.actions],
        "decisions_left": int(pending.decisions_left),
        "mandatory": bool(pending.mandatory),
        "resolved_count": int(pending.resolved_count),
        "draw_on_complete": bool(pending.on_complete_draw),
    }


def _fingerprint_payload(game: Game, viewer_index: int) -> dict:
    players = []
    for player in game.players:
        players.append(
            {
                "name": player.name,
                "hand": [_card_signature(card) for card in player.hand],
                "deck": [_card_signature(card) for card in player.deck],
                "ordered_top": [
                    int(card.index) for card in player.ordered_top
                ],
                "discard": [
                    _card_signature(card) for card in player.discard_pile
                ],
                "played": [
                    _card_signature(card) for card in player.played_cards
                ],
                "bases": [
                    _card_signature(card) for card in player.bases
                ],
                "health": int(player.health),
                "hand_size": int(player.hand_size),
                "trade": int(player.trade),
                "combat": int(player.combat),
                "authority_gained": int(player.authority_gained),
                "discard_debt": int(
                    player.pending_start_of_turn_discards
                ),
                "next_ship_to_top": int(player.next_ship_to_top),
                "faction_plays": [int(count) for count in player.faction_plays],
                "pending": [
                    _pending_signature(pending)
                    for pending in player.pending_action_sets
                ],
                "owned": _counter_items(player.public_owned_cards),
                "owned_instances": [
                    _card_signature(card)
                    for card in _canonical_card_order(
                        player.public_owned_card_instances
                    )
                ],
                "known_hand": [
                    _card_signature(card)
                    for card in player.public_known_hand
                ],
            }
        )
    return {
        "viewer": viewer_index,
        "players": players,
        "trade_row": [
            _card_signature(card) for card in game.trade_row
        ],
        "trade_deck": [
            _card_signature(card) for card in game.trade_deck
        ],
        "trade_supply": _counter_items(game.public_trade_supply),
        "explorer_pile": [
            _card_signature(card) for card in game.explorer_pile
        ],
        "current_turn": int(game.current_turn),
        "first_player": game.first_player_name,
        "total_turns": int(game.stats.total_turns),
        "is_game_over": bool(game.is_game_over),
        "winner": game.stats.winner,
        "config": game.config.to_dict(),
        "card_names": tuple(game.card_names),
        "card_index_map": tuple(sorted((game.card_index_map or {}).items())),
    }


def _fingerprint(game: Game, viewer_index: int) -> str:
    encoded = json.dumps(
        _fingerprint_payload(game, viewer_index),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_pending_visibility(game: Game, viewer: Player) -> None:
    for player in game.players:
        if player is not viewer and player.pending_action_sets:
            raise InformationSetError(
                "only the acting player may have pending decisions"
            )
    for pending in viewer.pending_action_sets:
        for action in pending.actions:
            if action.card_source == CardSource.OPPONENT:
                raise InformationSetError(
                    "pending decisions may not expose an opponent's hidden hand"
                )


def _validate_card_identity(game: Game) -> None:
    index_map = game.card_index_map
    required = {"Scout", "Viper", "Explorer"}
    if (
        not index_map
        or not required.issubset(index_map)
        or len(set(index_map.values())) != len(index_map)
    ):
        raise InformationSetError(
            "information projection requires an unambiguous canonical "
            "card_index_map"
        )
    names_by_index = {index: name for name, index in index_map.items()}
    if game.card_template_conflicts:
        raise InformationSetError(
            "conflicting card templates share an index: "
            f"{game.card_template_conflicts}"
        )
    for index, template in game.card_templates.items():
        expected = names_by_index.get(index)
        if expected is None or expected != template.name:
            raise InformationSetError(
                f"card index {index} ambiguously identifies "
                f"{template.name!r} instead of {expected!r}"
            )
    for player in game.players:
        instances = player.public_owned_card_instances
        if len({id(card) for card in instances}) != len(instances):
            raise InformationSetError(
                f"{player.name} public ownership repeats a card instance"
            )
        instance_counts = Counter(card.index for card in instances)
        if instance_counts != player.public_owned_cards:
            raise InformationSetError(
                f"{player.name} public ownership counts and instances differ"
            )
        for card in instances:
            expected = names_by_index.get(card.index)
            if expected != card.name:
                raise InformationSetError(
                    f"owned card index {card.index} identifies "
                    f"{card.name!r} instead of {expected!r}"
                )
        in_play = {id(card) for card in _physical_play_area(player)}
        if any(
            not _card_is_pristine(card)
            for card in instances
            if id(card) not in in_play
        ):
            raise InformationSetError(
                f"{player.name} has mutable card state outside public play"
            )
    if any(
        not _card_is_pristine(card)
        for card in [*game.trade_row, *game.explorer_pile]
    ):
        raise InformationSetError(
            "public supply contains mutable off-board card state"
        )


def project_information_set(
    game: Game,
    viewer: Player | None = None,
) -> InformationSet:
    """Project one decision state without reading true hidden card identities."""
    if len(game.players) != 2:
        raise InformationSetError("information search requires two players")
    _validate_card_identity(game)
    viewer = viewer or game.current_player
    if viewer is not game.current_player:
        raise InformationSetError("viewer must be the acting player")
    try:
        viewer_index = game.players.index(viewer)
    except ValueError as exc:
        raise InformationSetError("viewer is not part of this game") from exc
    if not game.public_trade_supply_initialized:
        raise InformationSetError(
            "game has no initialized public trade-supply ledger"
        )
    if any(
        not player.public_ownership_initialized
        for player in game.players
    ):
        raise InformationSetError(
            "game has no initialized public ownership ledger"
        )
    _validate_pending_visibility(game, viewer)

    template = _clone_without_controllers(game)
    hand_sizes = tuple(len(player.hand) for player in game.players)
    deck_sizes = tuple(len(player.deck) for player in game.players)
    known_hand_counts = tuple(
        _counter_items(Counter(
            card.index for card in player.public_known_hand
        ))
        for player in game.players
    )

    for player_index, original in enumerate(game.players):
        target = template.players[player_index]
        owned_instances = target.public_owned_card_instances
        visible_play = _physical_play_area(target)
        if player_index == viewer_index:
            deck_cards = _subtract_instances(
                owned_instances,
                [
                    *target.hand,
                    *target.discard_pile,
                    *visible_play,
                ],
                context=f"{original.name} visible zones",
            )
            ordered_top = list(target.ordered_top)
            unordered_deck = _subtract_instances(
                deck_cards,
                ordered_top,
                context=f"{original.name} known top-deck segment",
            )
            if len(unordered_deck) + len(ordered_top) != len(original.deck):
                raise InformationSetError(
                    f"{original.name} public ownership does not match deck size"
                )
            target.deck = [
                *_canonical_card_order(unordered_deck),
                *ordered_top,
            ]
            target.ordered_top = ordered_top
            continue

        hidden_cards = _subtract_instances(
            owned_instances,
            [
                *target.discard_pile,
                *visible_play,
            ],
            context=f"{original.name} public zones",
        )
        if len(hidden_cards) != (
            len(original.hand) + len(original.deck)
        ):
            raise InformationSetError(
                f"{original.name} public ownership does not match hidden "
                "hand and deck sizes"
            )
        known_hand_cards = list(target.public_known_hand)
        ordered_top = list(target.ordered_top)
        unknown_cards = _subtract_instances(
            hidden_cards,
            [*known_hand_cards, *ordered_top],
            context=f"{original.name} publicly known hand",
        )
        unknown_cards = _canonical_card_order(unknown_cards)
        unknown_hand_size = len(original.hand) - len(known_hand_cards)
        unordered_deck_size = len(original.deck) - len(ordered_top)
        if unknown_hand_size < 0 or unordered_deck_size < 0:
            raise InformationSetError(
                f"{original.name} public constraints exceed hidden zone sizes"
            )
        if len(unknown_cards) != unknown_hand_size + unordered_deck_size:
            raise InformationSetError(
                f"{original.name} hidden card pool cannot fill hand and deck"
            )
        target.hand = [
            *known_hand_cards,
            *unknown_cards[:unknown_hand_size],
        ]
        target.deck = [
            *unknown_cards[unknown_hand_size:],
            *ordered_top,
        ]
        target.ordered_top = ordered_top

    trade_deck_counts = _subtract_cards(
        Counter(game.public_trade_supply),
        game.trade_row,
        context="visible trade row",
    )
    if sum(trade_deck_counts.values()) != len(game.trade_deck):
        raise InformationSetError(
            "public trade supply does not match trade-deck size"
        )
    template.trade_deck = _expand_counts(template, trade_deck_counts)
    template.current_player = template.players[template.current_turn]

    return InformationSet(
        viewer_index=viewer_index,
        player_hand_sizes=hand_sizes,
        player_deck_sizes=deck_sizes,
        public_owned_counts=tuple(
            _counter_items(player.public_owned_cards)
            for player in game.players
        ),
        public_known_hand_counts=known_hand_counts,
        public_trade_supply_counts=_counter_items(
            game.public_trade_supply
        ),
        fingerprint=_fingerprint(template, viewer_index),
        _canonical_game=template,
    )


def determinize(information: InformationSet, seed: int) -> Game:
    """Sample one playable hidden world from an acting-player information set."""
    branch = copy.deepcopy(information._canonical_game)
    rng = random.Random(seed)
    branch.rng = rng
    for player in branch.players:
        player.rng = rng

    viewer_index = information.viewer_index
    for player_index, player in enumerate(branch.players):
        top_count = len(player.ordered_top)
        if player_index == viewer_index:
            unordered = player.deck[:-top_count] if top_count else list(
                player.deck
            )
            rng.shuffle(unordered)
            player.deck = [
                *unordered,
                *player.ordered_top,
            ]
            continue

        known_hand = list(player.public_known_hand)
        known_hand_count = len(known_hand)
        unknown_cards = [
            *player.hand[known_hand_count:],
            *(
                player.deck[:-top_count]
                if top_count
                else player.deck
            ),
        ]
        rng.shuffle(unknown_cards)
        unknown_hand_size = (
            information.player_hand_sizes[player_index]
            - known_hand_count
        )
        player.hand = [
            *known_hand,
            *unknown_cards[:unknown_hand_size],
        ]
        player.deck = [
            *unknown_cards[unknown_hand_size:],
            *player.ordered_top,
        ]

    rng.shuffle(branch.trade_deck)
    branch.current_player = branch.players[branch.current_turn]
    return branch


def assert_public_knowledge_consistent(game: Game) -> None:
    """Diagnostic oracle proving public ledgers match the complete true state."""
    if len(game.players) != 2:
        raise InformationSetError("public knowledge requires two players")
    for player in game.players:
        if not player.public_ownership_initialized:
            raise InformationSetError(
                f"{player.name} ownership ledger is not initialized"
            )
        actual_owned = Counter(
            card.index for card in _physical_owned_cards(player)
        )
        if actual_owned != player.public_owned_cards:
            raise InformationSetError(
                f"{player.name} ownership ledger differs from physical zones"
            )
        physical_instances = Counter(
            id(card) for card in _physical_owned_cards(player)
        )
        public_instances = Counter(
            id(card) for card in player.public_owned_card_instances
        )
        if physical_instances != public_instances:
            raise InformationSetError(
                f"{player.name} ownership instances differ from physical zones"
            )
        for card in player.public_known_hand:
            if card not in player.hand:
                raise InformationSetError(
                    f"{player.name} known hand contains a card outside its hand"
                )
        if player.ordered_top:
            if len(player.ordered_top) > len(player.deck):
                raise InformationSetError(
                    f"{player.name} known top exceeds deck size"
                )
            actual_top = player.deck[-len(player.ordered_top):]
            if any(
                expected is not actual
                for expected, actual in zip(
                    player.ordered_top,
                    actual_top,
                )
            ):
                raise InformationSetError(
                    f"{player.name} ordered top is not the deck suffix"
                )

    if not game.public_trade_supply_initialized:
        raise InformationSetError("trade-supply ledger is not initialized")
    actual_supply = Counter(
        card.index for card in [*game.trade_deck, *game.trade_row]
    )
    if actual_supply != game.public_trade_supply:
        raise InformationSetError(
            "public trade-supply ledger differs from physical zones"
        )
