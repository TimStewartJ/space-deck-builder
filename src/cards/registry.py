"""Immutable card registry — single source of truth for card identity and metadata.

The registry is built once at startup from the CSV loader output and shared
across all games, encoders, and agents. It owns the canonical card_names
list and card_index_map, eliminating per-call rebuilds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.cards.card import Card, CardType
from src.cards.factions import Faction, faction_display


@dataclass(frozen=True, slots=True)
class CardDef:
    """Immutable card definition — static metadata only.

    Holds all fields that are constant across games so that per-game
    Card objects can reference this definition by card_def_id instead
    of carrying their own copies of name/cost/faction/etc.
    """
    card_def_id: int
    name: str
    cost: int
    card_type: CardType
    defense: Optional[int] = None
    faction: Faction = Faction.NONE
    ally_factions: Faction = Faction.NONE
    set_name: Optional[str] = None


class CardRegistry:
    """Immutable, globally-shared card metadata.

    Built once from the loader output + starter card names. Provides:
    - card_names: canonical ordered list (matches action/state encoding)
    - card_index_map: O(1) name → index lookup
    - CardDef lookup by id or name
    """

    def __init__(self, card_defs: list[CardDef], starter_names: list[str]):
        self._card_defs: tuple[CardDef, ...] = tuple(card_defs)
        self._by_id: dict[int, CardDef] = {cd.card_def_id: cd for cd in card_defs}

        # Build canonical card_names: unique trade-deck names + starters
        seen: dict[str, int] = {}
        names: list[str] = []
        for cd in card_defs:
            if cd.name not in seen:
                seen[cd.name] = cd.card_def_id
                names.append(cd.name)
        for starter in starter_names:
            if starter not in seen:
                names.append(starter)
                seen[starter] = -1  # starters may not have CardDefs yet

        self.card_names: list[str] = names
        self.card_index_map: dict[str, int] = {
            name: i for i, name in enumerate(names)
        }
        self.num_cards: int = len(names)

    def get(self, card_def_id: int) -> CardDef:
        """Look up a CardDef by its unique ID."""
        return self._by_id[card_def_id]

    def get_by_name(self, name: str) -> CardDef | None:
        """Look up the first CardDef matching a name, or None."""
        for cd in self._card_defs:
            if cd.name == name:
                return cd
        return None

    @property
    def card_defs(self) -> tuple[CardDef, ...]:
        return self._card_defs

    def __len__(self) -> int:
        return self.num_cards


def build_registry(
    cards: list[Card],
    starter_names: list[str],
) -> CardRegistry:
    """Build a CardRegistry from loader output and starter card names.

    Deduplicates cards by name (the loader produces multiple copies per
    quantity). Each unique card name gets one CardDef with a stable
    card_def_id matching its position in the deduplicated list.

    Args:
        cards: Trade deck cards from load_trade_deck_cards().
        starter_names: Names of starter/explorer cards (e.g. ["Scout", "Viper", "Explorer"]).

    Returns:
        A CardRegistry with canonical card_names matching the existing
        DataConfig.get_card_names() ordering.
    """
    seen: set[str] = set()
    card_defs: list[CardDef] = []
    index = 0

    for card in cards:
        if card.name in seen:
            continue
        seen.add(card.name)

        card_defs.append(CardDef(
            card_def_id=index,
            name=card.name,
            cost=card.cost,
            card_type=card.card_type,
            defense=card.defense,
            faction=card.faction,
            ally_factions=card.ally_factions,
            set_name=card.set,
        ))
        index += 1

    # Canonical metadata for starter/explorer cards that aren't in the trade deck.
    # These must match the definitions in Game.create_starting_deck() and
    # Game.setup_explorer_pile().
    _STARTER_DEFS: dict[str, dict] = {
        "Scout":    {"cost": 0, "card_type": CardType.SHIP},
        "Viper":    {"cost": 0, "card_type": CardType.SHIP},
        "Explorer": {"cost": 2, "card_type": CardType.SHIP},
    }

    # Add starter card defs if not already present from trade deck
    for starter in starter_names:
        if starter not in seen:
            seen.add(starter)
            meta = _STARTER_DEFS.get(starter, {"cost": 0, "card_type": CardType.SHIP})
            card_defs.append(CardDef(
                card_def_id=index,
                name=starter,
                cost=meta["cost"],
                card_type=meta["card_type"],
            ))
            index += 1

    registry = CardRegistry(card_defs, starter_names)

    # Validate: registry card_names must match the legacy ordering
    # so that existing checkpoints and encodings remain compatible
    _validate_name_ordering(cards, starter_names, registry)

    return registry


def _validate_name_ordering(
    cards: list[Card],
    starter_names: list[str],
    registry: CardRegistry,
) -> None:
    """Assert registry card_names matches the legacy DataConfig.get_card_names() output."""
    legacy_names: list[str] = list(dict.fromkeys(c.name for c in cards))
    for starter in starter_names:
        if starter not in legacy_names:
            legacy_names.append(starter)

    if registry.card_names != legacy_names:
        raise ValueError(
            f"Registry card_names ordering mismatch.\n"
            f"  Registry: {registry.card_names}\n"
            f"  Legacy:   {legacy_names}"
        )
