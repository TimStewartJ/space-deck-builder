"""Tests for CardRegistry: construction, lookup, and ordering consistency."""
import pytest

from src.cards.card import Card, CardType
from src.cards.effects import Effect, CardEffectType
from src.cards.factions import Faction
from src.cards.registry import CardDef, CardRegistry, build_registry
from src.config import DataConfig


def _make_card(name: str, index: int, cost: int = 1, card_type=CardType.SHIP,
               faction=None, defense=None, set_name: str = "Core Set") -> Card:
    return Card(
        name=name, index=index, cost=cost,
        effects=[Effect(CardEffectType.TRADE, 1)],
        card_type=card_type, defense=defense,
        faction=faction, set=set_name,
    )


class TestCardDef:
    def test_frozen(self):
        cd = CardDef(card_def_id=0, name="Scout", cost=0, card_type=CardType.SHIP)
        with pytest.raises(AttributeError):
            cd.name = "Viper"

    def test_fields(self):
        cd = CardDef(
            card_def_id=3, name="Blob Fighter", cost=1, card_type=CardType.SHIP,
            defense=None, faction=Faction.BLOB, set_name="Core Set",
        )
        assert cd.card_def_id == 3
        assert cd.name == "Blob Fighter"
        assert cd.cost == 1
        assert cd.card_type == CardType.SHIP
        assert cd.faction == Faction.BLOB
        assert cd.set_name == "Core Set"


class TestCardRegistry:
    def test_basic_construction(self):
        defs = [
            CardDef(0, "Blob Fighter", 1, CardType.SHIP, faction=Faction.BLOB),
            CardDef(1, "Trade Pod", 2, CardType.SHIP, faction=Faction.BLOB),
        ]
        reg = CardRegistry(defs, starter_names=["Scout", "Viper", "Explorer"])
        assert reg.num_cards == 5
        assert reg.card_names == ["Blob Fighter", "Trade Pod", "Scout", "Viper", "Explorer"]

    def test_card_index_map(self):
        defs = [
            CardDef(0, "Blob Fighter", 1, CardType.SHIP),
            CardDef(1, "Trade Pod", 2, CardType.SHIP),
        ]
        reg = CardRegistry(defs, starter_names=["Scout", "Viper", "Explorer"])
        assert reg.card_index_map["Blob Fighter"] == 0
        assert reg.card_index_map["Trade Pod"] == 1
        assert reg.card_index_map["Scout"] == 2
        assert reg.card_index_map["Viper"] == 3
        assert reg.card_index_map["Explorer"] == 4

    def test_get_by_id(self):
        cd = CardDef(0, "Blob Fighter", 1, CardType.SHIP)
        reg = CardRegistry([cd], starter_names=[])
        assert reg.get(0) is cd

    def test_get_by_name(self):
        cd = CardDef(0, "Blob Fighter", 1, CardType.SHIP)
        reg = CardRegistry([cd], starter_names=[])
        assert reg.get_by_name("Blob Fighter") is cd
        assert reg.get_by_name("Nonexistent") is None

    def test_len(self):
        defs = [CardDef(i, f"Card{i}", i, CardType.SHIP) for i in range(5)]
        reg = CardRegistry(defs, starter_names=["Scout"])
        assert len(reg) == 6

    def test_starters_not_duplicated(self):
        """If a starter name already exists in trade deck, don't add it twice."""
        defs = [
            CardDef(0, "Scout", 0, CardType.SHIP),
            CardDef(1, "Blob Fighter", 1, CardType.SHIP),
        ]
        reg = CardRegistry(defs, starter_names=["Scout", "Viper"])
        assert reg.card_names.count("Scout") == 1
        assert reg.num_cards == 3  # Scout, Blob Fighter, Viper


class TestBuildRegistry:
    def test_deduplicates_cards(self):
        """Multiple copies of the same card should produce one CardDef."""
        card = _make_card("Blob Fighter", 0)
        cards = [card, card.clone(), card.clone()]
        reg = build_registry(cards, ["Scout", "Viper", "Explorer"])
        blob_defs = [cd for cd in reg.card_defs if cd.name == "Blob Fighter"]
        assert len(blob_defs) == 1

    def test_preserves_order(self):
        """Cards should appear in the order they first appear in the list."""
        cards = [
            _make_card("Trade Pod", 0, cost=2),
            _make_card("Blob Fighter", 1, cost=1),
            _make_card("Trade Pod", 0, cost=2),  # duplicate
        ]
        reg = build_registry(cards, ["Scout", "Viper", "Explorer"])
        assert reg.card_names[:2] == ["Trade Pod", "Blob Fighter"]

    def test_starters_appended(self):
        """Starter cards should appear after trade deck cards."""
        cards = [_make_card("Blob Fighter", 0)]
        reg = build_registry(cards, ["Scout", "Viper", "Explorer"])
        assert reg.card_names == ["Blob Fighter", "Scout", "Viper", "Explorer"]

    def test_multi_faction_card(self):
        """Multi-faction cards should store faction as bitmask."""
        card = _make_card("Patrol Cutter", 0, faction=Faction.TRADE_FEDERATION | Faction.STAR_EMPIRE)
        reg = build_registry([card], ["Scout"])
        cd = reg.get_by_name("Patrol Cutter")
        assert cd.faction == Faction.TRADE_FEDERATION | Faction.STAR_EMPIRE

    def test_card_def_fields(self):
        card = _make_card("Space Station", 0, cost=4, card_type=CardType.OUTPOST,
                          defense=5, faction=Faction.STAR_EMPIRE)
        reg = build_registry([card], ["Scout"])
        cd = reg.get_by_name("Space Station")
        assert cd.cost == 4
        assert cd.card_type == CardType.OUTPOST
        assert cd.defense == 5
        assert cd.faction == Faction.STAR_EMPIRE


class TestRegistryMatchesLegacy:
    """Verify registry card_names matches the legacy DataConfig.get_card_names() output."""

    def test_matches_data_config(self):
        """Build registry and legacy card_names from same cards — must match exactly."""
        data_cfg = DataConfig()
        cards = data_cfg.load_cards()

        # Legacy path
        legacy_names = data_cfg.get_card_names(cards)

        # Registry path
        registry = data_cfg.build_registry(cards)

        assert registry.card_names == legacy_names
        assert registry.num_cards == len(legacy_names)

    def test_card_index_map_matches(self):
        """Registry card_index_map must produce same mapping as build_card_index_map."""
        from src.encoding.state_encoder import build_card_index_map

        data_cfg = DataConfig()
        cards = data_cfg.load_cards()
        legacy_names = data_cfg.get_card_names(cards)
        legacy_map = build_card_index_map(legacy_names)

        registry = data_cfg.build_registry(cards)

        assert registry.card_index_map == legacy_map

    def test_action_space_size_unchanged(self):
        """Action space size must be the same with registry card_names."""
        from src.encoding.action_encoder import get_action_space_size

        data_cfg = DataConfig()
        cards = data_cfg.load_cards()
        legacy_names = data_cfg.get_card_names(cards)
        registry = data_cfg.build_registry(cards)

        assert get_action_space_size(registry.card_names) == get_action_space_size(legacy_names)

    def test_state_size_unchanged(self):
        """State encoding size must be the same with registry card_names."""
        from src.encoding.state_encoder import get_state_size

        data_cfg = DataConfig()
        cards = data_cfg.load_cards()
        legacy_names = data_cfg.get_card_names(cards)
        registry = data_cfg.build_registry(cards)

        assert get_state_size(registry.card_names) == get_state_size(legacy_names)

    def test_starter_card_metadata_is_accurate(self):
        """Starter card CardDefs must have correct cost/type, not placeholders."""
        data_cfg = DataConfig()
        cards = data_cfg.load_cards()
        registry = data_cfg.build_registry(cards)

        scout = registry.get_by_name("Scout")
        assert scout is not None
        assert scout.cost == 0
        assert scout.card_type == CardType.SHIP

        viper = registry.get_by_name("Viper")
        assert viper is not None
        assert viper.cost == 0
        assert viper.card_type == CardType.SHIP

        explorer = registry.get_by_name("Explorer")
        assert explorer is not None
        assert explorer.cost == 2
        assert explorer.card_type == CardType.SHIP
