"""Tests for Card.clone() and Effect.clone() methods."""
import pytest
from src.cards.card import Card, CardType
from src.cards.effects import Effect, CardEffectType
from src.cards.factions import Faction


class TestEffectClone:
    def test_clone_returns_new_object(self):
        effect = Effect(CardEffectType.COMBAT, value=5)
        cloned = effect.clone()
        assert cloned is not effect

    def test_clone_preserves_config(self):
        effect = Effect(
            CardEffectType.SCRAP, value=1, text="Scrap a card",
            faction_requirement="Trade Federation", is_scrap_effect=True,
            is_ally_effect=True, faction_requirement_count=2,
            is_or_effect=True, is_mandatory=True,
            card_targets=["hand", "discard"],
        )
        cloned = effect.clone()
        assert cloned.effect_type == CardEffectType.SCRAP
        assert cloned.value == 1
        assert cloned.text == "Scrap a card"
        assert cloned.faction_requirement == "Trade Federation"
        assert cloned.is_scrap_effect is True
        assert cloned.is_ally_effect is True
        assert cloned.faction_requirement_count == 2
        assert cloned.is_or_effect is True
        assert cloned.is_mandatory is True
        assert cloned.card_targets == ["hand", "discard"]

    def test_clone_has_fresh_applied_state(self):
        effect = Effect(CardEffectType.COMBAT, value=3)
        effect.applied = True
        cloned = effect.clone()
        assert cloned.applied is False
        assert effect.applied is True

    def test_clone_applied_mutation_is_independent(self):
        effect = Effect(CardEffectType.TRADE, value=2)
        cloned = effect.clone()
        cloned.applied = True
        assert effect.applied is False

    def test_clone_child_effects_are_independent(self):
        child1 = Effect(CardEffectType.COMBAT, value=2)
        child2 = Effect(CardEffectType.TRADE, value=3)
        parent = Effect(CardEffectType.PARENT, is_or_effect=True, child_effects=[child1, child2])

        cloned = parent.clone()

        # Children are distinct objects
        assert cloned.child_effects is not parent.child_effects
        assert cloned.child_effects[0] is not child1
        assert cloned.child_effects[1] is not child2

        # Mutating cloned child doesn't affect original
        cloned.child_effects[0].applied = True
        assert child1.applied is False

    def test_clone_deeply_nested_children(self):
        grandchild = Effect(CardEffectType.DRAW, value=1)
        child = Effect(CardEffectType.PARENT, child_effects=[grandchild])
        parent = Effect(CardEffectType.PARENT, child_effects=[child])

        cloned = parent.clone()

        cloned.child_effects[0].child_effects[0].applied = True
        assert grandchild.applied is False

    def test_clone_none_child_effects(self):
        effect = Effect(CardEffectType.COMBAT, value=5)
        assert effect.child_effects is None
        cloned = effect.clone()
        assert cloned.child_effects is None

    def test_clone_none_card_targets(self):
        effect = Effect(CardEffectType.COMBAT, value=5)
        assert effect.card_targets is None
        cloned = effect.clone()
        assert cloned.card_targets is None

    def test_clone_card_targets_are_independent(self):
        effect = Effect(CardEffectType.SCRAP, card_targets=["hand", "discard"])
        cloned = effect.clone()
        cloned.card_targets.append("trade")
        assert len(effect.card_targets) == 2


class TestCardClone:
    def _make_card(self):
        effects = [
            Effect(CardEffectType.COMBAT, value=4),
            Effect(CardEffectType.PARENT, is_or_effect=True, child_effects=[
                Effect(CardEffectType.TRADE, value=2),
                Effect(CardEffectType.DRAW, value=1),
            ]),
        ]
        return Card("Viper", index=0, cost=0, effects=effects,
                     card_type=CardType.SHIP, defense=None, faction=Faction.BLOB, set="Core Set")

    def test_clone_returns_new_card(self):
        card = self._make_card()
        cloned = card.clone()
        assert cloned is not card

    def test_clone_preserves_identity_fields(self):
        card = self._make_card()
        cloned = card.clone()
        assert cloned.name == "Viper"
        assert cloned.index == 0
        assert cloned.cost == 0
        assert cloned.card_type == CardType.SHIP
        assert cloned.defense is None
        assert cloned.faction == Faction.BLOB
        assert cloned.set == "Core Set"

    def test_clone_effects_are_independent(self):
        card = self._make_card()
        cloned = card.clone()

        assert cloned.effects is not card.effects
        assert cloned.effects[0] is not card.effects[0]

        cloned.effects[0].applied = True
        assert card.effects[0].applied is False

    def test_clone_nested_effects_are_independent(self):
        card = self._make_card()
        cloned = card.clone()

        # Mutate a child effect on the clone
        cloned.effects[1].child_effects[0].applied = True
        assert card.effects[1].child_effects[0].applied is False

    def test_clone_bitmask_faction_is_independent(self):
        """Faction bitmasks are immutable ints — cloning preserves the value."""
        card = Card("Patrol Cutter", index=1, cost=2,
                     effects=[Effect(CardEffectType.COMBAT, value=2)],
                     faction=Faction.TRADE_FEDERATION | Faction.STAR_EMPIRE)
        cloned = card.clone()
        assert cloned.faction == Faction.TRADE_FEDERATION | Faction.STAR_EMPIRE
        assert card.faction == cloned.faction

    def test_clone_string_faction_unchanged(self):
        card = self._make_card()
        cloned = card.clone()
        assert cloned.faction == Faction.BLOB

    def test_clone_none_faction(self):
        card = Card("Scout", index=2, cost=0,
                     effects=[Effect(CardEffectType.TRADE, value=1)],
                     faction=None)
        cloned = card.clone()
        assert cloned.faction == Faction.NONE


class TestEffectReset:
    def test_reset_clears_applied(self):
        effect = Effect(CardEffectType.COMBAT, value=5)
        effect.applied = True
        effect.reset()
        assert effect.applied is False

    def test_reset_recurses_into_children(self):
        child1 = Effect(CardEffectType.COMBAT, value=2)
        child2 = Effect(CardEffectType.TRADE, value=3)
        parent = Effect(CardEffectType.PARENT, is_or_effect=True, child_effects=[child1, child2])

        child1.applied = True
        parent.reset()
        assert child1.applied is False
        assert child2.applied is False

    def test_reset_recurses_deeply(self):
        grandchild = Effect(CardEffectType.DRAW, value=1)
        child = Effect(CardEffectType.PARENT, child_effects=[grandchild])
        parent = Effect(CardEffectType.PARENT, child_effects=[child])

        grandchild.applied = True
        child.applied = True
        parent.reset()
        assert grandchild.applied is False
        assert child.applied is False

    def test_reset_with_no_children(self):
        effect = Effect(CardEffectType.COMBAT, value=5)
        effect.applied = True
        effect.reset()
        assert effect.applied is False

    def test_card_reset_effects_resets_entire_tree(self):
        child = Effect(CardEffectType.TRADE, value=2)
        parent_effect = Effect(CardEffectType.PARENT, child_effects=[child])
        simple_effect = Effect(CardEffectType.COMBAT, value=5)
        card = Card("Test", index=0, cost=0,
                     effects=[simple_effect, parent_effect])

        simple_effect.applied = True
        child.applied = True
        card.reset_effects()

        assert simple_effect.applied is False
        assert child.applied is False
