"""Tests for game engine rule correctness — Work Unit A: Base & Outpost System."""
import pytest
from src.cards.card import Card, CardType
from src.cards.effects import Effect, CardEffectType
from src.cards.factions import Faction
from src.engine.player import Player
from src.engine.actions import get_available_actions, ActionType, Action, CardSource
from src.ai.agent import Agent


def _make_game_with_players():
    """Create a minimal Game with two players for testing."""
    from src.engine.game import Game
    game = Game(cards=[], card_names=["Scout", "Viper", "Explorer", "TestBase", "TestOutpost"])
    p1 = game.add_player("P1", Agent("P1"))
    p2 = game.add_player("P2", Agent("P2"))
    game.stats.add_player("P1")
    game.stats.add_player("P2")
    game.is_running = True
    game.current_player = p1
    game.current_turn = 0
    return game, p1, p2


def _make_base(name="TestBase", defense=5, card_type="base", effects=None, faction=None):
    return Card(name=name, index=3, cost=3, effects=effects or [],
                card_type=card_type, defense=defense, faction=faction)


def _make_outpost(name="TestOutpost", defense=4, effects=None, faction=None):
    return _make_base(name=name, defense=defense, card_type="outpost",
                      effects=effects, faction=faction)


class TestA1OutpostPersistence:
    """A1: Outposts must enter the bases list when played."""

    def test_outpost_enters_bases_on_play(self):
        player = Player("P1", Agent("P1"))
        outpost = _make_outpost()
        player.hand.append(outpost)
        player.play_card(outpost)

        assert outpost in player.bases
        assert outpost in player.played_cards

    def test_base_still_enters_bases_on_play(self):
        player = Player("P1", Agent("P1"))
        base = _make_base()
        player.hand.append(base)
        player.play_card(base)

        assert base in player.bases

    def test_ship_does_not_enter_bases(self):
        player = Player("P1", Agent("P1"))
        ship = Card("Ship", 0, 1, [], "ship")
        player.hand.append(ship)
        player.play_card(ship)

        assert ship not in player.bases
        assert ship in player.played_cards

    def test_outpost_persists_across_turns(self):
        player = Player("P1", Agent("P1"))
        outpost = _make_outpost()
        player.hand.append(outpost)
        player.play_card(outpost)
        player.end_turn()

        assert outpost in player.bases
        assert outpost not in player.discard_pile


class TestA2BaseEffectsEachTurn:
    """A2: Bases/outposts must provide effects each turn they're in play."""

    def test_base_effects_available_after_play(self):
        game, p1, _ = _make_game_with_players()
        base = _make_base(effects=[Effect(CardEffectType.TRADE, 3)])
        p1.hand.append(base)
        p1.play_card(base)

        actions = get_available_actions(game, p1)
        effect_actions = [a for a in actions if a.type == ActionType.APPLY_EFFECT]
        assert len(effect_actions) == 1

    def test_base_effects_available_next_turn(self):
        """After end_turn, base effects should reset and be available again."""
        game, p1, _ = _make_game_with_players()
        base = _make_base(effects=[Effect(CardEffectType.TRADE, 3)])
        p1.hand.append(base)
        p1.play_card(base)

        # Use the effect
        for effect in base.effects:
            effect.applied = True

        # End turn resets effects
        p1.end_turn()

        # Base effects should be available again
        actions = get_available_actions(game, p1)
        effect_actions = [a for a in actions if a.type == ActionType.APPLY_EFFECT]
        assert len(effect_actions) == 1

    def test_base_effects_reset_on_end_turn(self):
        player = Player("P1", Agent("P1"))
        base = _make_base(effects=[Effect(CardEffectType.COMBAT, 2)])
        player.hand.append(base)
        player.play_card(base)

        # Apply the effect
        base.effects[0].applied = True
        assert base.effects[0].applied is True

        player.end_turn()
        assert base.effects[0].applied is False

    def test_no_duplicate_effects_from_base_in_played_cards(self):
        """A base in both played_cards and bases should only offer effects once."""
        game, p1, _ = _make_game_with_players()
        base = _make_base(effects=[Effect(CardEffectType.TRADE, 2)])
        p1.hand.append(base)
        p1.play_card(base)

        # Base is in both played_cards and bases on the turn it's played
        assert base in p1.played_cards
        assert base in p1.bases

        actions = get_available_actions(game, p1)
        effect_actions = [a for a in actions if a.type == ActionType.APPLY_EFFECT]
        assert len(effect_actions) == 1  # not 2


class TestA3MultiFactionAllyCount:
    """A3: Faction ally counting must handle list factions and include bases."""

    def test_single_faction_string(self):
        player = Player("P1", Agent("P1"))
        card = Card("Ship", 0, 1, [], "ship", faction="Blob")
        player.played_cards.append(card)

        assert player.get_faction_ally_count("Blob") == 1
        assert player.get_faction_ally_count("Star Empire") == 0

    def test_multi_faction_list(self):
        player = Player("P1", Agent("P1"))
        card = Card("DualShip", 0, 1, [], "ship", faction=["Blob", "Star Empire"])
        player.played_cards.append(card)

        assert player.get_faction_ally_count("Blob") == 1
        assert player.get_faction_ally_count("Star Empire") == 1
        assert player.get_faction_ally_count("Trade Federation") == 0

    def test_bases_count_for_allies(self):
        player = Player("P1", Agent("P1"))
        base = _make_base(faction="Machine Cult")
        player.hand.append(base)
        player.play_card(base)
        # Simulate end of turn — base leaves played_cards but stays in bases
        player.end_turn()

        assert player.get_faction_ally_count("Machine Cult") == 1

    def test_case_insensitive(self):
        player = Player("P1", Agent("P1"))
        card = Card("Ship", 0, 1, [], "ship", faction="blob")
        player.played_cards.append(card)

        assert player.get_faction_ally_count("Blob") == 1
        assert player.get_faction_ally_count("BLOB") == 1

    def test_no_faction_not_counted(self):
        player = Player("P1", Agent("P1"))
        card = Card("Ship", 0, 1, [], "ship", faction=None)
        player.played_cards.append(card)

        assert player.get_faction_ally_count("Blob") == 0

    def test_base_and_ship_same_faction_counts_both(self):
        """A base + a ship of the same faction should count as 2."""
        player = Player("P1", Agent("P1"))
        base = _make_base(faction="Blob")
        ship = Card("Ship", 0, 1, [], "ship", faction="Blob")
        player.hand.append(base)
        player.play_card(base)
        player.played_cards.append(ship)

        assert player.get_faction_ally_count("Blob") == 2


class TestE1AllyFactions:
    """E1: Extensible ally_factions system — wildcard, specific list, fallback."""

    def test_wildcard_counts_for_any_faction(self):
        """A card with ally_factions=ALL counts as ally for every faction."""
        player = Player("P1", Agent("P1"))
        mech_world = _make_outpost(name="Mech World", faction="Machine Cult")
        mech_world.ally_factions = Faction.ALL
        player.hand.append(mech_world)
        player.play_card(mech_world)

        assert player.get_faction_ally_count("Blob") == 1
        assert player.get_faction_ally_count("Star Empire") == 1
        assert player.get_faction_ally_count("Trade Federation") == 1
        assert player.get_faction_ally_count("Machine Cult") == 1

    def test_specific_ally_factions_list(self):
        """A card with ally_factions=["Blob","SE"] counts only for those."""
        player = Player("P1", Agent("P1"))
        card = Card("DualAlly", 0, 3, [], "ship",
                     faction="Blob", ally_factions=["Blob", "Star Empire"])
        player.played_cards.append(card)

        assert player.get_faction_ally_count("Blob") == 1
        assert player.get_faction_ally_count("Star Empire") == 1
        assert player.get_faction_ally_count("Trade Federation") == 0
        assert player.get_faction_ally_count("Machine Cult") == 0

    def test_none_ally_factions_falls_back_to_faction(self):
        """Default behavior: ally_factions=Faction.NONE uses card.faction."""
        player = Player("P1", Agent("P1"))
        card = Card("Ship", 0, 1, [], "ship", faction="Blob")
        assert card.ally_factions == Faction.NONE
        player.played_cards.append(card)

        assert player.get_faction_ally_count("Blob") == 1
        assert player.get_faction_ally_count("Star Empire") == 0

    def test_ally_factions_case_insensitive(self):
        player = Player("P1", Agent("P1"))
        card = Card("Ship", 0, 1, [], "ship",
                     ally_factions=["blob", "STAR EMPIRE"])
        player.played_cards.append(card)

        assert player.get_faction_ally_count("Blob") == 1
        assert player.get_faction_ally_count("Star Empire") == 1

    def test_clone_preserves_ally_factions(self):
        card = Card("Mech World", 0, 5, [], "outpost",
                     faction="Machine Cult", ally_factions=["*"])
        cloned = card.clone()

        assert cloned.ally_factions == Faction.ALL

    def test_clone_none_ally_factions(self):
        card = Card("Ship", 0, 1, [], "ship")
        cloned = card.clone()
        assert cloned.ally_factions == Faction.NONE

    def test_mech_world_loaded_from_csv(self):
        """Integration: Mech World from cards.csv gets ally_factions=Faction.ALL."""
        from src.config import DataConfig
        cfg = DataConfig()
        cards = cfg.load_cards()
        mech_worlds = [c for c in cards if c.name == "Mech World"]
        assert len(mech_worlds) == 1
        assert mech_worlds[0].ally_factions == Faction.ALL

    def test_wildcard_enables_ally_ability(self):
        """Mech World should satisfy any faction's ally requirement."""
        game, p1, _ = _make_game_with_players()
        mech_world = _make_outpost(name="Mech World", faction="Machine Cult")
        mech_world.ally_factions = Faction.ALL
        p1.hand.append(mech_world)
        p1.play_card(mech_world)

        # Play a Blob ship with a Blob ally effect
        blob_ship = Card("BlobShip", 0, 1, [
            Effect(CardEffectType.COMBAT, 3),
            Effect(CardEffectType.COMBAT, 2, faction_requirement="Blob"),
        ], "ship", faction="Blob")
        p1.hand.append(blob_ship)
        p1.play_card(blob_ship)

        # Mech World counts as Blob ally → ally count = 2 (Mech World + BlobShip)
        assert p1.get_faction_ally_count("Blob") == 2
        # The ally effect should now be available as an action
        actions = get_available_actions(game, p1)
        ally_effects = [a for a in actions
                        if a.type == ActionType.APPLY_EFFECT
                        and a.card_effect
                        and a.card_effect.faction_requirement == "Blob"]
        assert len(ally_effects) == 1


class TestA4ScrapRemovesFromBases:
    """A4: Scrapping a base/outpost from play must remove it from bases list."""

    def test_scrap_removes_base_from_bases_list(self):
        game, p1, _ = _make_game_with_players()
        base = _make_base(
            effects=[Effect(CardEffectType.COMBAT, 5, is_scrap_effect=True)]
        )
        p1.hand.append(base)
        p1.play_card(base)

        assert base in p1.bases
        assert base in p1.played_cards

        # Apply scrap effect
        base.effects[0].apply(game, p1, base)

        assert base not in p1.bases
        assert base not in p1.played_cards

    def test_scrap_removes_outpost_from_bases_list(self):
        game, p1, _ = _make_game_with_players()
        outpost = _make_outpost(
            effects=[Effect(CardEffectType.COMBAT, 5, is_scrap_effect=True)]
        )
        p1.hand.append(outpost)
        p1.play_card(outpost)

        assert outpost in p1.bases

        outpost.effects[0].apply(game, p1, outpost)

        assert outpost not in p1.bases
        assert outpost not in p1.played_cards

    def test_scrap_ship_does_not_affect_bases(self):
        game, p1, _ = _make_game_with_players()
        ship = Card("Ship", 0, 1,
                     [Effect(CardEffectType.COMBAT, 2, is_scrap_effect=True)], "ship")
        p1.played_cards.append(ship)

        ship.effects[0].apply(game, p1, ship)

        assert ship not in p1.played_cards
        assert len(p1.bases) == 0

    def test_scrap_via_execute_action_removes_base(self):
        """End-to-end: APPLY_EFFECT through execute_action passes card ref."""
        game, p1, _ = _make_game_with_players()
        base = _make_base(
            effects=[Effect(CardEffectType.COMBAT, 5, is_scrap_effect=True)]
        )
        p1.hand.append(base)
        p1.play_card(base)

        assert base in p1.bases

        # Get the APPLY_EFFECT action from available actions
        actions = get_available_actions(game, p1)
        scrap_actions = [a for a in actions
                         if a.type == ActionType.APPLY_EFFECT
                         and a.card_effect and a.card_effect.is_scrap_effect]
        assert len(scrap_actions) == 1

        # Execute through the game engine
        game.execute_action(scrap_actions[0])

        assert base not in p1.bases
        assert base not in p1.played_cards
        assert p1.combat == 5


# ── Work Unit B: Effect Application ──────────────────────────────────────────


class TestB1ScrapEffectsNoAutoFire:
    """B1: Scrap effects must not auto-fire when a card is played."""

    def test_scrap_combat_not_auto_applied(self):
        """A ship with a scrap combat effect should NOT add combat on play."""
        game, p1, _ = _make_game_with_players()
        ship = Card("Explorer", 2, 2, [
            Effect(CardEffectType.TRADE, 2),
            Effect(CardEffectType.COMBAT, 2, is_scrap_effect=True),
        ], "ship")
        p1.hand.append(ship)

        # Play the card through execute_action
        play_action = Action(type=ActionType.PLAY_CARD, card=ship, card_id=ship.name)
        game.execute_action(play_action)

        # Trade effect should auto-apply, scrap combat should NOT
        assert p1.trade == 2
        assert p1.combat == 0

    def test_scrap_effect_available_as_action(self):
        """Scrap effects should appear as APPLY_EFFECT actions."""
        game, p1, _ = _make_game_with_players()
        ship = Card("Explorer", 2, 2, [
            Effect(CardEffectType.TRADE, 2),
            Effect(CardEffectType.COMBAT, 2, is_scrap_effect=True),
        ], "ship")
        p1.hand.append(ship)
        p1.play_card(ship)
        # Mark trade effect as applied (already auto-fired)
        ship.effects[0].applied = True

        actions = get_available_actions(game, p1)
        scrap_actions = [a for a in actions
                         if a.type == ActionType.APPLY_EFFECT
                         and a.card_effect and a.card_effect.is_scrap_effect]
        assert len(scrap_actions) == 1

    def test_ally_effect_not_auto_applied(self):
        """Effects with faction requirements should not auto-fire."""
        game, p1, _ = _make_game_with_players()
        ship = Card("Ship", 0, 1, [
            Effect(CardEffectType.COMBAT, 3),
            Effect(CardEffectType.COMBAT, 2, faction_requirement="Blob"),
        ], "ship", faction="Blob")
        p1.hand.append(ship)

        play_action = Action(type=ActionType.PLAY_CARD, card=ship, card_id=ship.name)
        game.execute_action(play_action)

        # Base combat auto-applies, ally bonus should NOT (no ally in play yet)
        assert p1.combat == 3

    def test_or_effect_not_auto_applied(self):
        """OR effects must not auto-fire — both branches would resolve."""
        game, p1, _ = _make_game_with_players()
        ship = Card("Patrol Mech", 0, 4, [
            Effect(CardEffectType.PARENT, 0, is_or_effect=True, child_effects=[
                Effect(CardEffectType.TRADE, 3),
                Effect(CardEffectType.COMBAT, 5),
            ]),
        ], "ship", faction="Machine Cult")
        p1.hand.append(ship)

        play_action = Action(type=ActionType.PLAY_CARD, card=ship, card_id=ship.name)
        game.execute_action(play_action)

        # Neither branch should have auto-fired
        assert p1.trade == 0
        assert p1.combat == 0

    def test_first_slot_ally_effect_not_auto_applied(self):
        """An ally effect in slot 0 should not auto-fire even when ally is present."""
        game, p1, _ = _make_game_with_players()
        # Put an ally in play first
        ally = Card("Ally", 0, 1, [Effect(CardEffectType.COMBAT, 1)],
                     "ship", faction="Star Empire")
        p1.played_cards.append(ally)

        ship = Card("Ship", 1, 1, [
            Effect(CardEffectType.COMBAT, 2, faction_requirement="Star Empire"),
        ], "ship", faction="Star Empire")
        p1.hand.append(ship)

        play_action = Action(type=ActionType.PLAY_CARD, card=ship, card_id=ship.name)
        game.execute_action(play_action)

        # Ally effect should NOT auto-fire — it should be an APPLY_EFFECT choice
        assert p1.combat == 1  # only the ally ship's own effect


class TestB3IterationSafety:
    """B3: Iterating played_cards during auto-apply must not skip cards."""

    def test_multiple_ships_all_effects_applied(self):
        game, p1, _ = _make_game_with_players()
        ship1 = Card("Ship1", 0, 1, [Effect(CardEffectType.COMBAT, 2)], "ship")
        ship2 = Card("Ship2", 1, 1, [Effect(CardEffectType.TRADE, 3)], "ship")
        p1.played_cards.extend([ship1, ship2])
        p1.hand.append(Card("Ship3", 2, 1, [Effect(CardEffectType.COMBAT, 1)], "ship"))

        play_action = Action(type=ActionType.PLAY_CARD,
                             card=p1.hand[0], card_id=p1.hand[0].name)
        game.execute_action(play_action)

        # All three ships' effects should have fired
        assert p1.combat == 3  # 2 + 1
        assert p1.trade == 3


class TestB4TradeRowScrapRefresh:
    """B4: Trade row should refill correctly after scrapping from it."""

    def test_refill_after_standalone_scrap(self):
        """Scrapping from trade row outside a pending set should refill."""
        game, p1, _ = _make_game_with_players()
        # Put some cards in trade deck and trade row
        filler = Card("Filler", 0, 1, [], "ship")
        game.trade_deck.append(filler)
        target = Card("Target", 1, 3, [], "ship")
        game.trade_row.append(target)
        initial_row_size = len(game.trade_row)

        scrap_action = Action(type=ActionType.SCRAP_CARD,
                              card_id=1, card=target, card_source=CardSource.TRADE)
        game.execute_action(scrap_action)

        # Row should have refilled with the filler card
        assert target not in game.trade_row
        assert filler in game.trade_row
        assert len(game.trade_row) == initial_row_size


# ── Work Unit C: Cross-Player Effects ────────────────────────────────────────


class TestC1ForcedDiscard:
    """C1: Forced discard defers to opponent's turn start (opponent chooses)."""

    def test_forced_discard_defers_to_opponent(self):
        """TARGET_DISCARD should increment opponent's pending_start_of_turn_discards."""
        game, p1, p2 = _make_game_with_players()
        opp_card = Card("OppCard", 0, 1, [], "ship")
        p2.hand.append(opp_card)

        effect = Effect(CardEffectType.TARGET_DISCARD, 1,
                        card_targets=["opponent"])
        effect.apply(game, p1, None)

        # No immediate pending actions on either player
        assert len(p1.pending_action_sets) == 0
        assert len(p2.pending_action_sets) == 0
        # Discard debt stored on opponent
        assert p2.pending_start_of_turn_discards == 1

    def test_forced_discard_materializes_at_turn_start(self):
        """Deferred discards become pending actions when it's the opponent's turn."""
        game, p1, p2 = _make_game_with_players()
        opp_card = Card("OppCard", 0, 1, [], "ship")
        p2.hand.append(opp_card)

        # P1 triggers forced discard on P2
        effect = Effect(CardEffectType.TARGET_DISCARD, 1,
                        card_targets=["opponent"])
        effect.apply(game, p1, None)

        # Simulate turn end → switch to P2
        game._materialize_start_of_turn_discards()  # no-op, P1 has no debt
        game.current_player = p2
        game._materialize_start_of_turn_discards()

        # Now P2 has pending discard actions from their own hand
        assert len(p2.pending_action_sets) == 1
        pending = p2.get_current_pending_set()
        assert pending.mandatory is True
        assert pending.decisions_left == 1
        assert p2.pending_start_of_turn_discards == 0

    def test_forced_discard_opponent_chooses_from_own_hand(self):
        """Opponent picks which card to discard (no hidden-info leak)."""
        game, p1, p2 = _make_game_with_players()
        my_card = Card("MyCard", 0, 1, [], "ship")
        p1.hand.append(my_card)
        opp_card = Card("OppCard", 1, 1, [], "ship")
        p2.hand.append(opp_card)

        effect = Effect(CardEffectType.TARGET_DISCARD, 1,
                        card_targets=["opponent"])
        effect.apply(game, p1, None)

        # Switch to P2's turn and materialize
        game.current_player = p2
        game._materialize_start_of_turn_discards()

        # Pending actions reference P2's hand cards, using SELF source
        pending = p2.get_current_pending_set()
        assert len(pending.actions) == 1
        assert pending.actions[0].card is opp_card
        assert pending.actions[0].card_source == CardSource.SELF

        # Execute discard
        game.execute_action(pending.actions[0])
        assert opp_card not in p2.hand
        assert opp_card in p2.discard_pile
        assert my_card in p1.hand  # P1's hand untouched

    def test_forced_discard_clamped_to_hand_size(self):
        """Discard debt > hand size doesn't soft-lock."""
        game, p1, p2 = _make_game_with_players()
        sole_card = Card("OnlyCard", 0, 1, [], "ship")
        p2.hand.append(sole_card)

        # Two discard effects but only 1 card in hand
        p2.pending_start_of_turn_discards = 2
        game.current_player = p2
        game._materialize_start_of_turn_discards()

        pending = p2.get_current_pending_set()
        assert pending.decisions_left == 1  # clamped to hand size
        assert p2.pending_start_of_turn_discards == 0

    def test_forced_discard_stacks(self):
        """Multiple discard effects in one turn accumulate."""
        game, p1, p2 = _make_game_with_players()
        for i in range(3):
            p2.hand.append(Card(f"Card{i}", i, 1, [], "ship"))

        effect = Effect(CardEffectType.TARGET_DISCARD, 1,
                        card_targets=["opponent"])
        effect.applied = False
        effect.apply(game, p1, None)
        effect.applied = False
        effect.apply(game, p1, None)

        assert p2.pending_start_of_turn_discards == 2


class TestC2DestroyBase:
    """C2: Destroy-base effects can target any base and are chosen by current player."""

    def test_destroy_base_pending_on_current_player(self):
        """DESTROY_BASE pending actions should go to the current player."""
        game, p1, p2 = _make_game_with_players()
        base = _make_base(name="EnemyBase")
        p2.bases.append(base)

        effect = Effect(CardEffectType.DESTROY_BASE, 1)
        effect.apply(game, p1, None)

        assert len(p1.pending_action_sets) == 1
        assert len(p2.pending_action_sets) == 0

    def test_destroy_can_target_any_base_not_just_outposts(self):
        """Destroy effects bypass outpost priority — can target regular bases."""
        game, p1, p2 = _make_game_with_players()
        outpost = _make_outpost(name="EnemyOutpost")
        base = _make_base(name="EnemyBase")
        p2.bases.extend([outpost, base])

        effect = Effect(CardEffectType.DESTROY_BASE, 1)
        effect.apply(game, p1, None)

        pending = p1.get_current_pending_set()
        target_cards = [a.card for a in pending.actions]
        # Both should be targetable
        assert outpost in target_cards
        assert base in target_cards

    def test_destroy_base_removes_from_opponent_bases(self):
        """Executing DESTROY_BASE should remove the base from opponent."""
        game, p1, p2 = _make_game_with_players()
        base = _make_base(name="EnemyBase")
        p2.bases.append(base)

        effect = Effect(CardEffectType.DESTROY_BASE, 1)
        effect.apply(game, p1, None)

        pending = p1.get_current_pending_set()
        game.execute_action(pending.actions[0])

        assert base not in p2.bases
        assert base in p2.discard_pile


# ── Work Unit D: COMPLEX Effects ─────────────────────────────────────────────


class TestD1EmbassyYacht:
    """D1: Embassy Yacht draws 2 cards when player has 2+ bases in play."""

    def test_draws_two_with_two_bases(self):
        game, p1, _ = _make_game_with_players()
        # Put two bases in play
        base1 = _make_base(name="Base1", effects=[])
        base2 = _make_base(name="Base2", effects=[])
        p1.bases.extend([base1, base2])

        # Create Embassy Yacht's CONDITIONAL_DRAW effect
        yacht_effect = Effect(CardEffectType.CONDITIONAL_DRAW, 2, text="bases_ge_2")
        # Give the player cards to draw
        p1.deck.extend([Card("D1", 0, 0, [], "ship"), Card("D2", 1, 0, [], "ship")])
        initial_hand = len(p1.hand)

        yacht_effect.apply(game, p1, None)

        assert len(p1.hand) == initial_hand + 2

    def test_no_draw_with_one_base(self):
        game, p1, _ = _make_game_with_players()
        base1 = _make_base(name="Base1", effects=[])
        p1.bases.append(base1)
        p1.deck.extend([Card("D1", 0, 0, [], "ship"), Card("D2", 1, 0, [], "ship")])
        initial_hand = len(p1.hand)

        yacht_effect = Effect(CardEffectType.CONDITIONAL_DRAW, 2, text="bases_ge_2")
        yacht_effect.apply(game, p1, None)

        assert len(p1.hand) == initial_hand

    def test_no_draw_with_zero_bases(self):
        game, p1, _ = _make_game_with_players()
        p1.deck.append(Card("D1", 0, 0, [], "ship"))
        initial_hand = len(p1.hand)

        yacht_effect = Effect(CardEffectType.CONDITIONAL_DRAW, 2, text="bases_ge_2")
        yacht_effect.apply(game, p1, None)

        assert len(p1.hand) == initial_hand

    def test_trade_and_authority_always_apply(self):
        """Embassy Yacht's resource effects fire regardless of base count."""
        game, p1, _ = _make_game_with_players()
        yacht = Card("Embassy Yacht", 0, 3, [
            Effect(CardEffectType.PARENT, 0, child_effects=[
                Effect(CardEffectType.PARENT, 0, child_effects=[
                    Effect(CardEffectType.TRADE, 2),
                    Effect(CardEffectType.HEAL, 3),
                ]),
                Effect(CardEffectType.CONDITIONAL_DRAW, 2, text="bases_ge_2"),
            ]),
        ], "ship", faction="Trade Federation")
        p1.hand.append(yacht)

        play_action = Action(type=ActionType.PLAY_CARD, card=yacht, card_id=yacht.index)
        game.execute_action(play_action)

        # Resources always apply, no bases → no draw
        assert p1.trade == 2
        assert p1.health == 50 + 3


class TestD2BlobWorldFactionSafety:
    """D2: Blob World's faction counting must handle list factions."""

    def test_list_faction_no_crash(self):
        """Multi-faction cards in played_cards shouldn't crash Blob World."""
        game, p1, _ = _make_game_with_players()
        multi = Card("DualShip", 0, 1, [], "ship", faction=["Blob", "Star Empire"])
        p1.played_cards.append(multi)
        p1.deck.append(Card("D1", 1, 0, [], "ship"))

        blob_world_effect = Effect(CardEffectType.DRAW_PER_FACTION, 0, faction_target="Blob")
        blob_world_effect.apply(game, p1, None)

        # Multi-faction card with "Blob" should count
        assert len(p1.hand) == 1

    def test_string_faction_still_works(self):
        """Single string factions should continue to work."""
        game, p1, _ = _make_game_with_players()
        blob1 = Card("BlobShip", 0, 1, [], "ship", faction="Blob")
        blob2 = Card("BlobShip2", 1, 1, [], "ship", faction="Blob")
        p1.played_cards.extend([blob1, blob2])
        p1.deck.extend([Card("D1", 2, 0, [], "ship"), Card("D2", 3, 0, [], "ship")])

        blob_world_effect = Effect(CardEffectType.DRAW_PER_FACTION, 0, faction_target="Blob")
        blob_world_effect.apply(game, p1, None)

        assert len(p1.hand) == 2

    def test_non_matching_faction_not_counted(self):
        """Non-Blob factions should not trigger draws."""
        game, p1, _ = _make_game_with_players()
        se = Card("SEShip", 0, 1, [], "ship", faction="Star Empire")
        p1.played_cards.append(se)
        p1.deck.append(Card("D1", 1, 0, [], "ship"))
        initial_hand = len(p1.hand)

        blob_world_effect = Effect(CardEffectType.DRAW_PER_FACTION, 0, faction_target="Blob")
        blob_world_effect.apply(game, p1, None)

        assert len(p1.hand) == initial_hand


class TestD3RecyclingStation:
    """D3: Recycling Station discard-then-draw via pending action completion."""

    def _make_recycling_station_effect(self):
        return Effect(CardEffectType.DISCARD_DRAW, 2,
                      text="discard up to two cards, then draw that many cards")

    def test_discard_two_draw_two(self):
        game, p1, _ = _make_game_with_players()
        c1 = Card("C1", 0, 0, [], "ship")
        c2 = Card("C2", 1, 0, [], "ship")
        p1.hand.extend([c1, c2])
        # Cards to draw after discard
        p1.deck.extend([Card("D1", 2, 0, [], "ship"), Card("D2", 3, 0, [], "ship")])

        effect = self._make_recycling_station_effect()
        effect.apply(game, p1, None)

        # Should have pending discard actions
        pending = p1.get_current_pending_set()
        assert pending is not None
        assert pending.on_complete_draw is True

        # Discard first card
        discard1 = [a for a in pending.actions if a.card is c1][0]
        game.execute_action(discard1)

        # Discard second card
        pending = p1.get_current_pending_set()
        discard2 = [a for a in pending.actions if a.card is c2][0]
        game.execute_action(discard2)

        # Pending set should be done, 2 cards drawn
        assert p1.get_current_pending_set() is None
        assert c1 not in p1.hand
        assert c2 not in p1.hand
        # Drew 2 cards from deck
        assert any(c.name == "D1" for c in p1.hand)
        assert any(c.name == "D2" for c in p1.hand)

    def test_discard_one_then_skip_draw_one(self):
        game, p1, _ = _make_game_with_players()
        c1 = Card("C1", 0, 0, [], "ship")
        c2 = Card("C2", 1, 0, [], "ship")
        p1.hand.extend([c1, c2])
        p1.deck.extend([Card("D1", 2, 0, [], "ship"), Card("D2", 3, 0, [], "ship")])

        effect = self._make_recycling_station_effect()
        effect.apply(game, p1, None)

        # Discard one card
        pending = p1.get_current_pending_set()
        discard1 = [a for a in pending.actions if a.card is c1][0]
        game.execute_action(discard1)

        # Skip the rest
        skip = Action(type=ActionType.SKIP_DECISION)
        game.execute_action(skip)

        assert p1.get_current_pending_set() is None
        # Only 1 card drawn (c1 discarded, c2 still in hand, 1 drawn from deck)
        assert c1 not in p1.hand
        assert c2 in p1.hand
        assert len(p1.hand) == 2  # c2 + 1 drawn

    def test_skip_immediately_draw_zero(self):
        game, p1, _ = _make_game_with_players()
        c1 = Card("C1", 0, 0, [], "ship")
        p1.hand.append(c1)
        p1.deck.append(Card("D1", 1, 0, [], "ship"))
        initial_hand_size = len(p1.hand)

        effect = self._make_recycling_station_effect()
        effect.apply(game, p1, None)

        # Skip without discarding
        skip = Action(type=ActionType.SKIP_DECISION)
        game.execute_action(skip)

        assert p1.get_current_pending_set() is None
        # No cards drawn, hand unchanged (c1 still there)
        assert len(p1.hand) == initial_hand_size

    def test_or_branch_trade_no_discard(self):
        """Choosing the trade branch of Recycling Station should not trigger discard/draw."""
        game, p1, _ = _make_game_with_players()
        # Recycling Station as an OR: {Gain 1 Trade} OR discard-then-draw
        trade_effect = Effect(CardEffectType.TRADE, 1)
        trade_effect.apply(game, p1, None)

        assert p1.trade == 1
        assert p1.get_current_pending_set() is None


class TestD4BrainWorld:
    """D4: Brain World scrap-then-draw via pending action completion."""

    def _make_brain_world_effect(self):
        return Effect(CardEffectType.SCRAP_FROM_HAND_DISCARD, 2,
                      text="Scrap up to two cards from your hand and/or discard pile")

    def test_scrap_two_draw_two(self):
        game, p1, _ = _make_game_with_players()
        c1 = Card("C1", 0, 0, [], "ship")
        c2 = Card("C2", 1, 0, [], "ship")
        p1.hand.append(c1)
        p1.discard_pile.append(c2)
        p1.deck.extend([Card("D1", 2, 0, [], "ship"), Card("D2", 3, 0, [], "ship")])

        effect = self._make_brain_world_effect()
        effect.apply(game, p1, None)

        pending = p1.get_current_pending_set()
        assert pending is not None
        assert pending.on_complete_draw is True

        # Scrap from hand
        scrap1 = [a for a in pending.actions if a.card is c1][0]
        game.execute_action(scrap1)

        # Scrap from discard
        pending = p1.get_current_pending_set()
        scrap2 = [a for a in pending.actions if a.card is c2][0]
        game.execute_action(scrap2)

        assert p1.get_current_pending_set() is None
        assert c1 not in p1.hand
        assert c2 not in p1.discard_pile
        assert any(c.name == "D1" for c in p1.hand)
        assert any(c.name == "D2" for c in p1.hand)

    def test_scrap_one_then_skip_draw_one(self):
        game, p1, _ = _make_game_with_players()
        c1 = Card("C1", 0, 0, [], "ship")
        p1.hand.append(c1)
        p1.deck.extend([Card("D1", 1, 0, [], "ship"), Card("D2", 2, 0, [], "ship")])

        effect = self._make_brain_world_effect()
        effect.apply(game, p1, None)

        # Scrap one
        pending = p1.get_current_pending_set()
        scrap1 = [a for a in pending.actions if a.card is c1][0]
        game.execute_action(scrap1)

        # Skip remaining
        skip = Action(type=ActionType.SKIP_DECISION)
        game.execute_action(skip)

        assert p1.get_current_pending_set() is None
        # c1 scrapped, 1 card drawn from deck
        assert c1 not in p1.hand
        assert len(p1.hand) == 1

    def test_skip_immediately_draw_zero(self):
        game, p1, _ = _make_game_with_players()
        c1 = Card("C1", 0, 0, [], "ship")
        p1.hand.append(c1)
        p1.deck.append(Card("D1", 1, 0, [], "ship"))
        initial_hand_size = len(p1.hand)

        effect = self._make_brain_world_effect()
        effect.apply(game, p1, None)

        skip = Action(type=ActionType.SKIP_DECISION)
        game.execute_action(skip)

        assert p1.get_current_pending_set() is None
        assert len(p1.hand) == initial_hand_size  # no draw

    def test_mixed_sources_hand_and_discard(self):
        """Scrapping one from hand and one from discard should both count."""
        game, p1, _ = _make_game_with_players()
        h1 = Card("Hand1", 0, 0, [], "ship")
        d1 = Card("Disc1", 1, 0, [], "ship")
        p1.hand.append(h1)
        p1.discard_pile.append(d1)
        p1.deck.extend([Card("D1", 2, 0, [], "ship"), Card("D2", 3, 0, [], "ship")])

        effect = self._make_brain_world_effect()
        effect.apply(game, p1, None)

        pending = p1.get_current_pending_set()
        # Scrap from hand
        scrap_hand = [a for a in pending.actions if a.card_source == CardSource.HAND and a.card is h1][0]
        game.execute_action(scrap_hand)

        # Scrap from discard
        pending = p1.get_current_pending_set()
        scrap_disc = [a for a in pending.actions if a.card_source == CardSource.DISCARD and a.card is d1][0]
        game.execute_action(scrap_disc)

        assert p1.get_current_pending_set() is None
        assert h1 not in p1.hand
        assert d1 not in p1.discard_pile
        assert len([c for c in p1.hand if c.name in ("D1", "D2")]) == 2

    def test_stale_action_does_not_inflate_draw_count(self):
        """Replaying a consumed action must not increment resolved_count."""
        game, p1, _ = _make_game_with_players()
        c1 = Card("C1", 0, 0, [], "ship")
        c2 = Card("C2", 1, 0, [], "ship")
        p1.hand.extend([c1, c2])
        p1.deck.extend([Card("D1", 2, 0, [], "ship"), Card("D2", 3, 0, [], "ship")])

        effect = self._make_brain_world_effect()
        effect.apply(game, p1, None)

        pending = p1.get_current_pending_set()
        scrap1 = [a for a in pending.actions if a.card is c1][0]
        game.execute_action(scrap1)

        # Replay the same stale action — should not match again
        game.execute_action(scrap1)

        # Skip remaining
        skip = Action(type=ActionType.SKIP_DECISION)
        game.execute_action(skip)

        # Only 1 scrap counted → 1 card drawn. Hand = c2 + 1 drawn = 2
        assert p1.get_current_pending_set() is None
        assert c1 not in p1.hand
        assert c2 in p1.hand
        assert len(p1.hand) == 2  # c2 + 1 drawn (not 3 from inflated count)


class TestE2DestroyBaseMandatory:
    """E2: 'Destroy target base' is mandatory; 'You may destroy' is optional."""

    def test_destroy_target_base_is_mandatory(self):
        """'Destroy target base' should create a mandatory pending set."""
        from src.cards.effects_parser import parse_effect_text
        effect = parse_effect_text("Destroy target base")
        assert effect.is_mandatory is True

    def test_you_may_destroy_is_optional(self):
        """'You may destroy target base' should create an optional pending set."""
        from src.cards.effects_parser import parse_effect_text
        effect = parse_effect_text("You may destroy target base")
        assert effect.is_mandatory is False

    def test_mandatory_destroy_creates_mandatory_pending(self):
        """Mandatory destroy should not allow skipping."""
        game, p1, p2 = _make_game_with_players()
        base = Card("EnemyBase", 10, 0, [], CardType.BASE, defense=3)
        p2.bases.append(base)

        effect = Effect(CardEffectType.DESTROY_BASE, 1, "Destroy target base",
                        is_mandatory=True)
        effect.apply(game, p1, None)

        pending = p1.get_current_pending_set()
        assert pending is not None
        assert pending.mandatory is True

    def test_optional_destroy_allows_skip(self):
        """Optional destroy should allow skipping."""
        game, p1, p2 = _make_game_with_players()
        base = Card("EnemyBase", 10, 0, [], CardType.BASE, defense=3)
        p2.bases.append(base)

        effect = Effect(CardEffectType.DESTROY_BASE, 1, "You may destroy",
                        is_mandatory=False)
        effect.apply(game, p1, None)

        pending = p1.get_current_pending_set()
        assert pending is not None
        assert pending.mandatory is False

    def test_mandatory_destroy_no_bases_no_block(self):
        """Mandatory destroy with no targets should not create pending actions."""
        game, p1, p2 = _make_game_with_players()
        # Opponent has no bases
        effect = Effect(CardEffectType.DESTROY_BASE, 1, "Destroy target base",
                        is_mandatory=True)
        effect.apply(game, p1, None)

        # Should not block — no pending actions created
        assert len(p1.pending_action_sets) == 0


class TestE3ExecutionValidation:
    """E3: Execution-time validation guards for defense in depth."""

    def test_buy_card_rejects_insufficient_trade(self):
        """BUY_CARD should fail if player can't afford it."""
        game, p1, _ = _make_game_with_players()
        expensive = Card("Expensive", 5, 8, [], "ship")
        game.trade_row.append(expensive)
        p1.trade = 3  # not enough

        action = Action(type=ActionType.BUY_CARD, card=expensive, card_id=5)
        game.execute_action(action)

        # Card should NOT be purchased
        assert expensive in game.trade_row
        assert expensive not in p1.discard_pile

    def test_attack_base_rejects_non_outpost_when_outpost_exists(self):
        """Cannot attack a regular base while outposts are standing."""
        game, p1, p2 = _make_game_with_players()
        outpost = Card("Outpost", 10, 0, [], CardType.OUTPOST, defense=3)
        base = Card("Base", 11, 0, [], CardType.BASE, defense=2)
        p2.bases.extend([outpost, base])
        p1.combat = 5

        # Try attacking the regular base while outpost exists
        action = Action(type=ActionType.ATTACK_BASE, card=base,
                        card_id=11, target_id=11)
        game.execute_action(action)

        # Base should still be alive
        assert base in p2.bases
        assert p1.combat == 5  # combat not consumed

    def test_attack_base_allows_outpost_when_outpost_exists(self):
        """Can attack outpost even when regular bases also exist."""
        game, p1, p2 = _make_game_with_players()
        outpost = Card("Outpost", 10, 0, [], CardType.OUTPOST, defense=3)
        base = Card("Base", 11, 0, [], CardType.BASE, defense=2)
        p2.bases.extend([outpost, base])
        p1.combat = 5

        action = Action(type=ActionType.ATTACK_BASE, card=outpost,
                        card_id=10, target_id=10)
        game.execute_action(action)

        assert outpost not in p2.bases
        assert p1.combat == 2  # 5 - 3 defense

    def test_attack_player_rejects_self_target(self):
        """ATTACK_PLAYER must not allow targeting self."""
        game, p1, p2 = _make_game_with_players()
        p1.combat = 10
        p1_idx = game.players.index(p1)

        action = Action(type=ActionType.ATTACK_PLAYER, target_id=p1_idx)
        game.execute_action(action)

        # P1's health should be unchanged
        assert p1.health == 50
        assert p1.combat == 10  # combat not consumed

    def test_attack_player_rejects_zero_combat(self):
        """ATTACK_PLAYER with 0 combat should not deal damage."""
        game, p1, p2 = _make_game_with_players()
        p1.combat = 0
        p2_idx = game.players.index(p2)

        action = Action(type=ActionType.ATTACK_PLAYER, target_id=p2_idx)
        game.execute_action(action)

        assert p2.health == 50

    def test_attack_player_blocked_by_outpost(self):
        """ATTACK_PLAYER should not deal damage while outpost is active."""
        game, p1, p2 = _make_game_with_players()
        outpost = Card("Wall", 10, 0, [], CardType.OUTPOST, defense=3)
        p2.bases.append(outpost)
        p1.combat = 10
        p2_idx = game.players.index(p2)

        action = Action(type=ActionType.ATTACK_PLAYER, target_id=p2_idx)
        game.execute_action(action)

        assert p2.health == 50
        assert p1.combat == 10


class TestE4GameOverCentralized:
    """E4: Game-over detection via centralized _check_player_defeated()."""

    def test_game_over_on_lethal_damage(self):
        """Game should end when a player's health reaches 0."""
        game, p1, p2 = _make_game_with_players()
        p2.health = 5
        p1.combat = 10
        p2_idx = game.players.index(p2)

        action = Action(type=ActionType.ATTACK_PLAYER, target_id=p2_idx)
        game.execute_action(action)

        assert game.is_game_over is True
        assert p2.health == -5

    def test_game_over_winner_is_attacker(self):
        """Winner should be the player who dealt lethal damage."""
        game, p1, p2 = _make_game_with_players()
        p2.health = 1
        p1.combat = 5

        action = Action(type=ActionType.ATTACK_PLAYER,
                        target_id=game.players.index(p2))
        game.execute_action(action)

        assert game.is_game_over is True
        assert game.get_winner() == p1.name

    def test_no_game_over_when_health_positive(self):
        """Game should not end when both players have positive health."""
        game, p1, p2 = _make_game_with_players()
        p2.health = 50
        p1.combat = 10

        action = Action(type=ActionType.ATTACK_PLAYER,
                        target_id=game.players.index(p2))
        game.execute_action(action)

        assert game.is_game_over is False
        assert p2.health == 40


class TestE5ExplorerPileConsistency:
    """E5: Explorer pile indexing consistency between action gen and execution."""

    def test_explorer_buy_works_end_to_end(self):
        """Buying an Explorer should work via action generation → execution."""
        game, p1, _ = _make_game_with_players()
        game.card_index_map = {"Scout": 0, "Viper": 1, "Explorer": 2}
        game.setup_explorer_pile()
        p1.trade = 5

        assert len(game.explorer_pile) > 0
        explorer = game.explorer_pile[-1]  # action gen references last

        action = Action(type=ActionType.BUY_CARD, card=explorer,
                        card_id=explorer.index)
        game.execute_action(action)

        assert explorer in p1.discard_pile
        assert p1.trade == 3  # 5 - 2 cost

    def test_explorer_pile_decrements(self):
        """Each explorer purchase should reduce pile size by 1."""
        game, p1, _ = _make_game_with_players()
        game.card_index_map = {"Scout": 0, "Viper": 1, "Explorer": 2}
        game.setup_explorer_pile()
        initial_count = len(game.explorer_pile)
        p1.trade = 10

        for _ in range(3):
            if game.explorer_pile:
                explorer = game.explorer_pile[-1]
                action = Action(type=ActionType.BUY_CARD, card=explorer,
                                card_id=explorer.index)
                game.execute_action(action)

        assert len(game.explorer_pile) == initial_count - 3
