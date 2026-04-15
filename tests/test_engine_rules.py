"""Tests for game engine rule correctness — Work Unit A: Base & Outpost System."""
import pytest
from src.cards.card import Card
from src.cards.effects import Effect, CardEffectType
from src.engine.player import Player
from src.engine.actions import get_available_actions, ActionType, Action
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
        """A card with ally_factions=["*"] counts as ally for every faction."""
        player = Player("P1", Agent("P1"))
        mech_world = _make_outpost(name="Mech World", faction="Machine Cult")
        mech_world.ally_factions = ["*"]
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
        """Default behavior: ally_factions=None uses card.faction."""
        player = Player("P1", Agent("P1"))
        card = Card("Ship", 0, 1, [], "ship", faction="Blob")
        assert card.ally_factions is None
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

        assert cloned.ally_factions == ["*"]
        assert cloned.ally_factions is not card.ally_factions

    def test_clone_none_ally_factions(self):
        card = Card("Ship", 0, 1, [], "ship")
        cloned = card.clone()
        assert cloned.ally_factions is None

    def test_mech_world_loaded_from_csv(self):
        """Integration: Mech World from cards.csv gets ally_factions=["*"]."""
        from src.config import DataConfig
        cfg = DataConfig()
        cards = cfg.load_cards()
        mech_worlds = [c for c in cards if c.name == "Mech World"]
        assert len(mech_worlds) == 1
        assert mech_worlds[0].ally_factions == ["*"]

    def test_wildcard_enables_ally_ability(self):
        """Mech World should satisfy any faction's ally requirement."""
        game, p1, _ = _make_game_with_players()
        mech_world = _make_outpost(name="Mech World", faction="Machine Cult")
        mech_world.ally_factions = ["*"]
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
                              card_id="Target", card_source="trade")
        game.execute_action(scrap_action)

        # Row should have refilled with the filler card
        assert target not in game.trade_row
        assert filler in game.trade_row
        assert len(game.trade_row) == initial_row_size


# ── Work Unit C: Cross-Player Effects ────────────────────────────────────────


class TestC1ForcedDiscard:
    """C1: Forced discard must target the opponent's hand."""

    def test_forced_discard_creates_pending_on_current_player(self):
        """TARGET_DISCARD should add pending actions to the current player."""
        game, p1, p2 = _make_game_with_players()
        opp_card = Card("OppCard", 0, 1, [], "ship")
        p2.hand.append(opp_card)

        effect = Effect(CardEffectType.TARGET_DISCARD, 1,
                        card_targets=["opponent"])
        effect.apply(game, p1, None)

        # Pending actions should be on P1 (current player), not P2
        assert len(p1.pending_action_sets) == 1
        assert len(p2.pending_action_sets) == 0

    def test_forced_discard_removes_from_opponent_hand(self):
        """Executing the discard action should remove from opponent's hand."""
        game, p1, p2 = _make_game_with_players()
        opp_card = Card("OppCard", 0, 1, [], "ship")
        p2.hand.append(opp_card)

        effect = Effect(CardEffectType.TARGET_DISCARD, 1,
                        card_targets=["opponent"])
        effect.apply(game, p1, None)

        # Execute the pending discard
        pending = p1.get_current_pending_set()
        assert pending is not None
        discard_action = pending.actions[0]
        game.execute_action(discard_action)

        assert opp_card not in p2.hand
        assert opp_card in p2.discard_pile

    def test_forced_discard_does_not_touch_current_player_hand(self):
        """Current player's hand must not be affected."""
        game, p1, p2 = _make_game_with_players()
        my_card = Card("MyCard", 0, 1, [], "ship")
        p1.hand.append(my_card)
        opp_card = Card("OppCard", 1, 1, [], "ship")
        p2.hand.append(opp_card)

        effect = Effect(CardEffectType.TARGET_DISCARD, 1,
                        card_targets=["opponent"])
        effect.apply(game, p1, None)

        pending = p1.get_current_pending_set()
        game.execute_action(pending.actions[0])

        assert my_card in p1.hand  # current player's hand untouched


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
        target_names = [a.target_id for a in pending.actions]
        # Both should be targetable
        assert "EnemyOutpost" in target_names
        assert "EnemyBase" in target_names

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

        # Create Embassy Yacht's COMPLEX child effect
        yacht_effect = Effect(CardEffectType.COMPLEX, 0,
                              text="If you have two or more bases in play, draw two cards")
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

        yacht_effect = Effect(CardEffectType.COMPLEX, 0,
                              text="If you have two or more bases in play, draw two cards")
        yacht_effect.apply(game, p1, None)

        assert len(p1.hand) == initial_hand

    def test_no_draw_with_zero_bases(self):
        game, p1, _ = _make_game_with_players()
        p1.deck.append(Card("D1", 0, 0, [], "ship"))
        initial_hand = len(p1.hand)

        yacht_effect = Effect(CardEffectType.COMPLEX, 0,
                              text="If you have two or more bases in play, draw two cards")
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
                Effect(CardEffectType.COMPLEX, 0,
                       text="If you have two or more bases in play, draw two cards"),
            ]),
        ], "ship", faction="Trade Federation")
        p1.hand.append(yacht)

        play_action = Action(type=ActionType.PLAY_CARD, card=yacht, card_id=yacht.name)
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

        blob_world_effect = Effect(CardEffectType.COMPLEX, 0,
                                   text="Draw a card for each Blob card that you've played this turn")
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

        blob_world_effect = Effect(CardEffectType.COMPLEX, 0,
                                   text="Draw a card for each Blob card that you've played this turn")
        blob_world_effect.apply(game, p1, None)

        assert len(p1.hand) == 2

    def test_non_matching_faction_not_counted(self):
        """Non-Blob factions should not trigger draws."""
        game, p1, _ = _make_game_with_players()
        se = Card("SEShip", 0, 1, [], "ship", faction="Star Empire")
        p1.played_cards.append(se)
        p1.deck.append(Card("D1", 1, 0, [], "ship"))
        initial_hand = len(p1.hand)

        blob_world_effect = Effect(CardEffectType.COMPLEX, 0,
                                   text="Draw a card for each Blob card that you've played this turn")
        blob_world_effect.apply(game, p1, None)

        assert len(p1.hand) == initial_hand


class TestD3RecyclingStation:
    """D3: Recycling Station discard-then-draw via pending action completion."""

    def _make_recycling_station_effect(self):
        return Effect(CardEffectType.COMPLEX, 0,
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
        return Effect(CardEffectType.COMPLEX, 0,
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
        scrap_hand = [a for a in pending.actions if a.card_source == "hand" and a.card is h1][0]
        game.execute_action(scrap_hand)

        # Scrap from discard
        pending = p1.get_current_pending_set()
        scrap_disc = [a for a in pending.actions if a.card_source == "discard" and a.card is d1][0]
        game.execute_action(scrap_disc)

        assert p1.get_current_pending_set() is None
        assert h1 not in p1.hand
        assert d1 not in p1.discard_pile
        assert len([c for c in p1.hand if c.name in ("D1", "D2")]) == 2
