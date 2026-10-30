"""The action layout is the single source of truth for action indices."""
from __future__ import annotations

import pytest

from src.config import DataConfig
from src.encoding.action_encoder import (
    CARD_ACTION_TYPE_NAMES,
    END_TURN_INDEX,
    INVALID_INDEX,
    NUM_CARD_ACTION_TYPES,
    NUM_GLOBAL_ACTIONS,
    SKIP_INDEX,
    card_group_of_index,
    decode_action,
    encode_action,
    get_action_offsets,
    get_action_space_layout,
    get_action_space_size,
    num_cards_for_action_dim,
)
from src.engine.actions import ActionType


@pytest.fixture(scope="module")
def names():
    data = DataConfig()
    return data.build_registry(data.load_cards()).card_names


def test_layout_partitions_every_index_exactly_once(names):
    layout = get_action_space_layout(len(names))
    owners = {}
    for name, index in layout["global_indices"].items():
        assert index not in owners
        owners[index] = name
    for name, offset, count in layout["card_groups"]:
        assert count == len(names)
        for index in range(offset, offset + count):
            assert index not in owners
            owners[index] = name
    assert sorted(owners) == list(range(layout["action_dim"]))
    assert layout["action_dim"] == get_action_space_size(names)
    assert layout["action_dim"] == (
        NUM_GLOBAL_ACTIONS + NUM_CARD_ACTION_TYPES * len(names)
    )
    assert [name for name, _, _ in layout["card_groups"]] == list(
        CARD_ACTION_TYPE_NAMES
    )


def test_existing_groups_keep_their_indices(names):
    """Groups are only ever appended, so earlier indices never move."""
    offsets = get_action_offsets(len(names))
    assert (INVALID_INDEX, END_TURN_INDEX, SKIP_INDEX) == (0, 1, 2)
    assert offsets.play_card == 3
    assert offsets.buy_card == 52
    assert offsets.attack_player == 101
    assert offsets.attack_base == 102
    assert offsets.destroy_base == 151
    assert offsets.effect_nonscrap == 200
    assert offsets.effect_scrap == 249
    assert offsets.scrap_hand == 298
    assert offsets.scrap_discard == 347
    assert offsets.scrap_trade == 396
    assert offsets.discard == 445


def test_action_dimension_inverts_to_card_count(names):
    dim = get_action_space_size(names)
    assert num_cards_for_action_dim(dim) == len(names)
    for bad in (dim + 1, dim - 1, NUM_GLOBAL_ACTIONS):
        with pytest.raises(ValueError, match="not canonical"):
            num_cards_for_action_dim(bad)


def test_decode_then_encode_round_trips_every_index(names):
    layout = get_action_space_layout(len(names))
    for index in range(SKIP_INDEX, layout["action_dim"]):
        action = decode_action(index, names)
        assert encode_action(action, names) == index, index
    assert decode_action(INVALID_INDEX, names).type == ActionType.END_TURN
    assert decode_action(END_TURN_INDEX, names).type == ActionType.END_TURN


def test_card_group_lookup_matches_layout(names):
    for name, offset, count in get_action_space_layout(len(names))["card_groups"]:
        assert card_group_of_index(offset, len(names)) == (name, 0)
        assert card_group_of_index(offset + count - 1, len(names)) == (
            name, count - 1,
        )
    assert card_group_of_index(END_TURN_INDEX, len(names)) is None
