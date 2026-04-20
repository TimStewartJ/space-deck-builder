import torch

# Zone names in fixed order — matches ZONE_NAMES in ppo_actor_critic.py.
# Exposed here so encoder helpers can build [B, Z, C] presence stacks
# without importing the model module.
ZONE_NAMES = [
    'trade_row',
    'train_hand', 'train_disc', 'train_deck', 'train_bases',
    'opp_unseen', 'opp_disc', 'opp_bases',
]


def unpack_state(x: torch.Tensor, num_cards: int, action_dim: int):
    """
    Splits a state-vector into named pieces.

    Training player has 4 card zones (hand, discard, deck, bases).
    Opponent has 3 card zones (unseen=hand+deck, discard, bases) and
    6 resource scalars (includes hand_size) to respect hidden information.
    """
    single = False
    if x.dim() == 1:
        x = x.unsqueeze(0)
        single = True

    idx = 0
    out = {}

    # 1) four flags
    out['is_train'] = x[:, idx:idx+1]; idx += 1
    out['is_first'] = x[:, idx:idx+1]; idx += 1
    out['can_buy'] = x[:, idx:idx+1]; idx += 1
    out['has_actions'] = x[:, idx:idx+1]; idx += 1

    # 2) trade row
    out['trade_row'] = x[:, idx:idx+num_cards]; idx += num_cards

    # 3) training player: 5 resources + 4 zones
    out['train_res']   = x[:, idx:idx+5]; idx += 5
    out['train_hand']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['train_disc']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['train_deck']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['train_bases'] = x[:, idx:idx+num_cards]; idx += num_cards

    # 4) opponent: 6 resources + 3 zones (hand+deck merged into unseen)
    out['opp_res']    = x[:, idx:idx+6]; idx += 6
    out['opp_unseen'] = x[:, idx:idx+num_cards]; idx += num_cards
    out['opp_disc']   = x[:, idx:idx+num_cards]; idx += num_cards
    out['opp_bases']  = x[:, idx:idx+num_cards]; idx += num_cards

    assert idx == x.shape[-1], f"State layout mismatch: unpacked {idx} elements but tensor has {x.shape[-1]}"

    return out, single


def unpack_state_tokens(x: torch.Tensor, num_cards: int, action_dim: int):
    """Reshape a flat state vector into per-zone presence + a numeric vector.

    Returns:
        presence: ``[B, Z, C]`` per-zone, per-card presence values, with zones
            in fixed ``ZONE_NAMES`` order.
        numerics: ``[B, NUMERIC_DIM]`` concatenation of all flag and resource
            scalars (4 flags + 5 train resources + 6 opponent resources).
        single: ``True`` when the input was 1-D (caller should squeeze the
            batch dim from downstream outputs).

    No new buffers are allocated for the encoder hot path — this is a pure
    view/stack of the existing flat layout produced by ``encode_state``.
    """
    pieces, single = unpack_state(x, num_cards, action_dim)
    presence = torch.stack(
        [pieces[name] for name in ZONE_NAMES], dim=1,
    )  # [B, Z, C]
    numerics = torch.cat([
        pieces['is_train'], pieces['is_first'], pieces['can_buy'],
        pieces['has_actions'], pieces['train_res'], pieces['opp_res'],
    ], dim=1)  # [B, 4 + 5 + 6]
    return presence, numerics, single