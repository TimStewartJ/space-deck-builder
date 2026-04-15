import torch

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