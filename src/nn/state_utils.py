import torch

def unpack_state(x: torch.Tensor, num_cards: int, action_dim: int):
    """
    Splits a state‐vector into:
      is_train, is_first, trade_row,
      train_res, train_hand, train_disc, train_deck, train_bases,
      opp_res, opp_hand, opp_disc, opp_deck, opp_bases
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

    # 2) trade_row one‐hot
    out['trade_row'] = x[:, idx:idx+num_cards]; idx += num_cards

    # 3) train player
    out['train_res']   = x[:, idx:idx+5]; idx += 5
    out['train_hand']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['train_disc']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['train_deck']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['train_bases']= x[:, idx:idx+num_cards]; idx += num_cards

    # 4) opponent
    out['opp_res']   = x[:, idx:idx+5]; idx += 5
    out['opp_hand']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['opp_disc']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['opp_deck']  = x[:, idx:idx+num_cards]; idx += num_cards
    out['opp_bases']= x[:, idx:idx+num_cards]; idx += num_cards

    return out, single