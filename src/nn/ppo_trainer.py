import argparse
from datetime import datetime
from pathlib import Path
import torch
import random

from src.cards.card import Card
from src.ai.agent import Agent
from src.nn.state_encoder import encode_state
from src.engine.actions import get_available_actions
from src.cards.loader import load_trade_deck_cards
from src.engine.game import Game
from src.ai.ppo_agent import PPOAgent
from src.ai.random_agent import RandomAgent
from src.utils.logger import log, set_verbose
from typing import Tuple

def run_episode(agent: PPOAgent, opponent: Agent, cards: list[Card], card_names: list[str]):
    game = Game(cards)
    game.add_player(agent.name, agent)
    game.add_player(opponent.name, opponent)
    game.start_game()
    done = False
    while not done:
        # agent's turn
        a1 = agent.make_decision(game)
        reward, done = game.step(a1)
        agent.store_reward(reward, done)
        if done: break
        # opponent's turn (no learning)
        a2 = opponent.make_decision(game)
        _, done = game.step(a2)
    # bootstrap value for last state
    _, next_value = agent.model(
        encode_state(game, True, card_names, get_available_actions(game, game.current_player))
        .to(agent.device)
    )
    return agent.finish_batch(next_value)

def main():
    parser = argparse.ArgumentParser("PPO Trainer")
    parser.add_argument("--episodes",    type=int,   default=1000)
    parser.add_argument("--updates",     type=int,   default=10)
    parser.add_argument("--cards-path",  type=str,   default="data/cards.csv")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--lam",         type=float, default=0.95)
    parser.add_argument("--clip-eps",    type=float, default=0.2)
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch-size",  type=int,   default=64)
    args = parser.parse_args()

    set_verbose(False)
    cards = load_trade_deck_cards(args.cards_path, filter_sets=["Core Set"])
    names = [c.name for c in cards]
    names = list(dict.fromkeys(names)) + ["Scout","Viper","Explorer"]

    agent    = PPOAgent("PPO", names,
                       lr=args.lr,
                       gamma=args.gamma,
                       lam=args.lam,
                       clip_eps=args.clip_eps,
                       epochs=args.epochs,
                       batch_size=args.batch_size)
    opponent = RandomAgent("Rand")

    for upd in range(1, args.updates + 1):
        log(f"Starting update {upd}/{args.updates}")
        # collect trajectories
        all_data = [run_episode(agent, opponent, cards, names)
                    for _ in range(args.episodes)]

        # unpack & concat
        S, A, OL, R, Adv = zip(*all_data)
        states   = torch.cat(S)
        actions  = torch.cat(A)
        old_lp   = torch.cat(OL)
        returns  = torch.cat(R)
        advs     = torch.cat(Adv)

        # perform PPO update
        agent.update(states, actions, old_lp, returns, advs)

        # save checkpoint per update
        ts = datetime.now().strftime("%m%d_%H%M")
        Path("models").mkdir(exist_ok=True)
        torch.save(agent.model.state_dict(),
                   f"models/ppo_agent_{ts}_upd{upd}.pth")
        log(f"Update {upd} complete.")

    log("All updates finished.")

if __name__ == "__main__":
    main()