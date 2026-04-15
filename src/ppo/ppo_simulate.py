import glob
import os
import torch
import pickle
from datetime import datetime
from pathlib import Path
from src.config import DataConfig, SimConfig
from src.engine.game import Game
from src.ai.ppo_agent import PPOAgent
from src.ai.random_agent import RandomAgent
from src.utils.logger import log, set_verbose


def get_latest_model(models_dir="models"):
    model_files = glob.glob(os.path.join(models_dir, "ppo_agent_*.pth"))
    if not model_files:
        return None
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]


def simulate(
    data_cfg: DataConfig,
    sim_cfg: SimConfig,
    model1_path: str | None = None,
    model2_path: str | None = None,
    device: str = "cuda",
):
    """Run PPO vs opponent simulation."""
    set_verbose(False)
    cards = data_cfg.load_cards()
    names = data_cfg.get_card_names(cards)

    # Player 1: PPO agent
    resolved_model1 = model1_path or get_latest_model(data_cfg.models_dir)
    if not resolved_model1:
        raise RuntimeError("No PPO model found for player 1.")
    log(f"Loading PPO model for player 1 from {resolved_model1}")
    agent1 = PPOAgent("PPO_1", names, model_path=resolved_model1)

    # Player 2: PPO agent or random agent
    if sim_cfg.player2_random or not model2_path:
        agent2 = RandomAgent("Rand")
        log("Player 2 set to RandomAgent.")
    else:
        log(f"Loading PPO model for player 2 from {model2_path}")
        agent2 = PPOAgent("PPO_2", names, model_path=model2_path)

    wins1, wins2 = 0, 0
    all_experiences = []
    for i in range(sim_cfg.games):
        game = Game(cards)
        game.add_player(agent1.name, agent1)
        game.add_player(agent2.name, agent2)
        game.start_game()
        done = False
        while not done:
            done = game.step()
        winner = game.get_winner()
        if winner == agent1.name:
            wins1 += 1
        elif winner == agent2.name:
            wins2 += 1
        # Save PPOAgent transitions for player 1
        if isinstance(agent1, PPOAgent):
            batch = agent1.finish_batch()
            cpu_batch = tuple(
                x.cpu().numpy() if hasattr(x, 'cpu') else x
                for x in batch if x is not None
            )
            all_experiences.append(cpu_batch)
    # Save all experiences to a file after all games
    if all_experiences:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_path = f"experiences_sim_{ts}.pkl"
        with open(exp_path, 'wb') as f:
            pickle.dump(all_experiences, f)
        print(f"Saved PPOAgent experiences to {exp_path}")
    print(f"\nResults after {sim_cfg.games} games:")
    print(f"{agent1.name} wins: {wins1}")
    print(f"{agent2.name} wins: {wins2}")
    print(f"Win rate: {agent1.name}: {wins1/sim_cfg.games:.2%}, {agent2.name}: {wins2/sim_cfg.games:.2%}")
