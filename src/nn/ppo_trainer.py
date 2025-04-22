import argparse
from datetime import datetime
from pathlib import Path
import torch
import time

from src.cards.card import Card
from src.ai.agent import Agent
from src.cards.loader import load_trade_deck_cards
from src.engine.game import Game
from src.ai.ppo_agent import PPOAgent
from src.ai.random_agent import RandomAgent
from src.utils.logger import log, set_verbose

def run_episode(agent: PPOAgent, opponent: Agent, cards: list[Card]):
    game = Game(cards)
    game.add_player(agent.name, agent)
    game.add_player(opponent.name, opponent)
    game.start_game()
    done = False
    while not done:
        current_player = game.current_player
        # Determine if the current player is the training agent
        is_agent = current_player.name == agent.name
        done = game.step()
        reward = 0.0
        if done:
            if game.get_winner() == agent.name:
                reward = 1.0
            else:
                reward = -1.0
        if is_agent:
            agent.store_reward(reward, done)
    return agent.finish_batch()

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
    parser.add_argument("--device",      type=str,   default="cuda", help="Device to run ML (cuda or cpu)")
    parser.add_argument("--model-path",  type=str,   default=None, help="Path to a pretrained PPO model to load")
    parser.add_argument("--load-latest-model", action="store_true", help="If set, load the latest PPO model from the models directory if available.")
    args = parser.parse_args()

    # Determine model path if --load-latest-model is set and --model-path is not provided
    model_path = args.model_path
    if args.load_latest_model and not model_path:
        import glob, os
        model_files = glob.glob(os.path.join("models", "ppo_agent_*.pth"))
        if model_files:
            model_files.sort(key=os.path.getmtime, reverse=True)
            model_path = model_files[0]
            log(f"Auto-loading latest PPO model: {model_path}")
        else:
            log("No PPO model found in models directory to auto-load.")

    set_verbose(False)
    cards = load_trade_deck_cards(args.cards_path, filter_sets=["Core Set"], log_cards=False)
    # Create list of unique card names
    names = [c.name for c in cards]
    names = list(dict.fromkeys(names)) + ["Scout","Viper","Explorer"]

    agent    = PPOAgent("PPO", names,
                       lr=args.lr,
                       gamma=args.gamma,
                       lam=args.lam,
                       clip_eps=args.clip_eps,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       device=args.device,
                       model_path=model_path)
    opponent = RandomAgent("Rand")

    total_time_spent_on_updates = 0.0
    total_time_spent_on_episodes = 0.0
    total_time_spent_on_eval = 0.0

    for upd in range(1, args.updates + 1):
        log(f"Starting update {upd}/{args.updates}")
        # collect trajectories
        start_time = time.time()
        all_data = [run_episode(agent, opponent, cards)
                    for _ in range(args.episodes)]
        duration_episodes = time.time() - start_time
        total_time_spent_on_episodes += duration_episodes
        log(f"Finished {args.episodes} episodes in {duration_episodes:.2f}s.")

        # unpack & concat
        S, A, OL, R, Adv = zip(*all_data)
        states   = torch.cat(S)
        actions  = torch.cat(A)
        old_lp   = torch.cat(OL)
        returns  = torch.cat(R)
        advs     = torch.cat(Adv)

        # perform PPO update
        start_time = time.time()
        agent.update(states, actions, old_lp, returns, advs)
        duration_update = time.time() - start_time
        total_time_spent_on_updates += duration_update
        log(f"Update {upd} complete in {duration_update:.2f}s.")

        # save checkpoint per update
        ts = datetime.now().strftime("%m%d_%H%M")
        Path("models").mkdir(exist_ok=True)
        torch.save(agent.model.state_dict(),
                   f"models/ppo_agent_{ts}_upd{upd}.pth")
        log(f"Checkpoint saved.")

        # Evaluate performance over 50 games
        start_time = time.time()
        eval_games = 50
        wins = 0
        for _ in range(eval_games):
            game = Game(cards)
            game.add_player(agent.name, agent)
            game.add_player(opponent.name, opponent)
            game.start_game()
            done = False
            while not done:
                done = game.step()
            if game.get_winner() == agent.name:
                wins += 1
        agent.clear_buffers()
        losses = eval_games - wins
        win_rate = wins / eval_games
        duration_eval = time.time() - start_time
        total_time_spent_on_eval += duration_eval
        log(f"Evaluation after update {upd}: {wins}/{eval_games} wins, {losses} losses (win rate {win_rate:.2%}) in {duration_eval:.2f}s.")

    log(f"Total time spent on episodes: {total_time_spent_on_episodes:.2f}s\n\tAverage per update: {total_time_spent_on_episodes / args.updates:.2f}s")
    log(f"Total time spent on PPO updates: {total_time_spent_on_updates:.2f}s\n\tAverage per update: {total_time_spent_on_updates / args.updates:.2f}s")
    log(f"Total time spent on evaluation: {total_time_spent_on_eval:.2f}s\n\tAverage per update: {total_time_spent_on_eval / args.updates:.2f}s")
    log("All updates finished.")

if __name__ == "__main__":
    main()