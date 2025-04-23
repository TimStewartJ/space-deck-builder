import argparse
from datetime import datetime
from pathlib import Path
import torch
import time
import copy
import random
from concurrent.futures import ProcessPoolExecutor

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
                agent.make_last_reward_negative()
        if is_agent:
            agent.store_reward(reward, done)
    return agent.finish_batch()

def run_episode_worker(state_dict, agent_kwargs, cards, seed):
    import random
    random.seed(seed)
    set_verbose(False)
    # Reconstruct agent with same weights
    agent = PPOAgent("PPO", **agent_kwargs)
    agent.model.load_state_dict(state_dict)
    # Always use a random opponent in worker
    opponent = RandomAgent("Rand")
    return run_episode(agent, opponent, cards)

def main():
    parser = argparse.ArgumentParser("PPO Trainer")
    parser.add_argument("--episodes",    type=int,   default=1024)
    parser.add_argument("--updates",     type=int,   default=8)
    parser.add_argument("--cards-path",  type=str,   default="data/cards.csv")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--lam",         type=float, default=0.95)
    parser.add_argument("--clip-eps",    type=float, default=0.2)
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch-size",  type=int,   default=256)
    parser.add_argument("--device",      type=str,   default="cuda", help="Device to run ML (cuda or cpu)")
    parser.add_argument("--model-path",  type=str,   default=None, help="Path to a pretrained PPO model to load")
    parser.add_argument("--load-latest-model", action="store_true", help="If set, load the latest PPO model from the models directory if available.")
    parser.add_argument("--main-device", type=str, default="cuda", help="Device for training/updates (cuda or cpu)")
    parser.add_argument("--simulation-device", type=str, default="cpu", help="Device for episode simulation (cpu or cuda)")
    parser.add_argument("--self-play", action="store_true", help="If set, the agent will play against itself instead of a random agent.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel processes for episodes")
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
    card_names = [c.name for c in cards]
    card_names = list(dict.fromkeys(card_names)) + ["Scout","Viper","Explorer"]

    agent    = PPOAgent("PPO", card_names,
                       lr=args.lr,
                       gamma=args.gamma,
                       lam=args.lam,
                       clip_eps=args.clip_eps,
                       epochs=args.epochs,
                       batch_size=args.batch_size,
                       main_device=args.main_device,
                       simulation_device=args.simulation_device,
                       model_path=model_path)
    opponent = RandomAgent("Rand")

    # Keep a record of all past agent iterations
    past_agents: list[Agent] = [opponent]  # Start with the random agent as the first opponent

    total_time_spent_on_updates = 0.0
    total_time_spent_on_episodes = 0.0
    total_time_spent_on_eval = 0.0
    overall_start_time = time.perf_counter()

    for upd in range(1, args.updates + 1):
        log(f"Starting update {upd}/{args.updates}")
        # collect trajectories
        start_time = time.time()

        # Parallel episode rollouts
        # Get model params from cpu
        state_dict   = agent.model.cpu().state_dict()
        agent_kwargs = {
            "card_names": card_names,
            "lr": args.lr,
            "gamma": args.gamma,
            "lam": args.lam,
            "clip_eps": args.clip_eps,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.simulation_device,
            "main_device": args.main_device,
            "simulation_device": args.simulation_device,
            "model_path": model_path,
            "log_debug": False,
        }
        # Generate random seeds for reproducibility
        seeds = [random.randint(0, 1_000_000_000) for _ in range(args.episodes)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(run_episode_worker, state_dict, agent_kwargs, cards, s)
                       for s in seeds]
            all_data = [f.result() for f in futures]
        duration_episodes = time.time() - start_time
        total_time_spent_on_episodes += duration_episodes
        log(f"Finished {args.episodes} episodes in {duration_episodes:.2f}s.")

        # unpack & concat
        S, A, OL, R, Adv = zip(*all_data)
        agent.device = agent.main_device  # Set device back to main device for training
        states   = torch.cat(S).to(agent.device)
        actions  = torch.cat(A).to(agent.device)
        old_lp   = torch.cat(OL).to(agent.device)
        returns  = torch.cat(R).to(agent.device)
        advs     = torch.cat(Adv).to(agent.device)

        # perform PPO update
        start_time = time.time()
        agent.update(states, actions, old_lp, returns, advs)
        duration_update = time.time() - start_time
        total_time_spent_on_updates += duration_update
        log(f"Update {upd} complete in {duration_update:.2f}s. State size: {states.shape}")

        # Save a deep copy of the agent after update
        past_agent = copy.deepcopy(agent)
        past_agent.clear_buffers()
        # Update the name of the agent to include the update number
        past_agent.name = f"PPO_{upd}"
        past_agents.append(past_agent)

        # Set opponent to a randomly chosen past agent (excluding current agent)
        if args.self_play:
            if len(past_agents) > 1:
                opponent = random.choice(past_agents[:-1])
            else:
                opponent = RandomAgent("Rand")
            log(f"Opponent set to {opponent.name}.")

        # Evaluate performance over 50 games
        log(f"Evaluating performance of {agent.name} vs {opponent.name}...")
        start_time = time.time()
        eval_games = 50
        wins = 0
        steps = []
        for _ in range(eval_games):
            game = Game(cards)
            game.add_player(agent.name, agent)
            game.add_player(opponent.name, opponent)
            game.start_game()
            done = False
            while not done:
                done = game.step()
            steps.append(len(agent.states))
            if game.get_winner() == agent.name:
                wins += 1
        agent.clear_buffers()
        losses = eval_games - wins
        win_rate = wins / eval_games
        duration_eval = time.time() - start_time
        total_time_spent_on_eval += duration_eval
        log(f"Evaluation after update {upd}: {wins}/{eval_games} wins, {losses} losses (win rate {win_rate:.2%}) in {duration_eval:.2f}s.")
        log(f"Average steps per game: {sum(steps) / len(steps):.2f}, avg time per step {duration_eval/sum(steps):.6f} seconds.")

        # save checkpoint per update
        ts = datetime.now().strftime("%m%d_%H%M")
        Path("models").mkdir(exist_ok=True)
        torch.save(agent.model.state_dict(),
                   f"models/ppo_agent_{ts}_upd{upd}_wins{wins}.pth")
        log(f"Checkpoint saved.")

    log(f"Total time spent on episodes: {total_time_spent_on_episodes:.2f}s\n\tAverage per update: {total_time_spent_on_episodes / args.updates:.2f}s")
    log(f"Total time spent on PPO updates: {total_time_spent_on_updates:.2f}s\n\tAverage per update: {total_time_spent_on_updates / args.updates:.2f}s")
    log(f"Total time spent on evaluation: {total_time_spent_on_eval:.2f}s\n\tAverage per update: {total_time_spent_on_eval / args.updates:.2f}s")
    log("All updates finished.")
    # Log average decision time per decision
    avg_decision_time = agent.get_average_decision_time()
    log(f"Average PPOAgent decision time: {avg_decision_time:.6f} seconds per decision.")
    log(f"Overall time spent: {time.perf_counter() - overall_start_time:.2f}s")

if __name__ == "__main__":
    main()