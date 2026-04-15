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
from src.nn.action_encoder import get_action_space_size
from src.ppo.batch_runner import BatchRunner
from src.utils.logger import log, set_disabled, set_verbose

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
            # agent.create_dummy_state(game)

            if game.get_winner() == agent.name:
                reward = 1.0
            else:
                reward = -1.0
            agent.fill_last_reward(reward)
    return agent.finish_batch()

def run_episode_worker(state_dict, agent_kwargs, opponent_type, opponent_state_dict, opponent_kwargs, cards, seed):
    random.seed(seed)
    torch.manual_seed(seed)
    set_disabled(True)

    # Reconstruct agent with same weights
    agent = PPOAgent("PPO", **agent_kwargs)
    agent.model.load_state_dict(state_dict)
    # Reconstruct opponent based on type
    if opponent_type == "ppo":
        opponent = PPOAgent("Opp", **opponent_kwargs)
        opponent.model.load_state_dict(opponent_state_dict)
    else:
        opponent = RandomAgent("Rand")
    return run_episode(agent, opponent, cards)

def run_eval_game_worker(agent_state_dict, agent_kwargs, opponent_type, opponent_state_dict, opponent_kwargs, cards, seed):
    random.seed(seed)
    set_disabled(True)

    # Reconstruct agent
    agent = PPOAgent("PPO", **agent_kwargs)
    agent.model.load_state_dict(agent_state_dict)
    # Reconstruct opponent
    if opponent_type == "ppo":
        opponent = PPOAgent("Opp", **opponent_kwargs)
        opponent.model.load_state_dict(opponent_state_dict)
    else:
        opponent = RandomAgent("Rand")

    # Play one game
    game = Game(cards)
    game.add_player(agent.name, agent)
    game.add_player(opponent.name, opponent)
    game.start_game()
    done = False
    while not done:
        done = game.step()
    win = 1 if game.get_winner() == agent.name else 0
    steps = len(agent.states)
    return win, steps

def main():
    parser = argparse.ArgumentParser("PPO Trainer")
    parser.add_argument("--episodes",    type=int,   default=1024)
    parser.add_argument("--updates",     type=int,   default=4)
    parser.add_argument("--cards-path",  type=str,   default="data/cards.csv")
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.995)
    parser.add_argument("--lam",         type=float, default=0.99)
    parser.add_argument("--clip-eps",    type=float, default=0.3)
    parser.add_argument("--epochs",      type=int,   default=4)
    parser.add_argument("--batch-size",  type=int,   default=1024)
    parser.add_argument("--entropy",     type=float, default=0.025, help="Entropy bonus coefficient for PPO")
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
                       entropy_coef=args.entropy,
                       main_device=args.main_device,
                       simulation_device=args.simulation_device,
                       model_path=model_path)
    # Log the parameter size of the model
    num_params = sum(p.numel() for p in agent.model.parameters() if p.requires_grad)
    # Get input and output size of the actor model
    actor = agent.model.actor
    # Try to infer input and output size from the first and last layers
    try:
        first_layer = actor[0]
        last_layer = actor[-1]
        input_size = first_layer.in_features if hasattr(first_layer, 'in_features') else "Unknown"
        output_size = last_layer.out_features if hasattr(last_layer, 'out_features') else "Unknown"
    except Exception:
        input_size = output_size = "Unknown"
    log(f"Model has {num_params / 1_000_000:.2f}M parameters. Actor input size: {input_size}, output size: {output_size}.")
    opponent = RandomAgent("Rand")

    # Keep a record of all past agent iterations
    past_agents: list[Agent] = [opponent]  # Start with the random agent as the first opponent

    total_time_spent_on_updates = 0.0
    total_time_spent_on_episodes = 0.0
    total_time_spent_on_eval = 0.0
    overall_start_time = time.perf_counter()

    for upd in range(1, args.updates + 1):
        # Select training opponent for this update
        if args.self_play and len(past_agents) > 1:
            opponent = random.choice(past_agents[:-1])
        else:
            opponent = RandomAgent("Rand")
        log(f"Starting update {upd}/{args.updates} (training opponent: {opponent.name})")
        # collect trajectories
        start_time = time.time()

        # Batched episode rollouts using BatchRunner
        runner = BatchRunner(
            model=agent.model,
            card_names=card_names,
            cards=cards,
            action_dim=get_action_space_size(card_names),
            device=agent.simulation_device,
            opponent_factory=lambda opp=opponent: opp if isinstance(opp, RandomAgent) else RandomAgent("Rand"),
            num_concurrent=min(args.episodes, 64),
        )
        states, actions, old_lp, returns, advs = runner.run_episodes(args.episodes)
        duration_episodes = time.time() - start_time
        total_time_spent_on_episodes += duration_episodes
        log(f"Finished {args.episodes} episodes in {duration_episodes:.2f}s.")

        # Move to main device for training
        agent.device = agent.main_device
        states = states.to(agent.device)
        actions = actions.to(agent.device)
        old_lp = old_lp.to(agent.device)
        returns = returns.to(agent.device)
        advs = advs.to(agent.device)

        # perform PPO update
        start_time = time.time()
        agent.update(states, actions, old_lp, returns, advs)
        duration_update = time.time() - start_time
        total_time_spent_on_updates += duration_update
        log(f"Update {upd} complete in {duration_update:.2f}s. State size: {states.shape}")
        log(f"Loc Emb: {agent.model.loc_emb.weight.grad is not None and agent.model.loc_emb.weight.grad.norm().item()}")

        # Save a deep copy of the agent after update
        past_agent = copy.deepcopy(agent)
        past_agent.clear_buffers()
        # Update the name of the agent to include the update number
        past_agent.name = f"PPO_{upd}"
        past_agents.append(past_agent)

        # Select eval opponent (same logic as training: past agent or random)

        # Evaluate performance
        log(f"Evaluating performance of {agent.name} vs {opponent.name}...")
        start_time = time.time()
        eval_games = 100
        # Prepare concurrent evaluation
        eval_state_dict = agent.model.cpu().state_dict()
        eval_agent_kwargs = {
            "card_names": card_names,
            "lr": args.lr,
            "gamma": args.gamma,
            "lam": args.lam,
            "clip_eps": args.clip_eps,
            "device": args.simulation_device,
            "main_device": args.main_device,
            "simulation_device": args.simulation_device,
            "log_debug": False,
        }
        if isinstance(opponent, PPOAgent):
            opponent_type = "ppo"
            opponent_state_dict = opponent.model.cpu().state_dict()
            opponent_kwargs = eval_agent_kwargs.copy()
        else:
            opponent_type = "random"
            opponent_state_dict = None
            opponent_kwargs = {}
        seeds = [random.randint(0, 1_000_000_000) for _ in range(eval_games)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_eval_game_worker,
                    eval_state_dict,
                    eval_agent_kwargs,
                    opponent_type,
                    opponent_state_dict,
                    opponent_kwargs,
                    cards,
                    s
                )
                for s in seeds
            ]
            results = [f.result() for f in futures]
        wins = sum(w for w, _ in results)
        steps = [s for _, s in results]
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