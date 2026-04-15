from datetime import datetime
from pathlib import Path
import torch
import time
import copy

from src.config import DataConfig, PPOConfig, RunConfig, DeviceConfig, save_checkpoint
from src.cards.card import Card
from src.ai.agent import Agent
from src.ai.ppo_agent import PPOAgent
from src.encoding.action_encoder import get_action_space_size
from src.ppo.batch_runner import BatchRunner
from src.ppo.opponent_pool import OpponentPool
from src.utils.logger import log, set_disabled, set_verbose


def _make_runner(agent, cards, card_names, action_dim, ppo_cfg, run_cfg, data_cfg,
                 make_opponent, registry, pool=None):
    """Create the appropriate batch runner based on num_workers config."""
    sim_device = agent.simulation_device
    num_concurrent = min(run_cfg.episodes, run_cfg.num_concurrent)

    if run_cfg.num_workers > 1:
        from src.ppo.mp_batch_runner import MultiProcessBatchRunner

        # Extract snapshot data from the pool for serialization to workers
        snapshot_state_dicts = None
        self_play_ratio = run_cfg.self_play_ratio
        pfsp_weights = None
        if pool is not None and pool.has_snapshots:
            snapshot_state_dicts = list(pool._snapshots)
            snap_names = [n for n, _ in snapshot_state_dicts]
            pfsp_weights = pool._pfsp_weights(snap_names)

        return MultiProcessBatchRunner(
            model=agent.model,
            card_names=card_names,
            cards=cards,
            action_dim=action_dim,
            device=sim_device,
            data_config=data_cfg,
            opponent_spec=run_cfg.opponents,
            num_concurrent=num_concurrent,
            num_workers=run_cfg.num_workers,
            ppo_config=ppo_cfg,
            registry=registry,
            snapshot_state_dicts=snapshot_state_dicts,
            self_play_ratio=self_play_ratio,
            pfsp_weights=pfsp_weights,
        )
    else:
        return BatchRunner(
            model=agent.model,
            card_names=card_names,
            cards=cards,
            action_dim=action_dim,
            device=sim_device,
            opponent_factory=make_opponent,
            num_concurrent=num_concurrent,
            ppo_config=ppo_cfg,
            registry=registry,
        )


def train(
    data_cfg: DataConfig,
    ppo_cfg: PPOConfig,
    run_cfg: RunConfig,
    dev_cfg: DeviceConfig,
    model_path: str | None = None,
    load_latest: bool = False,
):
    """Run PPO training with the given configuration."""
    if load_latest and not model_path:
        import glob, os
        model_files = glob.glob(os.path.join(data_cfg.models_dir, "ppo_agent_*.pth"))
        if model_files:
            model_files.sort(key=os.path.getmtime, reverse=True)
            model_path = model_files[0]
            print(f"Auto-loading latest PPO model: {model_path}")
        else:
            print("No PPO model found in models directory to auto-load.")

    set_verbose(False)
    set_disabled(True)
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    card_names = registry.card_names

    agent    = PPOAgent("PPO", card_names,
                       ppo_config=ppo_cfg,
                       device_config=dev_cfg,
                       model_path=model_path,
                       registry=registry)
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
    print(f"Model has {num_params / 1_000_000:.2f}M parameters. Actor input size: {input_size}, output size: {output_size}.")

    # Set up the opponent pool from config
    pool = OpponentPool(
        opponent_spec=run_cfg.opponents,
        self_play_ratio=run_cfg.self_play_ratio,
        pfsp_mode=run_cfg.pfsp_mode,
    )
    opp_types = pool.opponent_types
    pool_msg = f"Opponent pool: {', '.join(opp_types)}"
    if run_cfg.self_play:
        pfsp_info = f", pfsp: {run_cfg.pfsp_mode}" if run_cfg.pfsp_mode != "uniform" else ""
        pool_msg += f" + self-play (ratio: {run_cfg.self_play_ratio}{pfsp_info})"
    print(pool_msg)

    total_time_spent_on_updates = 0.0
    total_time_spent_on_episodes = 0.0
    total_time_spent_on_eval = 0.0
    overall_start_time = time.perf_counter()

    sim_device = str(agent.simulation_device)

    for upd in range(1, run_cfg.updates + 1):
        make_opponent = pool.make_factory(card_names, device=sim_device, registry=registry)
        snap_msg = " + snapshots" if pool.has_snapshots else ""
        print(f"Starting update {upd}/{run_cfg.updates} "
              f"(opponents: {', '.join(opp_types)}{snap_msg})")

        # Collect trajectories
        start_time = time.time()
        action_dim = get_action_space_size(card_names)
        runner = _make_runner(
            agent, cards, card_names, action_dim,
            ppo_cfg, run_cfg, data_cfg, make_opponent, registry, pool=pool,
        )
        states, actions, old_lp, returns, advs, masks = runner.run_episodes(run_cfg.episodes)
        duration_episodes = time.time() - start_time
        total_time_spent_on_episodes += duration_episodes
        workers_msg = f" ({run_cfg.num_workers} workers)" if run_cfg.num_workers > 1 else ""
        print(f"Finished {run_cfg.episodes} episodes in {duration_episodes:.2f}s.{workers_msg}")

        # Update PFSP win rate estimates from batch results
        if run_cfg.self_play and run_cfg.pfsp_mode != "uniform":
            pool.update_results(
                {k: tuple(v) for k, v in runner.opponent_results.items()}
            )
            pfsp_summary = pool.get_pfsp_summary()
            if pfsp_summary:
                parts = [
                    f"{k}: wr={v['ema_win_rate']:.0%} w={v['weight']:.2f}"
                    for k, v in pfsp_summary.items()
                ]
                print(f"  PFSP: {', '.join(parts)}")

        # --- Device boundary: sim_device → main_device ---
        # run_episodes() returns tensors on simulation_device.
        # Transfer to main_device for the PPO gradient update.
        agent.device = agent.main_device
        agent.model.to(agent.main_device)
        states = states.to(agent.device)
        actions = actions.to(agent.device)
        old_lp = old_lp.to(agent.device)
        returns = returns.to(agent.device)
        advs = advs.to(agent.device)
        if masks is not None:
            masks = masks.to(agent.device)

        # Perform PPO update
        start_time = time.time()
        agent.update(states, actions, old_lp, returns, advs, masks)
        duration_update = time.time() - start_time
        total_time_spent_on_updates += duration_update
        print(f"Update {upd} complete in {duration_update:.2f}s. State size: {states.shape}")
        print(f"Loc Emb: {agent.model.loc_emb.weight.grad is not None and agent.model.loc_emb.weight.grad.norm().item()}")

        # Add snapshot for self-play after each update
        if run_cfg.self_play:
            snapshot_sd = copy.deepcopy(agent.model).cpu().state_dict()
            pool.add_snapshot(snapshot_sd, f"PPO_{upd}")

        # Evaluate performance (every N updates, and always on the last)
        is_last_update = upd == run_cfg.updates
        wins = -1  # default: no eval this round
        if upd % run_cfg.eval_every == 0 or is_last_update:
            start_time = time.time()
            wins = _run_per_opponent_eval(
                agent, pool, card_names, cards, run_cfg, ppo_cfg, upd,
                registry=registry, data_cfg=data_cfg,
            )
            agent.clear_buffers()
            duration_eval = time.time() - start_time
            total_time_spent_on_eval += duration_eval
        else:
            print(f"Skipping eval (next eval at update {upd + run_cfg.eval_every - upd % run_cfg.eval_every}).")

        # Save checkpoint per update
        ts = datetime.now().strftime("%m%d_%H%M")
        Path(data_cfg.models_dir).mkdir(exist_ok=True)
        ckpt_path = f"{data_cfg.models_dir}/ppo_agent_{ts}_upd{upd}_wins{wins}.pth"
        save_checkpoint(
            ckpt_path,
            agent.model.state_dict(),
            ppo_config=ppo_cfg,
            model_config=agent.model_config,
            run_config=run_cfg,
            device_config=dev_cfg,
            update=upd,
        )
        print("Checkpoint saved.")

    print(f"Total time spent on episodes: {total_time_spent_on_episodes:.2f}s\n\tAverage per update: {total_time_spent_on_episodes / run_cfg.updates:.2f}s")
    print(f"Total time spent on PPO updates: {total_time_spent_on_updates:.2f}s\n\tAverage per update: {total_time_spent_on_updates / run_cfg.updates:.2f}s")
    print(f"Total time spent on evaluation: {total_time_spent_on_eval:.2f}s\n\tAverage per update: {total_time_spent_on_eval / run_cfg.updates:.2f}s")
    print("All updates finished.")
    # Log average decision time per decision
    avg_decision_time = agent.get_average_decision_time()
    print(f"Average PPOAgent decision time: {avg_decision_time:.6f} seconds per decision.")
    print(f"Overall time spent: {time.perf_counter() - overall_start_time:.2f}s")


def _run_per_opponent_eval(
    agent: PPOAgent,
    pool: OpponentPool,
    card_names: list[str],
    cards: list,
    run_cfg: RunConfig,
    ppo_cfg: PPOConfig,
    upd: int,
    registry=None,
    data_cfg: DataConfig | None = None,
) -> int:
    """Run evaluation against each opponent type separately, log per-type results.

    Uses MultiProcessBatchRunner when num_workers > 1 and data_cfg is provided.

    Returns the total wins across all opponent types.
    """
    opp_types = pool.opponent_types
    action_dim = get_action_space_size(card_names)
    games_per_type = max(1, run_cfg.eval_games // len(opp_types))
    total_wins = 0
    total_games = 0
    use_mp = run_cfg.num_workers > 1 and data_cfg is not None

    print(f"Evaluating after update {upd} ({games_per_type} games × {len(opp_types)} opponent types)...")

    for opp_type in opp_types:
        if use_mp:
            from src.ppo.mp_batch_runner import MultiProcessBatchRunner
            eval_runner = MultiProcessBatchRunner(
                model=agent.model,
                card_names=card_names,
                cards=cards,
                action_dim=action_dim,
                device=agent.simulation_device,
                data_config=data_cfg,
                opponent_spec=opp_type,
                num_concurrent=min(games_per_type, run_cfg.num_concurrent),
                num_workers=run_cfg.num_workers,
                ppo_config=ppo_cfg,
                registry=registry,
            )
        else:
            factory = pool.make_factory_for_type(opp_type)
            eval_runner = BatchRunner(
                model=agent.model,
                card_names=card_names,
                cards=cards,
                action_dim=action_dim,
                device=agent.simulation_device,
                opponent_factory=factory,
                num_concurrent=min(games_per_type, run_cfg.num_concurrent),
                ppo_config=ppo_cfg,
                registry=registry,
            )
        wins, losses, eval_steps = eval_runner.run_eval(games_per_type)
        win_rate = wins / games_per_type
        avg_steps = eval_steps / games_per_type if games_per_type > 0 else 0
        print(f"  vs {opp_type}: {wins}/{games_per_type} wins ({win_rate:.0%}), avg {avg_steps:.0f} steps/game")
        total_wins += wins
        total_games += games_per_type

    overall_rate = total_wins / total_games if total_games > 0 else 0
    print(f"  Overall: {total_wins}/{total_games} wins ({overall_rate:.0%})")
    return total_wins