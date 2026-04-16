from datetime import datetime
from pathlib import Path
import math
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


def _compute_self_play_ratio(
    start: float, end: float, schedule: str, progress: float
) -> float:
    """Compute the self-play ratio at a given training progress.

    Args:
        start: Initial self-play ratio (at progress=0).
        end: Final self-play ratio (at progress=1).
        schedule: "constant" (always *end*), "linear", or "cosine".
        progress: Training progress in [0.0, 1.0].
    """
    if schedule == "constant":
        return end
    progress = max(0.0, min(1.0, progress))
    if schedule == "linear":
        return start + (end - start) * progress
    if schedule == "cosine":
        return start + (end - start) * (1 - math.cos(math.pi * progress)) / 2
    return end


def _make_runner(agent, cards, card_names, action_dim, ppo_cfg, run_cfg, data_cfg,
                 make_opponent, registry, pool=None, *, current_self_play_ratio=None):
    """Create the appropriate batch runner based on num_workers config."""
    sim_device = agent.simulation_device
    num_concurrent = min(run_cfg.episodes, run_cfg.num_concurrent)

    if run_cfg.num_workers > 1:
        from src.ppo.mp_batch_runner import MultiProcessBatchRunner

        # Extract snapshot data from the pool for serialization to workers
        snapshot_state_dicts = None
        sp_ratio = current_self_play_ratio if current_self_play_ratio is not None else run_cfg.self_play_ratio
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
            self_play_ratio=sp_ratio,
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
    # Infer input and output size from the trunk and actor head
    try:
        trunk_first = agent.model.trunk[0]
        actor_last = agent.model.actor_head[-1]
        input_size = trunk_first.in_features if hasattr(trunk_first, 'in_features') else "Unknown"
        output_size = actor_last.out_features if hasattr(actor_last, 'out_features') else "Unknown"
    except Exception:
        input_size = output_size = "Unknown"
    print(f"Model has {num_params / 1_000_000:.2f}M parameters. Trunk input size: {input_size}, action dim: {output_size}.")

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
        sched_info = ""
        if run_cfg.self_play_schedule != "constant":
            sched_info = (f", schedule: {run_cfg.self_play_schedule} "
                          f"{run_cfg.self_play_ratio_start:.2f}->{run_cfg.self_play_ratio:.2f}")
        else:
            sched_info = f", ratio: {run_cfg.self_play_ratio}"
        pool_msg += f" + self-play ({sched_info.lstrip(', ')}{pfsp_info})"
    print(pool_msg)

    total_time_spent_on_updates = 0.0
    total_time_spent_on_episodes = 0.0
    total_time_spent_on_eval = 0.0
    overall_start_time = time.perf_counter()

    sim_device = str(agent.simulation_device)

    for upd in range(1, run_cfg.updates + 1):
        # Compute the self-play ratio for this update based on the schedule
        progress = (upd - 1) / max(1, run_cfg.updates - 1)
        current_sp_ratio = _compute_self_play_ratio(
            run_cfg.self_play_ratio_start,
            run_cfg.self_play_ratio,
            run_cfg.self_play_schedule,
            progress,
        )
        pool.self_play_ratio = current_sp_ratio

        make_opponent = pool.make_factory(card_names, device=sim_device, registry=registry)
        snap_msg = " + snapshots" if pool.has_snapshots else ""
        sp_ratio_msg = f", sp_ratio: {current_sp_ratio:.2f}" if run_cfg.self_play else ""
        print(f"Starting update {upd}/{run_cfg.updates} "
              f"(opponents: {', '.join(opp_types)}{snap_msg}{sp_ratio_msg})")

        # Collect trajectories
        start_time = time.time()
        action_dim = get_action_space_size(card_names)
        runner = _make_runner(
            agent, cards, card_names, action_dim,
            ppo_cfg, run_cfg, data_cfg, make_opponent, registry, pool=pool,
            current_self_play_ratio=current_sp_ratio,
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

        # --- Device boundary: sim_device -> main_device ---
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
        print(f"Card Emb grad: {agent.model.card_emb.weight.grad is not None and agent.model.card_emb.weight.grad.norm().item():.4f}")

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

    Delegates to :func:`src.ppo.ppo_eval.evaluate` for the actual game execution.

    Returns the total wins across all opponent types.
    """
    from src.ppo.ppo_eval import evaluate

    opponents_spec = ",".join(pool.opponent_types)
    result = evaluate(
        agent.model,
        data_cfg=data_cfg or DataConfig(),
        device=str(agent.simulation_device),
        opponents=opponents_spec,
        eval_games=run_cfg.eval_games,
        num_concurrent=run_cfg.num_concurrent,
        num_workers=run_cfg.num_workers if data_cfg is not None else 1,
        ppo_config=ppo_cfg,
        label=f"update {upd}",
        min_games_per_opponent=1,
    )
    return result.total_wins