from datetime import datetime
from pathlib import Path
import math
import torch
import torch.optim.lr_scheduler as lr_sched
import time
import copy

from src.config import (
    DataConfig, PPOConfig, RunConfig, DeviceConfig, ModelConfig,
    save_checkpoint, load_checkpoint,
)
from src.cards.card import Card
from src.ai.agent import Agent
from src.ai.ppo_agent import PPOAgent
from src.encoding.action_encoder import get_action_space_size
from src.ppo.batch_runner import BatchRunner
from src.ppo.opponent_pool import OpponentPool
from src.ppo.train_logger import setup_training_logger, MetricsWriter, format_ppo_metrics
from src.utils.logger import set_disabled, set_verbose


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
    num_concurrent = run_cfg.num_concurrent or run_cfg.episodes // max(1, run_cfg.num_workers)
    num_concurrent = min(run_cfg.episodes, num_concurrent)

    if run_cfg.num_workers > 1:
        from src.ppo.mp_batch_runner import MultiProcessBatchRunner

        # Extract snapshot data from the pool for serialization to workers
        snapshot_state_dicts = None
        sp_ratio = current_self_play_ratio if current_self_play_ratio is not None else run_cfg.self_play_ratio
        pfsp_weights = None
        if pool is not None and pool.has_snapshots:
            snapshot_state_dicts = list(pool._snapshots)
            snap_names = [n for n, _, _ in snapshot_state_dicts]
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
    model_config: ModelConfig | None = None,
):
    """Run PPO training with the given configuration."""
    # --- Logging setup ---
    logger = setup_training_logger("logs")
    metrics_writer = MetricsWriter("logs")
    logger.info(f"Metrics file: {metrics_writer.path}")

    if load_latest and not model_path:
        import glob, os
        model_files = glob.glob(os.path.join(data_cfg.models_dir, "ppo_agent_*.pth"))
        if model_files:
            model_files.sort(key=os.path.getmtime, reverse=True)
            model_path = model_files[0]
            logger.info(f"Auto-loading latest PPO model: {model_path}")
        else:
            logger.info("No PPO model found in models directory to auto-load.")

    # Resume mode: load full training state from checkpoint. Takes precedence
    # over --model-path / --load-latest-model since it implies model_path too.
    resume_path = run_cfg.resume
    resume_ckpt: dict | None = None
    start_update = 1
    if resume_path:
        resume_ckpt = load_checkpoint(resume_path, map_location="cpu")
        prev_update = int(resume_ckpt.get("update") or 0)
        start_update = prev_update + 1
        # Use resumed checkpoint's saved model_config so the rebuilt model
        # matches the stored weights regardless of CLI defaults.
        saved_mcfg = resume_ckpt.get("config", {}).get("model")
        if saved_mcfg:
            model_config = ModelConfig.from_dict(saved_mcfg)
        model_path = resume_path  # PPOAgent loads weights from here
        logger.info(
            f"Resuming from {resume_path}: prev_update={prev_update}, "
            f"will run updates {start_update}..{start_update + run_cfg.updates - 1}"
        )

    set_verbose(False)
    set_disabled(True)
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    card_names = registry.card_names

    agent    = PPOAgent("PPO", card_names,
                       ppo_config=ppo_cfg,
                       model_config=model_config,
                       device_config=dev_cfg,
                       model_path=model_path,
                       registry=registry,
                       load_optimizer_state=resume_ckpt is not None)
    # Log the parameter size of the model
    num_params = sum(p.numel() for p in agent.model.parameters() if p.requires_grad)
    actor_type = getattr(agent.model, 'actor_type', 'mlp')
    pool_type = getattr(agent.model, 'pool_type', 'sum')
    logger.info(f"Model has {num_params / 1_000_000:.2f}M parameters (actor: {actor_type}, pool: {pool_type}). "
                f"Action dim: {agent.action_dim}.")

    # Set up the opponent pool from config
    pool = OpponentPool(
        opponent_spec=run_cfg.opponents,
        self_play_ratio=run_cfg.self_play_ratio,
        pfsp_mode=run_cfg.pfsp_mode,
    )
    if resume_ckpt is not None and run_cfg.self_play:
        manifest = resume_ckpt.get("pool_manifest")
        if manifest:
            loaded = pool.load_from_manifest(
                manifest, models_dir=data_cfg.models_dir, log_fn=logger.warning,
            )
            logger.info(f"Restored {loaded} self-play snapshot(s) from pool manifest.")
        else:
            logger.warning(
                "Resume requested but checkpoint has no pool_manifest; "
                "self-play snapshot pool starts empty."
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
    logger.info(pool_msg)

    # Set up LR scheduler for cosine annealing. T_max spans the entire
    # planned training horizon — when resuming, that's prev_update + this
    # run's updates so the final update lands at exactly lr_end.
    scheduler = None
    if ppo_cfg.lr_schedule == "cosine":
        total_horizon = (start_update - 1) + run_cfg.updates
        scheduler = lr_sched.CosineAnnealingLR(
            agent.optimizer,
            T_max=max(1, total_horizon - 1),
            eta_min=ppo_cfg.lr_end,
        )
        if resume_ckpt is not None:
            sched_state = resume_ckpt.get("scheduler_state_dict")
            if sched_state is not None:
                scheduler.load_state_dict(sched_state)
                logger.info(
                    f"Restored LR scheduler state (last_epoch={scheduler.last_epoch})."
                )
            else:
                # Manually fast-forward the scheduler to the resumed step so
                # the LR curve continues smoothly even for old checkpoints.
                for _ in range(start_update - 1):
                    scheduler.step()
                logger.info(
                    f"No scheduler state in checkpoint; fast-forwarded "
                    f"{start_update - 1} step(s) to align cosine schedule."
                )
        logger.info(f"LR schedule: cosine {ppo_cfg.lr:.1e} -> {ppo_cfg.lr_end:.1e} "
                     f"over {total_horizon} updates")
    else:
        logger.info(f"LR schedule: constant {ppo_cfg.lr:.1e}")

    total_time_spent_on_updates = 0.0
    total_time_spent_on_episodes = 0.0
    total_time_spent_on_eval = 0.0
    overall_start_time = time.perf_counter()

    sim_device = str(agent.simulation_device)

    end_update = start_update + run_cfg.updates - 1
    total_horizon = end_update  # used for self-play progress and final-update detection
    for upd in range(start_update, end_update + 1):
        # Compute the self-play ratio for this update based on the schedule.
        # Progress is measured against the full horizon (including resumed
        # updates) so the curve continues from where it left off.
        progress = (upd - 1) / max(1, total_horizon - 1)
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
        logger.info(f"--- Update {upd}/{end_update} "
                     f"(opponents: {', '.join(opp_types)}{snap_msg}{sp_ratio_msg}) ---")

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

        # Derive rollout stats from available data
        num_samples = states.shape[0]
        avg_decisions_per_ep = num_samples / max(1, run_cfg.episodes)
        opp_results = runner.opponent_results
        total_wins = sum(v[0] for v in opp_results.values())
        total_games = sum(v[1] for v in opp_results.values())
        rollout_win_rate = total_wins / max(1, total_games)
        throughput = run_cfg.episodes / max(0.001, duration_episodes)

        workers_msg = f" ({run_cfg.num_workers} workers)" if run_cfg.num_workers > 1 else ""
        logger.info(f"Rollout: {run_cfg.episodes} episodes, {num_samples} steps in "
                     f"{duration_episodes:.1f}s ({throughput:.1f} ep/s){workers_msg}  "
                     f"win_rate={rollout_win_rate:.0%}  avg_decisions/ep={avg_decisions_per_ep:.0f}")

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
                logger.info(f"  PFSP: {', '.join(parts)}")

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
        current_lr = agent.optimizer.param_groups[0]['lr']
        ppo_metrics = agent.update(states, actions, old_lp, returns, advs, masks)
        duration_update = time.time() - start_time
        total_time_spent_on_updates += duration_update

        # Log PPO metrics summary
        logger.info(f"PPO: {format_ppo_metrics(ppo_metrics)}  "
                     f"lr={current_lr:.2e}  {duration_update:.1f}s  "
                     f"samples={num_samples}")

        # Card embedding gradient norm (specific architecture diagnostic)
        card_emb_grad = (agent.model.card_emb.weight.grad is not None
                         and agent.model.card_emb.weight.grad.norm().item())
        logger.debug(f"Card Emb grad: {card_emb_grad:.4f}")

        # NaN guard: abort training immediately if loss diverged
        if ppo_metrics.get("nan_detected"):
            logger.error("NaN/Inf detected in loss — aborting training.")
            break

        # Step LR scheduler after each update
        if scheduler is not None:
            scheduler.step()

        # Add snapshot for self-play after each update
        if run_cfg.self_play:
            snapshot_sd = copy.deepcopy(agent.model).cpu().state_dict()
            pool.add_snapshot(snapshot_sd, f"PPO_{upd}",
                              model_config=agent.model_config)

        # Evaluate performance (every N updates, and always on the last)
        is_last_update = upd == end_update
        wins = -1  # default: no eval this round
        eval_result = None
        if upd % run_cfg.eval_every == 0 or is_last_update:
            start_time = time.time()
            wins, eval_result = _run_per_opponent_eval(
                agent, pool, card_names, cards, run_cfg, ppo_cfg, upd,
                registry=registry, data_cfg=data_cfg,
                log_fn=logger.info,
            )
            agent.clear_buffers()
            duration_eval = time.time() - start_time
            total_time_spent_on_eval += duration_eval
        else:
            next_eval = upd + run_cfg.eval_every - upd % run_cfg.eval_every
            logger.debug(f"Skipping eval (next eval at update {next_eval}).")

        # Save checkpoint per update. Includes optimizer + scheduler state
        # and the snapshot pool manifest so this checkpoint is sufficient
        # to fully resume training via --resume.
        ts = datetime.now().strftime("%m%d_%H%M")
        Path(data_cfg.models_dir).mkdir(exist_ok=True)
        ckpt_path = f"{data_cfg.models_dir}/ppo_agent_{ts}_upd{upd}_wins{wins}.pth"
        # Pre-record this checkpoint's path on the snapshot we just added so
        # the manifest written into the same checkpoint references itself
        # (which lets a future resume immediately have one snapshot to fight).
        if run_cfg.self_play:
            pool.set_snapshot_path(f"PPO_{upd}", ckpt_path)
        save_checkpoint(
            ckpt_path,
            agent.model.state_dict(),
            ppo_config=ppo_cfg,
            model_config=agent.model_config,
            run_config=run_cfg,
            device_config=dev_cfg,
            update=upd,
            optimizer_state_dict=agent.optimizer.state_dict(),
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
            pool_manifest=pool.to_manifest() if run_cfg.self_play else None,
        )
        logger.info(f"Checkpoint saved: {ckpt_path}")

        # Write JSONL metrics row for this update
        row: dict = {
            "update": upd,
            "timestamp": datetime.now().isoformat(),
            "lr": current_lr,
            "self_play_ratio": current_sp_ratio if run_cfg.self_play else None,
            "rollout": {
                "episodes": run_cfg.episodes,
                "num_samples": num_samples,
                "duration_s": round(duration_episodes, 2),
                "throughput_eps": round(throughput, 2),
                "win_rate": round(rollout_win_rate, 4),
                "avg_decisions_per_ep": round(avg_decisions_per_ep, 1),
                "per_opponent": {k: {"wins": v[0], "games": v[1]} for k, v in opp_results.items()},
            },
            "ppo": {k: round(v, 6) if isinstance(v, float) else v for k, v in ppo_metrics.items()},
            "ppo_duration_s": round(duration_update, 2),
            "card_emb_grad": round(card_emb_grad, 6) if card_emb_grad else None,
        }
        if eval_result is not None:
            row["eval"] = {
                "overall_win_rate": round(eval_result.overall_win_rate, 4),
                "total_wins": eval_result.total_wins,
                "total_games": eval_result.total_games,
                "per_opponent": {
                    r.opponent: {"wins": r.wins, "games": r.games, "win_rate": round(r.win_rate, 4)}
                    for r in eval_result.per_opponent
                },
            }
        metrics_writer.write(row)

    # --- Training complete ---
    logger.info(f"Episodes:    {total_time_spent_on_episodes:.1f}s total, "
                f"{total_time_spent_on_episodes / max(1, run_cfg.updates):.1f}s avg/update")
    logger.info(f"PPO updates: {total_time_spent_on_updates:.1f}s total, "
                f"{total_time_spent_on_updates / max(1, run_cfg.updates):.1f}s avg/update")
    logger.info(f"Evaluation:  {total_time_spent_on_eval:.1f}s total, "
                f"{total_time_spent_on_eval / max(1, run_cfg.updates):.1f}s avg/update")
    avg_decision_time = agent.get_average_decision_time()
    logger.info(f"Avg PPO decision time: {avg_decision_time:.6f}s/decision")
    logger.info(f"Total wall time: {time.perf_counter() - overall_start_time:.1f}s")
    logger.info("Training complete.")
    metrics_writer.close()

    # Visibility hook for the shutdown-hang watchdog — if anything shows up
    # here, it's a process we failed to reap and the interpreter may block
    # at atexit on its queues. See RCA 2026-04-17.
    import multiprocessing as _mp
    _survivors = _mp.active_children()
    if _survivors:
        logger.warning(
            "Lingering child processes at shutdown: %s",
            [(p.name, p.pid) for p in _survivors],
        )


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
    log_fn=None,
) -> tuple[int, 'EvalResult']:
    """Run evaluation against each opponent type separately, log per-type results.

    Delegates to :func:`src.ppo.ppo_eval.evaluate` for the actual game execution.

    Returns a tuple of (total_wins, EvalResult).
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
        log_fn=log_fn,
    )
    return result.total_wins, result