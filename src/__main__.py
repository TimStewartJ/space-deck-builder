"""Unified CLI entrypoint for the Star Realms AI project.

Usage:
    python -m src train [options]
    python -m src simulate [options]
    python -m src benchmark [options]
    python -m src elo [options]
"""
import argparse
import sys


def _add_common_args(parser: argparse.ArgumentParser):
    """Add args shared across subcommands."""
    from src.config import DataConfig
    parser.add_argument("--cards-path", type=str, default=DataConfig().cards_path)


def _build_train_parser(sub: argparse._SubParsersAction):
    from src.config import PPOConfig, RunConfig, DeviceConfig, ModelConfig
    _ppo = PPOConfig()
    _run = RunConfig()
    _dev = DeviceConfig()
    _mdl = ModelConfig()

    p = sub.add_parser("train", help="Run PPO training")
    _add_common_args(p)
    # PPO hyperparameters (defaults sourced from PPOConfig dataclass)
    p.add_argument("--lr",          type=float, default=_ppo.lr)
    p.add_argument("--lr-end",     type=float, default=_ppo.lr_end,
                   help="Terminal learning rate for cosine schedule")
    p.add_argument("--lr-schedule", type=str, default=_ppo.lr_schedule,
                   choices=["constant", "cosine"],
                   help="LR schedule: cosine annealing (default) or constant")
    p.add_argument("--gamma",       type=float, default=_ppo.gamma)
    p.add_argument("--lam",         type=float, default=_ppo.lam)
    p.add_argument("--clip-eps",    type=float, default=_ppo.clip_eps)
    p.add_argument("--epochs",      type=int,   default=_ppo.epochs)
    p.add_argument("--batch-size",  type=int,   default=_ppo.batch_size)
    p.add_argument("--entropy",     type=float, default=_ppo.entropy_coef,
                   help="Entropy bonus coefficient")
    p.add_argument("--adv-norm",    type=str,   default=_ppo.adv_norm,
                   choices=["per_episode", "global"],
                   help="Advantage normalization mode")
    # Model architecture
    p.add_argument("--actor-type",  type=str, default=_mdl.actor_type,
                   choices=["mlp", "attention"],
                   help="Actor head type: mlp (flat linear) or attention (query-key dot product)")
    p.add_argument("--pool-type",   type=str, default=_mdl.pool_type,
                   choices=["sum", "attention"],
                   help="Zone feature pooling: sum (presence-weighted) or attention (learned per-zone query)")
    # Run topology (defaults sourced from RunConfig dataclass)
    p.add_argument("--episodes",    type=int,   default=_run.episodes)
    p.add_argument("--updates",     type=int,   default=_run.updates)
    p.add_argument("--eval-every",  type=int,   default=_run.eval_every,
                   help="Evaluate every N updates (always on last)")
    p.add_argument("--eval-games",  type=int,   default=_run.eval_games)
    p.add_argument("--self-play",   action="store_true")
    p.add_argument("--opponents",   type=str, default=_run.opponents,
                   help="Opponent mix: 'random,heuristic' or 'random:0.6,heuristic:0.4'")
    p.add_argument("--self-play-ratio", type=float, default=_run.self_play_ratio,
                   help="Final self-play ratio (or constant when schedule=constant)")
    p.add_argument("--self-play-ratio-start", type=float, default=_run.self_play_ratio_start,
                   help="Initial self-play ratio at the start of training (used with schedule)")
    p.add_argument("--self-play-schedule", type=str, default=_run.self_play_schedule,
                   choices=["constant", "linear", "cosine"],
                   help="Self-play ratio schedule: linear (default), constant, or cosine ramp")
    p.add_argument("--num-workers", type=int, default=_run.num_workers,
                   help=f"Simulation worker processes (1=single-process, >1=multi-process, default: {_run.num_workers})")
    p.add_argument("--num-concurrent", type=int, default=None,
                   help="Concurrent games per worker (default: episodes/workers)")
    p.add_argument("--pfsp", type=str, default=_run.pfsp_mode,
                   choices=["uniform", "hard", "variance"],
                   help="PFSP snapshot weighting: uniform (default), hard, or variance")
    # Devices (defaults sourced from DeviceConfig dataclass)
    p.add_argument("--main-device",       type=str, default=_dev.main_device,
                   help="Device for training updates")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for episode simulation")
    # Model loading
    p.add_argument("--model-path",        type=str, default=None,
                   help="Path to a pretrained PPO model (weights only)")
    p.add_argument("--load-latest-model", action="store_true",
                   help="Auto-load the latest model from models/ (weights only)")
    p.add_argument("--resume",            type=str, default=None,
                   help="Resume training from a checkpoint: restores weights, "
                        "optimizer, LR scheduler, snapshot pool, and update counter. "
                        "--updates is interpreted as additional updates beyond the resumed step.")
    p.add_argument("--lr-horizon",        type=int, default=None,
                   help="Override the cosine LR scheduler horizon (final update). "
                        "Use this when chunking a multi-process resume so each chunk "
                        "re-pins T_max to the same final-target update (e.g. 200) "
                        "and the cosine LR curve flows smoothly across chunks "
                        "instead of decaying to the floor inside each one.")
    return p


def _build_simulate_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig, SimConfig
    _dev = DeviceConfig()
    _sim = SimConfig()

    p = sub.add_parser("simulate", help="Run PPO vs opponent simulation")
    _add_common_args(p)
    p.add_argument("--model1", type=str, default=None,
                   help="PPO model for player 1 (default: latest)")
    p.add_argument("--model2", type=str, default=None,
                   help="PPO model for player 2")
    p.add_argument("--player2-random", action="store_true", default=True,
                   help="Use random agent for player 2")
    p.add_argument("--games", type=int, default=_sim.games)
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    return p


def _build_benchmark_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig
    _dev = DeviceConfig()

    p = sub.add_parser("benchmark", help="Benchmark training throughput")
    _add_common_args(p)
    p.add_argument("--episodes", type=int, default=128)
    p.add_argument("--mode", type=str, default="both",
                   choices=["sequential", "batched", "parallel", "both"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    return p


def _build_elo_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig, RunConfig
    _dev = DeviceConfig()
    _run = RunConfig()

    p = sub.add_parser("elo", help="Run Elo tournament between checkpoints and/or agents")
    _add_common_args(p)
    p.add_argument("--checkpoints", type=str, nargs="*", default=[],
                   help="Checkpoint glob patterns or paths "
                        "(e.g. 'models/ppo_agent_0415_*upd*0.pth')")
    p.add_argument("--agents", type=str, default=None,
                   help="Comma-separated built-in agent types to include "
                        "(e.g. 'random,heuristic,simple')")
    p.add_argument("--games-per-pair", type=int, default=1000,
                   help="Games to play per pairing (default: 1000)")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    p.add_argument("--num-concurrent", type=int, default=None,
                   help="Concurrent games per worker (default: games-per-pair)")
    p.add_argument("--num-workers", type=int, default=_run.num_workers,
                   help=f"Worker processes for PPO pairings (default: {_run.num_workers})")
    p.add_argument("--analyze", action="store_true", default=False,
                   help="Collect replays and generate comparative dashboard")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory for replay files (default: analysis/elo_<timestamp>/)")
    return p


def _build_eval_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig, RunConfig
    _dev = DeviceConfig()
    _run = RunConfig()

    p = sub.add_parser("eval", help="Evaluate a trained model against fixed opponents")
    _add_common_args(p)
    p.add_argument("--model", type=str, default=None,
                   help="Model checkpoint path (default: latest)")
    p.add_argument("--load-latest", action="store_true",
                   help="Auto-load the latest model from models/")
    p.add_argument("--opponents", type=str, default="random,heuristic,simple",
                   help="Comma-separated opponent types (default: random,heuristic,simple)")
    p.add_argument("--games", type=int, default=_run.eval_games,
                   help=f"Total evaluation games (distributed across opponent types, default: {_run.eval_games})")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    p.add_argument("--num-concurrent", type=int, default=None,
                   help="Concurrent games per worker (default: games/workers)")
    p.add_argument("--num-workers", type=int, default=_run.num_workers,
                   help=f"Simulation worker processes (default: {_run.num_workers})")


def _build_analyze_parser(sub: argparse._SubParsersAction):
    from src.config import DeviceConfig, RunConfig
    _dev = DeviceConfig()
    _run = RunConfig()

    p = sub.add_parser("analyze", help="Collect replays and analyze agent behavior")
    _add_common_args(p)
    p.add_argument("--model", type=str, default=None,
                   help="Model checkpoint path (default: latest)")
    p.add_argument("--games", type=int, default=200,
                   help="Number of games to collect (default: 200)")
    p.add_argument("--opponents", type=str, default="random,heuristic,simple",
                   help="Opponent types: random,heuristic,simple,self (default: random,heuristic,simple)")
    p.add_argument("--opponent-model", type=str, default=None,
                   help="Path to a model checkpoint to use as opponent (overrides --opponents)")
    p.add_argument("--output", type=str, default=None,
                   help="Output replay file path (default: analysis/replays_<timestamp>.json.gz)")
    p.add_argument("--replay", type=str, default=None,
                   help="Path to existing replay file to analyze (skips game collection)")
    p.add_argument("--simulation-device", type=str, default=_dev.simulation_device,
                   help="Device for inference (cuda or cpu)")
    p.add_argument("--num-concurrent", type=int, default=None,
                   help="Concurrent games per worker (default: games/workers)")
    p.add_argument("--dashboard", action="store_true", default=False,
                   help="Generate interactive HTML dashboard (in addition to PNGs)")
    return p


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="space-deck-builder",
        description="Star Realms AI — training, simulation, and benchmarking",
    )
    sub = parser.add_subparsers(dest="command")
    _build_train_parser(sub)
    _build_simulate_parser(sub)
    _build_benchmark_parser(sub)
    _build_elo_parser(sub)
    _build_eval_parser(sub)
    _build_analyze_parser(sub)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "train":
        _run_train(args)
    elif args.command == "simulate":
        _run_simulate(args)
    elif args.command == "benchmark":
        _run_benchmark(args)
    elif args.command == "elo":
        _run_elo(args)
    elif args.command == "eval":
        _run_eval(args)
    elif args.command == "analyze":
        _run_analyze(args)


def _run_train(args):
    """Construct configs from CLI args and delegate to the trainer."""
    from src.config import DataConfig, PPOConfig, RunConfig, DeviceConfig, ModelConfig
    from src.ppo.ppo_trainer import train

    data_cfg = DataConfig(cards_path=args.cards_path)
    ppo_cfg = PPOConfig(
        lr=args.lr, gamma=args.gamma, lam=args.lam,
        clip_eps=args.clip_eps, epochs=args.epochs,
        batch_size=args.batch_size, entropy_coef=args.entropy,
        adv_norm=args.adv_norm,
        lr_end=args.lr_end, lr_schedule=args.lr_schedule,
    )
    model_cfg = ModelConfig(actor_type=args.actor_type, pool_type=args.pool_type)
    run_cfg = RunConfig(
        episodes=args.episodes, updates=args.updates,
        eval_every=args.eval_every, eval_games=args.eval_games,
        self_play=args.self_play,
        opponents=args.opponents,
        self_play_ratio=args.self_play_ratio,
        self_play_ratio_start=args.self_play_ratio_start,
        self_play_schedule=args.self_play_schedule,
        pfsp_mode=args.pfsp,
        num_workers=args.num_workers,
        num_concurrent=args.num_concurrent,
        resume=args.resume,
        lr_horizon=args.lr_horizon,
    )
    dev_cfg = DeviceConfig(
        main_device=args.main_device,
        simulation_device=args.simulation_device,
    )
    train(data_cfg, ppo_cfg, run_cfg, dev_cfg,
          model_path=args.model_path,
          load_latest=args.load_latest_model,
          model_config=model_cfg)


def _run_simulate(args):
    """Construct configs and delegate to the simulator."""
    from src.config import DataConfig, SimConfig
    from src.ppo.ppo_simulate import simulate

    data_cfg = DataConfig(cards_path=args.cards_path)
    sim_cfg = SimConfig(games=args.games, player2_random=args.player2_random)
    simulate(data_cfg, sim_cfg,
             model1_path=args.model1,
             model2_path=args.model2,
             device=args.simulation_device)


def _run_benchmark(args):
    """Delegate to the benchmark function."""
    from scripts.benchmark import benchmark
    benchmark(
        num_episodes=args.episodes,
        device=args.simulation_device,
        mode=args.mode,
        workers=args.workers,
    )


def _run_elo(args):
    """Resolve checkpoint paths, build config, and run the Elo tournament."""
    from src.config import DataConfig
    from src.ppo.elo_tournament import run_tournament, resolve_checkpoint_paths

    paths = resolve_checkpoint_paths(args.checkpoints)
    agent_types = [a.strip() for a in args.agents.split(",") if a.strip()] if args.agents else []

    # Validate agent types early for clear CLI error messages
    from src.ppo.elo_tournament import BUILTIN_AGENT_TYPES
    invalid = [a for a in agent_types if a not in BUILTIN_AGENT_TYPES]
    if invalid:
        print(f"Error: unknown agent type(s): {', '.join(invalid)}. "
              f"Valid types: {', '.join(sorted(BUILTIN_AGENT_TYPES))}")
        sys.exit(1)

    total_participants = len(paths) + len(agent_types)
    if total_participants < 2:
        print(f"Error: need at least 2 participants, found {total_participants} "
              f"({len(paths)} checkpoints, {len(agent_types)} agents)")
        sys.exit(1)

    data_cfg = DataConfig(cards_path=args.cards_path)
    num_concurrent = args.num_concurrent or args.games_per_pair
    run_tournament(
        checkpoint_paths=paths,
        data_cfg=data_cfg,
        games_per_pair=args.games_per_pair,
        device=args.simulation_device,
        num_concurrent=num_concurrent,
        agent_types=agent_types or None,
        num_workers=args.num_workers,
        collect_replays=args.analyze,
        replay_output_dir=getattr(args, 'output_dir', None),
    )


def _run_eval(args):
    """Load a model and evaluate it against fixed opponent types."""
    from src.config import DataConfig
    from src.ppo.ppo_eval import load_and_evaluate

    data_cfg = DataConfig(cards_path=args.cards_path)
    try:
        load_and_evaluate(
            model_path=args.model,
            load_latest=args.load_latest or args.model is None,
            data_cfg=data_cfg,
            device=args.simulation_device,
            opponents=args.opponents,
            eval_games=args.games,
            num_concurrent=args.num_concurrent,
            num_workers=args.num_workers,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _run_analyze(args):
    """Collect replays from games and/or analyze existing replay data."""
    from src.analysis.analyzer import analyze_replays

    # If a replay file is provided, skip collection and just analyze
    if args.replay:
        print(f"Analyzing existing replay file: {args.replay}")
        analyze_replays(args.replay)
        if args.dashboard:
            from src.analysis.dashboard import generate_dashboard
            dash_path = generate_dashboard(args.replay)
            print(f"Dashboard saved to: {dash_path}")
        return

    # Otherwise, collect replays by playing games
    import os
    import time
    import torch
    from datetime import datetime
    from src.config import DataConfig, DeviceConfig
    from src.ppo.ppo_simulate import get_latest_model
    from src.ppo.batch_runner import BatchRunner
    from src.ppo.ppo_actor_critic import PPOActorCritic
    from src.encoding.state_encoder import get_state_size
    from src.encoding.action_encoder import get_action_space_size
    from src.config import load_checkpoint
    from src.analysis.replay_collector import ReplayCollector
    from src.ppo.opponent_pool import OpponentPool
    from src.utils.logger import set_verbose, set_disabled

    set_verbose(False)
    set_disabled(True)

    data_cfg = DataConfig(cards_path=args.cards_path)
    device = DeviceConfig.resolve(args.simulation_device)

    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    card_names = registry.card_names

    model_path = args.model or get_latest_model(data_cfg.models_dir)
    if not model_path:
        print("Error: no model found. Provide --model or train one first.")
        sys.exit(1)

    print(f"Loading model: {model_path}")
    ckpt = load_checkpoint(model_path, map_location=device)
    saved_cfg = ckpt.get("config", {}).get("model")
    from src.config import ModelConfig
    model_config = ModelConfig.from_dict(saved_cfg) if saved_cfg else ModelConfig()

    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)

    model = PPOActorCritic(
        state_dim, action_dim, len(card_names), model_config=model_config
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Distribute games across opponent types, spreading the remainder
    opponent_types = [o.strip().split(":")[0] for o in args.opponents.split(",")]

    # --opponent-model overrides --opponents with a single model opponent
    if args.opponent_model:
        opponent_types = ["ppo-opponent"]

    num_types = len(opponent_types)
    base_games = args.games // num_types
    remainder = args.games % num_types
    games_schedule = [base_games + (1 if i < remainder else 0) for i in range(num_types)]

    collector = ReplayCollector(card_names, action_dim)
    total_wins = 0
    total_games = 0

    # Build opponent factories — model-based types use a shared model to
    # avoid per-game GPU allocations; fixed agent types go through OpponentPool.
    from src.ppo.elo_tournament import _make_opponent_factory

    # Pre-load opponent model if --opponent-model was given
    opp_model_sd = None
    opp_model_config = None
    if args.opponent_model:
        print(f"Loading opponent model: {args.opponent_model}")
        opp_ckpt = load_checkpoint(args.opponent_model, map_location=device)
        opp_model_sd = opp_ckpt["model_state_dict"]
        saved_opp_cfg = opp_ckpt.get("config", {}).get("model")
        opp_model_config = ModelConfig.from_dict(saved_opp_cfg) if saved_opp_cfg else model_config

    # Fixed agent types go through OpponentPool
    fixed_types = [t for t in opponent_types if t not in ("self", "ppo-opponent")]
    pool = OpponentPool(opponent_spec=",".join(fixed_types)) if fixed_types else None

    for opp_type, num_games in zip(opponent_types, games_schedule):
        if num_games == 0:
            continue
        print(f"Collecting {num_games} games vs {opp_type}...")

        if opp_type == "self":
            factory = _make_opponent_factory(
                ckpt["model_state_dict"], card_names, device,
                model_config=model_config,
            )
        elif opp_type == "ppo-opponent":
            factory = _make_opponent_factory(
                opp_model_sd, card_names, device,
                model_config=opp_model_config,
            )
        else:
            factory = pool.make_factory_for_type(opp_type, card_names, device=device)

        runner = BatchRunner(
            model=model,
            card_names=card_names,
            cards=cards,
            action_dim=action_dim,
            device=torch.device(device),
            opponent_factory=factory,
            num_concurrent=min(args.num_concurrent or args.games, num_games),
            registry=registry,
        )

        start = time.time()
        w, l, s = runner.run_analysis(num_games, collector, opp_type)
        elapsed = time.time() - start
        total_wins += w
        total_games += num_games
        print(f"  {w}/{num_games} wins ({w / num_games:.0%}) in {elapsed:.1f}s")

    print(f"\nTotal: {total_wins}/{total_games} wins "
          f"({total_wins / total_games:.0%}), "
          f"{len(collector.replays)} games collected, "
          f"{sum(len(r.decisions) for r in collector.replays):,} decisions")

    # Save replays
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"analysis/replays_{ts}.json.gz"
    os.makedirs(os.path.dirname(output_path) or "analysis", exist_ok=True)
    collector.save(output_path)
    print(f"Replays saved to: {output_path}")

    # Analyze
    analyze_replays(output_path, output_dir=os.path.dirname(output_path) or "analysis")

    # Generate interactive dashboard if requested
    if args.dashboard:
        from src.analysis.dashboard import generate_dashboard
        dash_path = generate_dashboard(
            output_path,
            model_info=f"Model: {model_path}",
        )
        print(f"Dashboard saved to: {dash_path}")


if __name__ == "__main__":
    main()
