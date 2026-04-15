"""Simulation worker process for multi-process training.

Each worker runs game simulation and state encoding on CPU, sending
encoded states to the InferenceServer for GPU inference via
multiprocessing Queues. Workers merge their own rollout data locally
and return a single pre-merged result to the main process.
"""
import random
import traceback
from typing import Optional

import numpy as np
import torch
import multiprocessing as mp

from src.config import DataConfig, PPOConfig
from src.engine.game import Game
from src.engine.actions import ActionType
from src.ai.agent import Agent
from src.ai.random_agent import RandomAgent
from src.ai.simple_agent import SimpleAgent
from src.ai.heuristic_agent import HeuristicAgent
from src.cards.card import Card
from src.cards.registry import CardRegistry
from src.encoding.state_encoder import encode_state, get_state_size
from src.encoding.action_encoder import get_action_space_size, END_TURN_INDEX
from src.encoding.action_context import build_action_context, ActionContext
from src.ppo.rollout_buffer import RolloutBuffer
from src.ppo.mp_inference_server import (
    InferenceRequest, InferenceResponse, WorkerResult, WorkerError,
    EvalWorkerResult,
)
from src.utils.logger import set_disabled


# Agent factory registry (mirrors opponent_pool.AGENT_REGISTRY)
_AGENT_FACTORIES = {
    "random":    lambda name: RandomAgent(name),
    "heuristic": lambda name: HeuristicAgent(name),
    "simple":    lambda name: SimpleAgent(name),
}


class _DummyAgent(Agent):
    """Placeholder agent for the PPO player slot."""
    def make_decision(self, game_state):
        raise RuntimeError("DummyAgent.make_decision should never be called")


def _parse_opponent_spec(spec: str) -> list[tuple[str, float]]:
    """Parse opponent spec into (name, weight) pairs."""
    entries = []
    has_weights = ":" in spec
    for part in spec.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if ":" in part:
            name, w = part.split(":", 1)
            entries.append((name.strip(), float(w.strip())))
        else:
            entries.append((part, 0.0))
    if not has_weights:
        eq = 1.0 / len(entries) if entries else 1.0
        entries = [(n, eq) for n, _ in entries]
    total = sum(w for _, w in entries)
    return [(n, w / total) for n, w in entries] if total > 0 else entries


def _build_opponent_factory(opponent_spec: str):
    """Build an opponent factory function from a serializable spec string."""
    entries = _parse_opponent_spec(opponent_spec)
    names = [n for n, _ in entries]
    weights = [w for _, w in entries]

    def factory() -> Agent:
        chosen = random.choices(names, weights=weights, k=1)[0]
        return _AGENT_FACTORIES[chosen](chosen.capitalize())

    return factory


def _advance_non_ppo(game: Game, training_agent_name: str):
    """Advance the game while the current player is NOT the PPO training agent."""
    while not game.is_game_over:
        player = game.current_player
        if player.name == training_agent_name:
            break
        action = player.make_decision(game)
        game.apply_decision(action)


def _start_game(
    cards: list[Card],
    card_index_map: dict[str, int],
    opponent_factory,
    training_agent_name: str = "PPO",
):
    """Initialize a new game. Returns (game, rollout_buffer, opponent_name)."""
    opponent = opponent_factory()
    game = Game(cards, card_index_map=card_index_map)
    game.add_player(training_agent_name, _DummyAgent(training_agent_name))
    game.add_player(opponent.name, opponent)
    game.start_game()
    buf = RolloutBuffer()
    return game, buf, opponent.name


def _finish_game(
    i: int,
    games: list,
    buffers: list,
    game_opponents: list,
    completed_rollouts: list,
    opponent_results: dict,
    ppo_config: PPOConfig,
    training_agent_name: str = "PPO",
):
    """Handle a completed game: record result, compute reward, finalize rollout."""
    winner = games[i].get_winner()
    won = winner == training_agent_name

    opp_name = game_opponents[i]
    if opp_name is not None:
        entry = opponent_results.get(opp_name)
        if entry is None:
            entry = [0, 0]
            opponent_results[opp_name] = entry
        if won:
            entry[0] += 1
        entry[1] += 1

    if buffers[i] is None or len(buffers[i]) == 0:
        games[i] = None
        buffers[i] = None
        game_opponents[i] = None
        return

    reward = 1.0 if won else -1.0
    buffers[i].fill_last_reward(reward)
    # Compute GAE on CPU — no GPU needed
    rollout = buffers[i].finish(
        gamma=ppo_config.gamma,
        lam=ppo_config.lam,
        device=torch.device("cpu"),
        normalize=(ppo_config.adv_norm == "per_episode"),
    )
    completed_rollouts.append(rollout)
    games[i] = None
    buffers[i] = None
    game_opponents[i] = None


def sim_worker_main(
    worker_id: int,
    num_episodes: int,
    num_concurrent: int,
    data_config_dict: dict,
    card_names: list[str],
    card_index_map: dict[str, int],
    action_dim: int,
    state_size: int,
    ppo_config_dict: dict,
    opponent_spec: str,
    seed: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """Entry point for a simulation worker process.

    Runs game simulation and state encoding on CPU. Sends encoded states
    to the InferenceServer for GPU inference. Merges rollout data locally
    and returns a single pre-merged result via result_queue.

    card_names and card_index_map are passed from the parent process to
    guarantee encoding consistency across all workers.
    """
    try:
        _sim_worker_inner(
            worker_id, num_episodes, num_concurrent, data_config_dict,
            card_names, card_index_map, action_dim, state_size,
            ppo_config_dict, opponent_spec, seed,
            request_queue, response_queue, result_queue,
        )
    except Exception:
        result_queue.put(WorkerError(
            worker_id=worker_id,
            error=traceback.format_exc(),
        ))


def _sim_worker_inner(
    worker_id, num_episodes, num_concurrent, data_config_dict,
    card_names, card_index_map, action_dim, state_size,
    ppo_config_dict, opponent_spec, seed,
    request_queue, response_queue, result_queue,
):
    """Core worker logic, separated for clean error wrapping."""
    # Worker initialization
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.set_num_threads(1)  # avoid inter-worker contention
    set_disabled(True)

    ppo_cfg = PPOConfig.from_dict(ppo_config_dict)

    # Load cards independently (avoids pickling Card/Effect objects)
    data_cfg = DataConfig.from_dict(data_config_dict)
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)

    # Verify encoding alignment with parent process
    assert registry.card_names == card_names, (
        f"Worker {worker_id} card_names mismatch: "
        f"expected {len(card_names)} cards, got {len(registry.card_names)}"
    )

    training_agent_name = "PPO"
    opponent_factory = _build_opponent_factory(opponent_spec)

    # Pre-allocate reusable buffers for encoding
    state_buf = np.zeros(state_size, dtype=np.float32)
    mask_buf = np.zeros(action_dim, dtype=bool)

    # Game slots
    games: list[Optional[Game]] = [None] * num_concurrent
    buffers: list[Optional[RolloutBuffer]] = [None] * num_concurrent
    game_opponents: list[Optional[str]] = [None] * num_concurrent

    completed_rollouts: list[tuple] = []
    opponent_results: dict[str, list[int]] = {}
    episodes_started = 0
    episodes_completed = 0
    request_id = 0

    # Fill initial game slots
    active_count = min(num_concurrent, num_episodes)
    for i in range(active_count):
        games[i], buffers[i], game_opponents[i] = _start_game(
            cards, card_index_map, opponent_factory, training_agent_name,
        )
        episodes_started += 1

    while episodes_completed < num_episodes:
        # Step 1: Advance all active games past non-PPO decisions
        for i in range(num_concurrent):
            if games[i] is not None and not games[i].is_game_over:
                _advance_non_ppo(games[i], training_agent_name)

        # Step 2: Handle completed games
        for i in range(num_concurrent):
            if games[i] is None or not games[i].is_game_over:
                continue
            _finish_game(
                i, games, buffers, game_opponents,
                completed_rollouts, opponent_results, ppo_cfg, training_agent_name,
            )
            episodes_completed += 1
            if episodes_started < num_episodes:
                games[i], buffers[i], game_opponents[i] = _start_game(
                    cards, card_index_map, opponent_factory, training_agent_name,
                )
                episodes_started += 1
                _advance_non_ppo(games[i], training_agent_name)
                if games[i].is_game_over:
                    _finish_game(
                        i, games, buffers, game_opponents,
                        completed_rollouts, opponent_results, ppo_cfg,
                        training_agent_name,
                    )
                    episodes_completed += 1
                    games[i] = None
                    buffers[i] = None

        # Step 3: Collect pending PPO decisions
        pending_indices: list[int] = []
        pending_states: list[torch.Tensor] = []
        pending_contexts: list[ActionContext] = []

        for i in range(num_concurrent):
            if games[i] is None or games[i].is_game_over:
                continue
            player = games[i].current_player
            if player.name != training_agent_name:
                continue

            ctx = build_action_context(
                games[i], player, card_index_map,
                action_dim, mask_buf=mask_buf,
            )
            state = encode_state(
                games[i], is_current_player_training=True,
                cards=card_names,
                card_index_map=card_index_map,
                state_buf=state_buf,
                can_buy=ctx.can_buy,
                has_actions=ctx.has_meaningful,
            )

            pending_indices.append(i)
            pending_states.append(state)
            pending_contexts.append(ActionContext(
                mask=ctx.mask.copy(),
                has_meaningful=ctx.has_meaningful,
                can_buy=ctx.can_buy,
                resolvers=ctx.resolvers,
            ))

        if not pending_states:
            continue

        # Step 4: Build numpy batch with END_TURN suppression, send to GPU
        # Use uint8 masks to reduce IPC payload size
        states_np = np.stack([s.numpy() for s in pending_states])
        masks_np = np.zeros((len(pending_states), action_dim), dtype=np.uint8)
        for j, ctx in enumerate(pending_contexts):
            masks_np[j] = ctx.mask.astype(np.uint8)
            if ctx.has_meaningful:
                masks_np[j, END_TURN_INDEX] = 0

        request_queue.put(InferenceRequest(
            worker_id=worker_id,
            request_id=request_id,
            states=states_np,
            masks=masks_np,
        ))
        request_id += 1

        # Step 5: Block waiting for response
        response: InferenceResponse = response_queue.get()

        # Step 6: Distribute actions and record in buffers
        for j, i in enumerate(pending_indices):
            act_idx = int(response.action_indices[j])
            action = pending_contexts[j].resolvers[act_idx]

            buffers[i].add(
                pending_states[j],
                act_idx,
                torch.tensor(response.log_probs[j]),
                torch.tensor(response.values[j]),
                reward=0.0,
                done=False,
                mask=torch.from_numpy(masks_np[j].astype(np.float32)).bool(),
            )

            games[i].apply_decision(action)

    # Merge rollouts locally before sending (reduces IPC overhead)
    if completed_rollouts:
        S, A, OL, R, Adv, M = zip(*completed_rollouts)
        has_masks = all(m is not None for m in M)
        result_queue.put(WorkerResult(
            worker_id=worker_id,
            states=torch.cat(S),
            actions=torch.cat(A),
            log_probs=torch.cat(OL),
            returns=torch.cat(R),
            advantages=torch.cat(Adv),
            masks=torch.cat(M) if has_masks else None,
            opponent_results=opponent_results,
            episodes_completed=episodes_completed,
        ))
    else:
        # Edge case: all games ended without PPO decisions
        result_queue.put(WorkerResult(
            worker_id=worker_id,
            states=torch.empty(0),
            actions=torch.empty(0, dtype=torch.int64),
            log_probs=torch.empty(0),
            returns=torch.empty(0),
            advantages=torch.empty(0),
            masks=None,
            opponent_results=opponent_results,
            episodes_completed=episodes_completed,
        ))


def sim_worker_eval(
    worker_id: int,
    num_games: int,
    num_concurrent: int,
    data_config_dict: dict,
    card_names: list[str],
    card_index_map: dict[str, int],
    action_dim: int,
    state_size: int,
    opponent_spec: str,
    seed: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    result_queue: mp.Queue,
):
    """Entry point for an evaluation worker process.

    Same game loop as sim_worker_main but without rollout collection.
    Returns win/loss/step counts.
    """
    try:
        _sim_worker_eval_inner(
            worker_id, num_games, num_concurrent, data_config_dict,
            card_names, card_index_map, action_dim, state_size,
            opponent_spec, seed, request_queue, response_queue, result_queue,
        )
    except Exception:
        result_queue.put(WorkerError(
            worker_id=worker_id,
            error=traceback.format_exc(),
        ))


def _sim_worker_eval_inner(
    worker_id, num_games, num_concurrent, data_config_dict,
    card_names, card_index_map, action_dim, state_size,
    opponent_spec, seed, request_queue, response_queue, result_queue,
):
    """Core eval worker logic."""
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    torch.set_num_threads(1)
    set_disabled(True)

    data_cfg = DataConfig.from_dict(data_config_dict)
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)

    assert registry.card_names == card_names, (
        f"Eval worker {worker_id} card_names mismatch"
    )

    training_agent_name = "PPO"
    opponent_factory = _build_opponent_factory(opponent_spec)

    state_buf = np.zeros(state_size, dtype=np.float32)
    mask_buf = np.zeros(action_dim, dtype=bool)

    games: list[Optional[Game]] = [None] * num_concurrent
    step_counts: list[int] = [0] * num_concurrent

    wins = 0
    losses = 0
    total_steps = 0
    games_started = 0
    games_completed = 0
    request_id = 0

    active_count = min(num_concurrent, num_games)
    for i in range(active_count):
        opponent = opponent_factory()
        game = Game(cards, card_index_map=card_index_map)
        game.add_player(training_agent_name, _DummyAgent(training_agent_name))
        game.add_player(opponent.name, opponent)
        game.start_game()
        games[i] = game
        step_counts[i] = 0
        games_started += 1

    while games_completed < num_games:
        for i in range(num_concurrent):
            if games[i] is not None and not games[i].is_game_over:
                _advance_non_ppo(games[i], training_agent_name)

        for i in range(num_concurrent):
            if games[i] is None or not games[i].is_game_over:
                continue
            winner = games[i].get_winner()
            if winner == training_agent_name:
                wins += 1
            else:
                losses += 1
            total_steps += step_counts[i]
            games_completed += 1
            if games_started < num_games:
                opponent = opponent_factory()
                game = Game(cards, card_index_map=card_index_map)
                game.add_player(training_agent_name, _DummyAgent(training_agent_name))
                game.add_player(opponent.name, opponent)
                game.start_game()
                games[i] = game
                step_counts[i] = 0
                games_started += 1
                _advance_non_ppo(games[i], training_agent_name)
                if games[i].is_game_over:
                    w = games[i].get_winner()
                    if w == training_agent_name:
                        wins += 1
                    else:
                        losses += 1
                    total_steps += step_counts[i]
                    games_completed += 1
                    games[i] = None
            else:
                games[i] = None

        pending_indices: list[int] = []
        pending_states: list[torch.Tensor] = []
        pending_contexts: list[ActionContext] = []

        for i in range(num_concurrent):
            if games[i] is None or games[i].is_game_over:
                continue
            player = games[i].current_player
            if player.name != training_agent_name:
                continue

            ctx = build_action_context(
                games[i], player, card_index_map,
                action_dim, mask_buf=mask_buf,
            )
            state = encode_state(
                games[i], is_current_player_training=True,
                cards=card_names,
                card_index_map=card_index_map,
                state_buf=state_buf,
                can_buy=ctx.can_buy,
                has_actions=ctx.has_meaningful,
            )

            pending_indices.append(i)
            pending_states.append(state)
            pending_contexts.append(ActionContext(
                mask=ctx.mask.copy(),
                has_meaningful=ctx.has_meaningful,
                can_buy=ctx.can_buy,
                resolvers=ctx.resolvers,
            ))

        if not pending_states:
            continue

        states_np = np.stack([s.numpy() for s in pending_states])
        masks_np = np.zeros((len(pending_states), action_dim), dtype=np.uint8)
        for j, ctx in enumerate(pending_contexts):
            masks_np[j] = ctx.mask.astype(np.uint8)
            if ctx.has_meaningful:
                masks_np[j, END_TURN_INDEX] = 0

        request_queue.put(InferenceRequest(
            worker_id=worker_id,
            request_id=request_id,
            states=states_np,
            masks=masks_np,
        ))
        request_id += 1

        response: InferenceResponse = response_queue.get()

        for j, i in enumerate(pending_indices):
            act_idx = int(response.action_indices[j])
            action = pending_contexts[j].resolvers[act_idx]
            games[i].apply_decision(action)
            step_counts[i] += 1

    result_queue.put(EvalWorkerResult(
        worker_id=worker_id,
        wins=wins,
        losses=losses,
        total_steps=total_steps,
    ))
