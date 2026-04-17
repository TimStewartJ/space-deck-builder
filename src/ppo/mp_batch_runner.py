"""Multi-process batch runner — drop-in replacement for BatchRunner.

Distributes game simulation across N worker processes while keeping
GPU inference centralized in an InferenceServer thread. Workers run
game logic and encoding on CPU, sending encoded states to the GPU
via multiprocessing Queues.
"""
import random
import time
import multiprocessing as mp
from typing import Callable, Optional

import torch
import numpy as np

from src.config import DataConfig, PPOConfig, RunConfig
from src.ai.agent import Agent
from src.ai.random_agent import RandomAgent
from src.cards.card import Card
from src.cards.registry import CardRegistry
from src.encoding.state_encoder import get_state_size
from src.encoding.action_encoder import get_action_space_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.ppo.mp_inference_server import (
    InferenceServer, WorkerResult, WorkerError, EvalWorkerResult,
)
from src.ppo.mp_sim_worker import sim_worker_main, sim_worker_eval


def _divide_work(total: int, num_workers: int) -> list[int]:
    """Divide total items across workers, distributing remainder evenly."""
    base = total // num_workers
    remainder = total % num_workers
    return [base + (1 if i < remainder else 0) for i in range(num_workers)]


class MultiProcessBatchRunner:
    """Runs episodes across multiple worker processes with centralized GPU inference.

    Drop-in replacement for BatchRunner — same run_episodes() and run_eval() API.
    Each worker manages a share of the concurrent games, running game simulation
    and state encoding on CPU. The InferenceServer thread handles all GPU forward
    passes in the main process.
    """

    def __init__(
        self,
        model: PPOActorCritic,
        card_names: list[str],
        cards: list[Card],
        action_dim: int,
        device: torch.device,
        data_config: DataConfig,
        opponent_spec: str = "random",
        num_concurrent: int | None = None,
        num_workers: int = 4,
        ppo_config: PPOConfig | None = None,
        registry: CardRegistry | None = None,
        snapshot_state_dicts: list[tuple[str, dict, "ModelConfig | None"]] | None = None,
        self_play_ratio: float = 0.5,
        pfsp_weights: list[float] | None = None,
    ):
        self.model = model
        self.card_names = card_names
        self.cards = cards
        self.action_dim = action_dim
        self.device = device
        self.data_config = data_config
        self.opponent_spec = opponent_spec
        self.num_concurrent = num_concurrent if num_concurrent is not None else RunConfig().num_concurrent
        self.num_workers = num_workers
        self.ppo_config = ppo_config or PPOConfig()
        self.training_agent_name = "PPO"
        self.snapshot_state_dicts = snapshot_state_dicts
        self.self_play_ratio = self_play_ratio
        self.pfsp_weights = pfsp_weights

        if registry is not None:
            self.card_index_map = registry.card_index_map
        else:
            from src.encoding.state_encoder import build_card_index_map
            self.card_index_map = build_card_index_map(card_names)

        self._state_size = get_state_size(card_names)
        # Per-opponent result tracking (merged from all workers)
        self.opponent_results: dict[str, list[int]] = {}

        # Use explicit spawn context for Windows compatibility
        self._mp_ctx = mp.get_context("spawn")

    def _spawn_workers(
        self,
        target_fn,
        num_items: int,
        items_per_worker: list[int],
        *,
        use_snapshots: bool = False,
        extra_args_fn=None,
        extra_kwargs_fn=None,
    ):
        """Spawn worker processes with a centralized InferenceServer.

        Shared setup for both run_episodes() and run_eval(): creates the
        inference server (with optional opponent snapshots on GPU), generates
        seeds, and spawns one process per worker.

        Args:
            target_fn: Worker entry point (sim_worker_main or sim_worker_eval)
            num_items: Total episodes/games to run
            items_per_worker: Pre-computed work distribution
            use_snapshots: Load opponent snapshots into InferenceServer
            extra_args_fn: Optional callable(worker_id) -> tuple of extra
                positional args inserted after the common args
            extra_kwargs_fn: Optional callable(snapshot_names) -> dict of
                extra kwargs merged into the snapshot kwargs

        Returns: (server, processes, result_queue)
        """
        actual_workers = len(items_per_worker)

        # Start inference server, optionally with opponent snapshots on GPU
        snapshots = self.snapshot_state_dicts if use_snapshots else None
        server = InferenceServer(
            self.model, self.device, actual_workers, ctx=self._mp_ctx,
            opponent_snapshots=snapshots,
        )
        server.start()

        # Extract snapshot names for workers (state_dicts stay on server)
        snapshot_names = (
            [name for name, _, _ in self.snapshot_state_dicts]
            if use_snapshots and self.snapshot_state_dicts else None
        )

        result_queue = self._mp_ctx.Queue()
        seeds = [random.randint(0, 1_000_000_000) for _ in range(actual_workers)]

        # Snapshot kwargs forwarded to workers
        snapshot_kwargs = {
            "snapshot_names": snapshot_names,
            "self_play_ratio": self.self_play_ratio,
            "pfsp_weights": self.pfsp_weights,
        }
        if extra_kwargs_fn:
            snapshot_kwargs.update(extra_kwargs_fn(snapshot_names))

        processes = []
        for i in range(actual_workers):
            if items_per_worker[i] == 0:
                continue

            # Common positional args shared by all worker types
            common_args = (
                i,                              # worker_id
                items_per_worker[i],            # num_episodes / num_games
                self.num_concurrent,            # num_concurrent (per worker)
                self.data_config.to_dict(),
                self.card_names,
                self.card_index_map,
                self.action_dim,
                self._state_size,
            )
            # Worker-specific positional args (e.g. ppo_config_dict for training)
            extra = extra_args_fn(i) if extra_args_fn else ()

            # Remaining common positional args
            tail_args = (
                self.opponent_spec,
                seeds[i],
                server.request_queue,
                server.response_queues[i],
                result_queue,
            )

            p = self._mp_ctx.Process(
                target=target_fn,
                args=common_args + extra + tail_args,
                kwargs=snapshot_kwargs,
            )
            p.start()
            processes.append(p)

        return server, processes, result_queue

    def run_episodes(self, num_episodes: int) -> tuple:
        """Run num_episodes games across worker processes and return aggregated rollout data.

        Returns: (states, actions, old_log_probs, returns, advantages, masks)
        """
        self.model.to(self.device)
        self.model.eval()
        self.opponent_results = {}

        actual_workers = min(self.num_workers, num_episodes)
        if actual_workers <= 0:
            raise ValueError(f"Cannot run {num_episodes} episodes with {self.num_workers} workers")

        episodes_per_worker = _divide_work(num_episodes, actual_workers)

        # Training workers need ppo_config_dict as an extra positional arg
        ppo_dict = self.ppo_config.to_dict()

        server, processes, result_queue = self._spawn_workers(
            sim_worker_main,
            num_episodes,
            episodes_per_worker,
            use_snapshots=bool(self.snapshot_state_dicts),
            extra_args_fn=lambda _i: (ppo_dict,),
        )

        # Collect results with cleanup guaranteed on error
        try:
            results = self._collect_results(
                result_queue, len(processes), processes, timeout_per_result=300,
            )
        finally:
            server.stop()
            self._reap_processes(processes, result_queue)

        # Merge results from all workers
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []
        all_masks = []
        has_masks = True

        for r in results:
            if r.states.numel() == 0:
                continue
            all_states.append(r.states)
            all_actions.append(r.actions)
            all_log_probs.append(r.log_probs)
            all_returns.append(r.returns)
            all_advantages.append(r.advantages)
            if r.masks is not None:
                all_masks.append(r.masks)
            else:
                has_masks = False

            # Merge opponent results
            for opp_name, counts in r.opponent_results.items():
                existing = self.opponent_results.get(opp_name)
                if existing is None:
                    self.opponent_results[opp_name] = list(counts)
                else:
                    existing[0] += counts[0]
                    existing[1] += counts[1]

        if not all_states:
            raise RuntimeError("No completed rollouts from any worker")

        states = torch.cat(all_states).to(self.device)
        actions = torch.cat(all_actions).to(self.device)
        log_probs = torch.cat(all_log_probs).to(self.device)
        returns = torch.cat(all_returns).to(self.device)
        advs = torch.cat(all_advantages)
        if self.ppo_config.adv_norm == "global":
            advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)
        advs = advs.to(self.device)
        masks = torch.cat(all_masks).to(self.device) if has_masks and all_masks else None

        return states, actions, log_probs, returns, advs, masks

    def run_eval(self, num_games: int) -> tuple[int, int, int]:
        """Run evaluation games across workers. Returns (wins, losses, total_steps).

        Supports PPO snapshot opponents when snapshot_state_dicts is set —
        both training agent and opponent get batched GPU inference through
        the InferenceServer.
        """
        self.model.to(self.device)
        self.model.eval()

        actual_workers = min(self.num_workers, num_games)
        if actual_workers <= 0:
            return 0, 0, 0

        games_per_worker = _divide_work(num_games, actual_workers)

        server, processes, result_queue = self._spawn_workers(
            sim_worker_eval,
            num_games,
            games_per_worker,
            use_snapshots=bool(self.snapshot_state_dicts),
        )

        # Collect eval results with cleanup guaranteed
        total_wins = 0
        total_losses = 0
        total_steps = 0
        errors = []

        try:
            results_collected = 0
            deadline = time.monotonic() + 300

            while results_collected < len(processes):
                remaining = max(1, deadline - time.monotonic())
                try:
                    result = result_queue.get(timeout=min(remaining, 10))
                except Exception:
                    alive = [p for p in processes if p.is_alive()]
                    if not alive:
                        break
                    if time.monotonic() > deadline:
                        errors.append(f"Timed out ({results_collected}/{len(processes)} results)")
                        break
                    continue

                if isinstance(result, WorkerError):
                    errors.append(f"Worker {result.worker_id}: {result.error}")
                    results_collected += 1
                    continue

                total_wins += result.wins
                total_losses += result.losses
                total_steps += result.total_steps
                results_collected += 1
        finally:
            server.stop()
            self._reap_processes(processes, result_queue)

        if errors:
            actual_games = total_wins + total_losses
            print(f"WARNING: Eval incomplete ({actual_games}/{num_games} games). "
                  f"Errors: {'; '.join(errors)}")

        return total_wins, total_losses, total_steps

    @staticmethod
    def _reap_processes(processes, result_queue):
        """Join workers (escalating to terminate/kill) and release the result
        queue. See RCA 2026-04-17: a leaked mp.Queue with a still-active feeder
        thread will wedge the interpreter at atexit, producing the
        "Training complete." hang.
        """
        for p in processes:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            if p.is_alive():
                # Last resort on Windows: TerminateProcess via kill().
                try:
                    p.kill()
                except Exception:
                    pass
        try:
            result_queue.close()
        except Exception:
            pass
        try:
            result_queue.cancel_join_thread()
        except Exception:
            pass

    def _collect_results(
        self,
        result_queue: mp.Queue,
        expected: int,
        processes: list,
        timeout_per_result: float = 300,
    ) -> list[WorkerResult]:
        """Collect WorkerResults with timeout and health monitoring."""
        results: list[WorkerResult] = []
        collected = 0
        deadline = time.monotonic() + timeout_per_result

        while collected < expected:
            remaining = max(1, deadline - time.monotonic())
            try:
                result = result_queue.get(timeout=min(remaining, 10))
            except Exception:
                # Check worker health
                alive = [p for p in processes if p.is_alive()]
                if not alive and collected < expected:
                    raise RuntimeError(
                        f"All workers exited but only {collected}/{expected} results received. "
                        f"Exit codes: {[p.exitcode for p in processes]}"
                    )
                if time.monotonic() > deadline:
                    raise RuntimeError(
                        f"Timed out waiting for worker results ({collected}/{expected} received). "
                        f"Alive: {len(alive)}/{len(processes)}"
                    )
                continue

            if isinstance(result, WorkerError):
                raise RuntimeError(
                    f"Worker {result.worker_id} crashed:\n{result.error}"
                )

            results.append(result)
            collected += 1
            # Reset deadline for next result
            deadline = time.monotonic() + timeout_per_result

        return results
