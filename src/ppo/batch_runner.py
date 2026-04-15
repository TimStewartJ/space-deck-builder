"""BatchRunner: runs N concurrent games with batched GPU inference."""
import random
import numpy as np
import torch
from typing import Callable, Optional
from concurrent.futures import ProcessPoolExecutor
from src.config import PPOConfig
from src.engine.game import Game
from src.engine.actions import ActionType, get_available_actions
from src.ai.agent import Agent
from src.ai.random_agent import RandomAgent
from src.cards.card import Card
from src.cards.registry import CardRegistry
from src.encoding.state_encoder import encode_state, get_state_size
from src.encoding.action_encoder import encode_action, get_action_space_size
from src.encoding.action_context import build_action_context, ActionContext
from src.ppo.rollout_buffer import RolloutBuffer
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.utils.logger import log, set_disabled


class _DummyAgent(Agent):
    """Placeholder agent for PPO player slot — decisions are made externally."""
    def make_decision(self, game_state):
        raise RuntimeError("DummyAgent.make_decision should never be called")


class BatchRunner:
    """Runs multiple games concurrently with batched neural net inference.
    
    Instead of one forward pass per game step, collects all pending PPO decisions
    across active games and evaluates them in a single batched forward pass.
    """

    def __init__(
        self,
        model: PPOActorCritic,
        card_names: list[str],
        cards: list[Card],
        action_dim: int,
        device: torch.device,
        opponent_factory: Callable[[], Agent] = lambda: RandomAgent("Rand"),
        num_concurrent: int = 64,
        ppo_config: PPOConfig | None = None,
        registry: CardRegistry | None = None,
    ):
        self.model = model
        self.card_names = card_names
        self.cards = cards
        self.action_dim = action_dim
        self.device = device
        self.opponent_factory = opponent_factory
        self.num_concurrent = num_concurrent
        self.ppo_config = ppo_config or PPOConfig()
        self.training_agent_name = "PPO"
        # Use registry's pre-built map, or build one on the fly for compat
        from src.encoding.state_encoder import get_state_size
        if registry is not None:
            self.card_index_map = registry.card_index_map
        else:
            from src.encoding.state_encoder import build_card_index_map
            self.card_index_map = build_card_index_map(card_names)
        # Pre-allocate reusable numpy buffer for state encoding
        self._state_size = get_state_size(card_names)
        self._state_buf = np.zeros(self._state_size, dtype=np.float32)
        # Pre-allocate reusable bool buffer for action mask generation
        self._mask_buf = np.zeros(action_dim, dtype=bool)

    def run_episodes(self, num_episodes: int) -> tuple:
        """Run num_episodes games and return aggregated rollout data.
        
        Returns: (states, actions, old_log_probs, returns, advantages, masks)
            masks is a Tensor of valid-action masks per step, or None if unavailable.
        """
        self.model.to(self.device)
        self.model.eval()

        # Active game slots
        games: list[Optional[Game]] = [None] * self.num_concurrent
        buffers: list[Optional[RolloutBuffer]] = [None] * self.num_concurrent

        completed_rollouts: list[tuple] = []
        episodes_started = 0
        episodes_completed = 0

        # Fill initial game slots
        active_count = min(self.num_concurrent, num_episodes)
        for i in range(active_count):
            games[i], buffers[i] = self._start_game()
            episodes_started += 1

        while episodes_completed < num_episodes:
            # Step 1: Advance all active games past non-PPO decisions.
            # After this, each active game is either game-over or waiting for PPO.
            for i in range(self.num_concurrent):
                if games[i] is None or games[i].is_game_over:
                    continue
                self._advance_non_ppo(games[i])

            # Step 2: Handle completed games BEFORE collecting new decisions
            for i in range(self.num_concurrent):
                if games[i] is None or not games[i].is_game_over:
                    continue
                self._finish_game(i, games, buffers, completed_rollouts)
                episodes_completed += 1
                if episodes_started < num_episodes:
                    games[i], buffers[i] = self._start_game()
                    episodes_started += 1
                    # Advance the new game past opponent's opening moves
                    self._advance_non_ppo(games[i])
                    if games[i].is_game_over:
                        self._finish_game(i, games, buffers, completed_rollouts)
                        episodes_completed += 1
                        games[i] = None
                        buffers[i] = None

            # Step 3: Collect pending PPO decisions using unified action context
            pending_indices: list[int] = []
            pending_states: list[torch.Tensor] = []
            pending_contexts: list[ActionContext] = []

            for i in range(self.num_concurrent):
                if games[i] is None or games[i].is_game_over:
                    continue
                player = games[i].current_player
                if player.name != self.training_agent_name:
                    continue

                ctx = build_action_context(
                    games[i], player, self.card_index_map,
                    self.action_dim, mask_buf=self._mask_buf,
                )
                state = encode_state(
                    games[i], is_current_player_training=True,
                    cards=self.card_names,
                    card_index_map=self.card_index_map,
                    state_buf=self._state_buf,
                    can_buy=ctx.can_buy,
                    has_actions=ctx.has_meaningful,
                )

                pending_indices.append(i)
                pending_states.append(state)
                # Copy mask since buffer is reused across iterations
                pending_contexts.append(ActionContext(
                    mask=ctx.mask.copy(),
                    has_meaningful=ctx.has_meaningful,
                    can_buy=ctx.can_buy,
                    resolvers=ctx.resolvers,
                ))

            if not pending_states:
                continue

            # Step 4: Batch construction — states, masks, forward pass
            states_batch = torch.stack(pending_states).to(self.device)
            masks_batch = torch.zeros(len(pending_states), self.action_dim, device=self.device)
            for j, ctx in enumerate(pending_contexts):
                masks_batch[j] = torch.from_numpy(ctx.mask.astype(np.float32))
                # Suppress END_TURN when meaningful actions exist
                if ctx.has_meaningful:
                    masks_batch[j, 1] = 0

            with torch.no_grad():
                logits_batch, values_batch = self.model(states_batch)

            logits_batch = logits_batch.masked_fill(masks_batch == 0, float('-inf'))
            dist = torch.distributions.Categorical(logits=logits_batch)
            act_indices = dist.sample()
            log_probs = dist.log_prob(act_indices)

            # Step 5: Distribute actions and apply
            act_indices_cpu = act_indices.tolist()
            for j, i in enumerate(pending_indices):
                act_idx = act_indices_cpu[j]
                action = pending_contexts[j].resolvers[act_idx]

                buffers[i].add(
                    pending_states[j],
                    act_idx,
                    log_probs[j],
                    values_batch[j],
                    reward=0.0,
                    done=False,
                    mask=masks_batch[j],
                )

                games[i].apply_decision(action)

        # Merge all rollouts — tensors already on self.device from RolloutBuffer
        if not completed_rollouts:
            raise RuntimeError("No completed rollouts")

        S, A, OL, R, Adv, M = zip(*completed_rollouts)
        has_masks = all(m is not None for m in M)
        advs = torch.cat(Adv)
        if self.ppo_config.adv_norm == "global":
            advs = (advs - advs.mean()) / (advs.std(unbiased=False) + 1e-8)
        return (
            torch.cat(S),
            torch.cat(A),
            torch.cat(OL),
            torch.cat(R),
            advs,
            torch.cat(M) if has_masks else None,
        )

    def _start_game(self):
        """Initialize a new game."""
        opponent = self.opponent_factory()
        game = Game(self.cards, card_index_map=self.card_index_map)
        game.add_player(self.training_agent_name, _DummyAgent(self.training_agent_name))
        game.add_player(opponent.name, opponent)
        game.start_game()
        buf = RolloutBuffer()
        return game, buf

    def _finish_game(self, i, games, buffers, completed_rollouts):
        """Handle a completed game: compute reward, finalize rollout."""
        if buffers[i] is None or len(buffers[i]) == 0:
            games[i] = None
            buffers[i] = None
            return

        winner = games[i].get_winner()
        reward = 1.0 if winner == self.training_agent_name else -1.0
        buffers[i].fill_last_reward(reward)
        rollout = buffers[i].finish(
            gamma=self.ppo_config.gamma,
            lam=self.ppo_config.lam,
            device=self.device,
            normalize=(self.ppo_config.adv_norm == "per_episode"),
        )
        completed_rollouts.append(rollout)
        games[i] = None
        buffers[i] = None

    def _advance_non_ppo(self, game: Game):
        """Advance the game while the current player is NOT the PPO training agent."""
        while not game.is_game_over:
            player = game.current_player
            if player.name == self.training_agent_name:
                break
            action = player.make_decision(game)
            game.apply_decision(action)

    def run_eval(self, num_games: int) -> tuple[int, int, int]:
        """Run num_games evaluation games. Returns (wins, losses, total_steps).
        
        Same batched inference loop as run_episodes but without rollout storage.
        """
        self.model.to(self.device)
        self.model.eval()

        games: list[Optional[Game]] = [None] * self.num_concurrent
        step_counts: list[int] = [0] * self.num_concurrent

        wins = 0
        losses = 0
        total_steps = 0
        games_started = 0
        games_completed = 0

        active_count = min(self.num_concurrent, num_games)
        for i in range(active_count):
            games[i] = self._start_eval_game()
            step_counts[i] = 0
            games_started += 1

        while games_completed < num_games:
            # Advance past non-PPO decisions
            for i in range(self.num_concurrent):
                if games[i] is None or games[i].is_game_over:
                    continue
                self._advance_non_ppo(games[i])

            # Handle completed games
            for i in range(self.num_concurrent):
                if games[i] is None or not games[i].is_game_over:
                    continue
                winner = games[i].get_winner()
                if winner == self.training_agent_name:
                    wins += 1
                else:
                    losses += 1
                total_steps += step_counts[i]
                games_completed += 1
                if games_started < num_games:
                    games[i] = self._start_eval_game()
                    step_counts[i] = 0
                    games_started += 1
                    self._advance_non_ppo(games[i])
                    if games[i].is_game_over:
                        w = games[i].get_winner()
                        if w == self.training_agent_name:
                            wins += 1
                        else:
                            losses += 1
                        total_steps += step_counts[i]
                        games_completed += 1
                        games[i] = None
                else:
                    games[i] = None

            # Collect pending PPO decisions using unified action context
            pending_indices: list[int] = []
            pending_states: list[torch.Tensor] = []
            pending_contexts: list[ActionContext] = []

            for i in range(self.num_concurrent):
                if games[i] is None or games[i].is_game_over:
                    continue
                player = games[i].current_player
                if player.name != self.training_agent_name:
                    continue

                ctx = build_action_context(
                    games[i], player, self.card_index_map,
                    self.action_dim, mask_buf=self._mask_buf,
                )
                state = encode_state(
                    games[i], is_current_player_training=True,
                    cards=self.card_names,
                    card_index_map=self.card_index_map,
                    state_buf=self._state_buf,
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

            # Batched forward pass with batch-constructed masks
            states_batch = torch.stack(pending_states).to(self.device)
            masks_batch = torch.zeros(len(pending_states), self.action_dim, device=self.device)
            for j, ctx in enumerate(pending_contexts):
                masks_batch[j] = torch.from_numpy(ctx.mask.astype(np.float32))
                if ctx.has_meaningful:
                    masks_batch[j, 1] = 0

            with torch.no_grad():
                logits_batch, values_batch = self.model(states_batch)

            logits_batch = logits_batch.masked_fill(masks_batch == 0, float('-inf'))
            dist = torch.distributions.Categorical(logits=logits_batch)
            act_indices = dist.sample()

            act_indices_cpu = act_indices.tolist()
            for j, i in enumerate(pending_indices):
                act_idx = act_indices_cpu[j]
                action = pending_contexts[j].resolvers[act_idx]
                games[i].apply_decision(action)
                step_counts[i] += 1

        return wins, losses, total_steps

    def _start_eval_game(self):
        """Initialize a new game for evaluation (no rollout buffer needed)."""
        opponent = self.opponent_factory()
        game = Game(self.cards, card_index_map=self.card_index_map)
        game.add_player(self.training_agent_name, _DummyAgent(self.training_agent_name))
        game.add_player(opponent.name, opponent)
        game.start_game()
        return game


def _worker_run_episodes(state_dict, card_names, cards, action_dim,
                         num_episodes, num_concurrent, opponent_type,
                         opponent_state_dict, seed):
    """Top-level worker function for ProcessPoolExecutor.
    
    Reconstructs model + BatchRunner in a subprocess, runs episodes on CPU,
    returns rollout tensors.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    set_disabled(True)

    # Limit PyTorch internal threads to avoid contention across workers
    torch.set_num_threads(1)

    device = torch.device("cpu")
    model = PPOActorCritic(
        get_state_size(card_names), action_dim, len(card_names)
    ).to(device)
    model.load_state_dict(state_dict)

    if opponent_type == "ppo":
        opp_sd = opponent_state_dict
        def make_opponent():
            from src.ai.ppo_agent import PPOAgent
            opp = PPOAgent("Opp", card_names, device="cpu",
                           main_device="cpu", simulation_device="cpu")
            opp.model.load_state_dict(opp_sd)
            return opp
    else:
        def make_opponent():
            return RandomAgent("Rand")

    runner = BatchRunner(
        model=model,
        card_names=card_names,
        cards=cards,
        action_dim=action_dim,
        device=device,
        opponent_factory=make_opponent,
        num_concurrent=num_concurrent,
    )
    return runner.run_episodes(num_episodes)


def run_episodes_parallel(model, card_names, cards, action_dim,
                          num_episodes, num_workers=4,
                          games_per_worker=16,
                          opponent_type="random",
                          opponent_state_dict=None,
                          device=torch.device("cpu")):
    """Run episodes across multiple worker processes, each with its own BatchRunner.
    
    Returns merged (states, actions, old_lp, returns, advantages, masks) on the given device.
    masks is a Tensor of valid-action masks per step, or None if unavailable.
    """
    state_dict = model.cpu().state_dict()

    # Divide episodes across workers
    base = num_episodes // num_workers
    remainder = num_episodes % num_workers
    episode_counts = [base + (1 if i < remainder else 0) for i in range(num_workers)]

    seeds = [random.randint(0, 1_000_000_000) for _ in range(num_workers)]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                _worker_run_episodes,
                state_dict, card_names, cards, action_dim,
                ep_count, games_per_worker, opponent_type,
                opponent_state_dict, seed
            )
            for ep_count, seed in zip(episode_counts, seeds)
        ]
        results = [f.result() for f in futures]

    S, A, OL, R, Adv, M = zip(*results)
    has_masks = all(m is not None for m in M)
    return (
        torch.cat(S).to(device),
        torch.cat(A).to(device),
        torch.cat(OL).to(device),
        torch.cat(R).to(device),
        torch.cat(Adv).to(device),
        torch.cat(M).to(device) if has_masks else None,
    )
