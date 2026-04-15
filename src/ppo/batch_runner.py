"""BatchRunner: runs N concurrent games with batched GPU inference."""
import torch
from typing import Callable, Optional
from src.engine.game import Game
from src.engine.actions import ActionType, get_available_actions
from src.ai.agent import Agent
from src.ai.random_agent import RandomAgent
from src.cards.card import Card
from src.nn.state_encoder import encode_state
from src.nn.action_encoder import encode_action
from src.ppo.rollout_buffer import RolloutBuffer
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.utils.logger import log


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
    ):
        self.model = model
        self.card_names = card_names
        self.cards = cards
        self.action_dim = action_dim
        self.device = device
        self.opponent_factory = opponent_factory
        self.num_concurrent = num_concurrent
        self.training_agent_name = "PPO"

    def run_episodes(self, num_episodes: int) -> tuple:
        """Run num_episodes games and return aggregated rollout data.
        
        Returns: (states, actions, old_log_probs, returns, advantages)
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

            # Step 3: Collect pending PPO decisions
            pending_indices: list[int] = []
            pending_states: list[torch.Tensor] = []
            pending_masks: list[torch.Tensor] = []
            pending_available: list[list] = []
            pending_encoded: list[list[int]] = []

            for i in range(self.num_concurrent):
                if games[i] is None or games[i].is_game_over:
                    continue
                player = games[i].current_player
                if player.name != self.training_agent_name:
                    continue

                available = get_available_actions(games[i], player)
                encoded_actions = [encode_action(a, cards=self.card_names) for a in available]
                has_meaningful = any(
                    a.type in (ActionType.ATTACK_PLAYER, ActionType.PLAY_CARD)
                    for a in available
                )
                state = encode_state(
                    games[i], is_current_player_training=True,
                    cards=self.card_names, available_actions=available
                ).to(self.device)

                mask = torch.zeros(self.action_dim, device=self.device)
                mask[encoded_actions] = 1
                if has_meaningful and 1 in encoded_actions:
                    mask[1] = 0

                pending_indices.append(i)
                pending_states.append(state)
                pending_masks.append(mask)
                pending_available.append(available)
                pending_encoded.append(encoded_actions)

            if not pending_states:
                continue

            # Step 4: Batched forward pass
            states_batch = torch.stack(pending_states)
            masks_batch = torch.stack(pending_masks)

            with torch.no_grad():
                logits_batch, values_batch = self.model(states_batch)

            logits_batch = logits_batch.masked_fill(masks_batch == 0, float('-inf'))
            probs_batch = torch.softmax(logits_batch, dim=-1)
            dist = torch.distributions.Categorical(probs_batch)
            act_indices = dist.sample()
            log_probs = dist.log_prob(act_indices)

            # Step 5: Distribute actions and apply
            for j, i in enumerate(pending_indices):
                act_idx = act_indices[j].item()
                encoded = pending_encoded[j]
                available = pending_available[j]

                action = available[encoded.index(int(act_idx))]

                buffers[i].add(
                    pending_states[j],
                    act_idx,
                    log_probs[j],
                    values_batch[j],
                    reward=0.0,
                    done=False,
                )

                games[i].apply_decision(action)

        # Merge all rollouts
        if not completed_rollouts:
            raise RuntimeError("No completed rollouts")

        S, A, OL, R, Adv = zip(*completed_rollouts)
        return (
            torch.cat(S).to(self.device),
            torch.cat(A).to(self.device),
            torch.cat(OL).to(self.device),
            torch.cat(R).to(self.device),
            torch.cat(Adv).to(self.device),
        )

    def _start_game(self):
        """Initialize a new game."""
        opponent = self.opponent_factory()
        game = Game(self.cards)
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
        rollout = buffers[i].finish(gamma=0.995, lam=0.99, device=self.device)
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
