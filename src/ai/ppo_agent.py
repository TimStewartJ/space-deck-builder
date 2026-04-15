import torch
import torch.nn as nn
import torch.optim as optim
from typing import TYPE_CHECKING, List, Optional
import time
from src.config import PPOConfig, ModelConfig, DeviceConfig, load_checkpoint
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.encoding.state_utils import unpack_state
from src.ai.agent import Agent
from src.engine.actions import ActionType, get_available_actions
from src.encoding.state_encoder import encode_state, get_state_size, build_card_index_map
from src.encoding.action_encoder import encode_action, decode_action, get_action_space_size, END_TURN_INDEX
from src.utils.logger import log

if TYPE_CHECKING:
    from src.engine.game import Game

class PPOAgent(Agent):
    def __init__(
        self,
        name: str,
        card_names: List[str],
        lr: float | None = None,
        gamma: float | None = None,
        lam: float | None = None,
        clip_eps: float | None = None,
        epochs: int | None = None,
        batch_size: int | None = None,
        entropy_coef: float | None = None,
        device: str | None = None,
        main_device: str | None = None,
        simulation_device: str | None = None,
        model_path: Optional[str] = None,
        log_debug: bool = False,
        # Config-based construction (preferred)
        ppo_config: PPOConfig | None = None,
        model_config: ModelConfig | None = None,
        device_config: DeviceConfig | None = None,
        registry=None,
    ):
        super().__init__(name)

        # Store registry's pre-built card_index_map to avoid per-call rebuilds
        if registry is not None:
            self.card_index_map = registry.card_index_map
        else:
            self.card_index_map = build_card_index_map(card_names)

        # Resolve configs: explicit kwargs override config object values
        ppo = ppo_config or PPOConfig()
        mdl = model_config or ModelConfig()
        dev = device_config or DeviceConfig()

        self.ppo_config = PPOConfig(
            lr=lr if lr is not None else ppo.lr,
            gamma=gamma if gamma is not None else ppo.gamma,
            lam=lam if lam is not None else ppo.lam,
            clip_eps=clip_eps if clip_eps is not None else ppo.clip_eps,
            epochs=epochs if epochs is not None else ppo.epochs,
            batch_size=batch_size if batch_size is not None else ppo.batch_size,
            entropy_coef=entropy_coef if entropy_coef is not None else ppo.entropy_coef,
            grad_clip=ppo.grad_clip,
            critic_loss_coef=ppo.critic_loss_coef,
            adv_norm=ppo.adv_norm,
        )
        self.model_config = mdl

        # Convenience aliases for backward compat with internal usage
        self.gamma = self.ppo_config.gamma
        self.lam = self.ppo_config.lam
        self.clip_eps = self.ppo_config.clip_eps
        self.epochs = self.ppo_config.epochs
        self.batch_size = self.ppo_config.batch_size
        self.entropy_coef = self.ppo_config.entropy_coef

        # Resolve devices: DeviceConfig > explicit kwargs > DeviceConfig defaults
        if device_config is not None:
            main_dev_str = dev.main_device
            sim_dev_str = dev.simulation_device
        else:
            main_dev_str = main_device if main_device is not None else dev.main_device
            sim_dev_str = simulation_device if simulation_device is not None else dev.simulation_device
        # Legacy `device` kwarg overrides both if provided
        if device is not None:
            main_dev_str = device
            sim_dev_str = device

        self.main_device = torch.device(DeviceConfig.resolve(main_dev_str))
        self.simulation_device = torch.device(DeviceConfig.resolve(sim_dev_str))
        self.device = self.simulation_device  # active device starts as simulation

        self.cards = card_names

        self.state_dim = get_state_size(card_names)
        self.action_dim = get_action_space_size(card_names)

        if log_debug:
            log(f"State size: {self.state_dim}, Action size: {self.action_dim}")
            log(f"Using device: {self.device}")

        # When loading a checkpoint, use its saved ModelConfig so the
        # architecture matches the stored weights (e.g. embedding dims,
        # hidden sizes). Fall back to the current default if the checkpoint
        # predates config storage.
        if model_path:
            log(f"Loading PPO model from {model_path}")
            ckpt = load_checkpoint(model_path, map_location=self.device)
            saved_model_cfg = ckpt.get("config", {}).get("model")
            if saved_model_cfg:
                self.model_config = ModelConfig.from_dict(saved_model_cfg)

        self.model = PPOActorCritic(
            self.state_dim, self.action_dim, len(card_names),
            model_config=self.model_config
        ).to(self.device)
        if model_path:
            self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.ppo_config.lr)

        # rollout buffers
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards: list[float] = []
        self.values: list[torch.Tensor] = []
        self.dones: list[bool] = []
        self.masks: list[torch.Tensor] = []

        # decision timing
        self.total_decision_time = 0.0
        self.num_decisions = 0

    def make_decision(self, game_state: 'Game'):
        # Device contract: model is placed on the correct device at
        # construction or by the caller (e.g. OpponentPool factory).
        # No per-call model.to() — that would be expensive in self-play.
        start_time = time.perf_counter()
        available = get_available_actions(game_state, game_state.current_player)
        encoded_actions = [encode_action(a, cards=self.cards, card_index_map=self.card_index_map) for a in available]
        has_available_actions = True if any(
            action.type == ActionType.ATTACK_PLAYER or action.type == ActionType.PLAY_CARD
            for action in available
        ) else False
        state = encode_state(
            game_state, is_current_player_training=True,
            cards=self.cards,
            available_actions=available,
            card_index_map=self.card_index_map,
        ).to(self.device)

        logits: torch.Tensor
        value: torch.Tensor
        logits, value = self.model(state)
        # mask out unavailable actions using encoded_actions
        mask = torch.zeros(self.action_dim, dtype=torch.bool, device=self.device)
        mask[encoded_actions] = True
        # Suppress END_TURN when meaningful actions are available
        if has_available_actions and END_TURN_INDEX in encoded_actions:
            mask[END_TURN_INDEX] = False
        logits = logits.masked_fill(~mask, float('-inf'))

        dist = torch.distributions.Categorical(logits=logits)
        act_idx = dist.sample().item()
        logp: torch.Tensor = dist.log_prob(torch.tensor(act_idx, device=self.device))

        # decode back to Action
        action = available[encoded_actions.index(int(act_idx))]

        reward = 0.0
        # If the end turn action was taken while other actions were available,
        # lower the reward to encourage taking other actions
        # if has_available_actions and action.type != ActionType.END_TURN:
        #     reward = 0.00025

        # store
        self.states.append(state.detach())
        self.actions.append(act_idx)
        self.log_probs.append(logp.detach())
        self.values.append(value.detach())
        self.rewards.append(reward)
        self.dones.append(False)
        self.masks.append(mask.detach())

        # timing
        elapsed = time.perf_counter() - start_time
        self.total_decision_time += elapsed
        self.num_decisions += 1

        return action

    def fill_last_reward(self, reward: float):
        self.rewards[-1] = reward
        self.dones[-1] = True

    def create_dummy_state(self, game: 'Game'):
        # encode final state so we get a “next‐state” entry with value=0
        final_state = encode_state(game, True, cards=self.cards, available_actions=[])
        self.states.append(final_state.detach())
        self.actions.append(0)          # dummy
        self.log_probs.append(torch.tensor(0.0, device=self.device))
        self.values.append(torch.tensor(0.0, device=self.device))
        self.rewards.append(0.0)
        self.dones.append(True)         # marks bootstrap point

    def finish_batch(self):
        # Device contract: tensors are already on self.device from make_decision().
        # No model.to() needed — model stays on its current device.
        returns: list[torch.Tensor] = []
        advs: list[torch.Tensor] = []
        gae = 0.0
        vals = self.values + [torch.tensor(0.0, device=self.device)]
        # Compute Generalized Advantage Estimation (GAE) and returns
        for step in reversed(range(len(self.rewards))):
            # Temporal difference error (delta)
            delta = (
                self.rewards[step]                           # immediate reward at this step
                + self.gamma * vals[step + 1] * (1 - self.dones[step])  # discounted next value if not done
                - vals[step]                                 # subtract current value estimate
            )
            # GAE calculation
            gae = delta + self.gamma * self.lam * (1 - self.dones[step]) * gae
            advs.insert(0, gae)  # Insert advantage at the beginning
            returns.insert(0, gae + vals[step])  # Return = advantage + value

        # log_probs, values, masks are already on self.device (from model output).
        # states originate from encode_state() on CPU and need device placement.
        states   = torch.stack(self.states).to(self.device)
        actions  = torch.tensor(self.actions, dtype=torch.int64, device=self.device)
        old_lp   = torch.stack(self.log_probs)
        returnsTensor  = torch.stack(returns).to(self.device)
        advsTensor     = torch.stack(advs).to(self.device)
        advsTensor = (advsTensor - advsTensor.mean()) / (advsTensor.std(unbiased=False) + 1e-8) # normalize advantages
        masksTensor = torch.stack(self.masks) if self.masks else None

        # clear buffers
        self.clear_buffers()

        return states, actions, old_lp, returnsTensor, advsTensor, masksTensor

    def update(self, states, actions, old_lp, returns, advs, masks=None):
        # Device contract: caller (ppo_trainer) is responsible for moving
        # model and tensors to main_device before calling update().

        actor_loss_sum  = 0.0
        critic_loss_sum = 0.0
        ratio_sum       = 0.0
        entropy_sum     = 0.0
        batch_count     = 0

        for _ in range(self.epochs):
            idxs = torch.randperm(len(states))  # Shuffle indices for mini-batch SGD
            for start in range(0, len(states), self.batch_size):
                b = idxs[start:start+self.batch_size]  # Indices for current batch
                s = states[b]        # Batch of states
                a = actions[b]       # Batch of actions
                olp = old_lp[b]      # Batch of old log probabilities
                R = returns[b]       # Batch of returns (discounted rewards)
                A = advs[b]          # Batch of advantages

                logits, vals = self.model(s)  # Forward pass: get action logits and value estimates

                # Apply action masks to ensure consistency with the behavior policy
                if masks is not None:
                    m = masks[b]
                    logits = logits.masked_fill(~m, float('-inf'))

                dist = torch.distributions.Categorical(logits=logits)  # Action distribution from masked logits
                nl = dist.log_prob(a)  # New log probabilities for taken actions
                ratio = (nl - olp).exp()  # Probability ratio for PPO objective

                s1 = ratio * A  # Unclipped surrogate objective
                s2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * A  # Clipped surrogate objective
                actor_loss = -torch.min(s1, s2).mean()  # PPO actor loss (policy update)
                critic_loss = nn.MSELoss()(vals, R)     # Critic loss (value function update)
                entropy = dist.entropy().mean()         # entropy bonus term
                loss = actor_loss + self.ppo_config.critic_loss_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.ppo_config.grad_clip)
                self.optimizer.step()

                # accumulate for averages
                actor_loss_sum  += actor_loss.item()
                critic_loss_sum += critic_loss.item()
                ratio_sum       += ratio.mean().item()
                entropy_sum     += entropy.item()
                batch_count     += 1

        # compute true averages
        if batch_count > 0:
            avg_actor   = actor_loss_sum  / batch_count
            avg_critic  = critic_loss_sum / batch_count
            avg_ratio   = ratio_sum       / batch_count
            avg_entropy = entropy_sum     / batch_count
        else:
            avg_actor = avg_critic = avg_ratio = avg_entropy = 0.0

        # final diagnostic log
        log(
            f"PPO update done. "
            f"Avg actor loss: {avg_actor:.3f}, "
            f"Avg critic loss: {avg_critic:.3f}, "
            f"Mean ratio: {avg_ratio:.3f}, "
            f"Mean entropy: {avg_entropy:.3f}"
        )

    def clear_buffers(self):
        """Clear all rollout buffers."""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []

    def get_average_decision_time(self, reset: bool = True):
        if self.num_decisions == 0:
            avg = 0.0
        else:
            avg = self.total_decision_time / self.num_decisions
        if reset:
            self.total_decision_time = 0.0
            self.num_decisions = 0
        return avg