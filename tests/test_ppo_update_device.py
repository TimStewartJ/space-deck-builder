"""PPO updates on a CUDA/HIP device: fused Adam and device-side NaN handling."""
import copy

import pytest
import torch

from src.config import DataConfig, DeviceConfig, PPOConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA/HIP device"
)


@pytest.fixture(scope="module")
def card_setup():
    data_cfg = DataConfig()
    cards = data_cfg.load_cards()
    registry = data_cfg.build_registry(cards)
    return registry


def _agent(registry, device: str, *, epochs=1, batch_size=32):
    from src.ai.ppo_agent import PPOAgent

    torch.manual_seed(0)
    return PPOAgent(
        "PPO",
        registry.card_names,
        ppo_config=PPOConfig(epochs=epochs, batch_size=batch_size),
        device_config=DeviceConfig(main_device=device, simulation_device=device),
        registry=registry,
    )


def _rollout(agent, rows: int, device: str):
    generator = torch.Generator().manual_seed(1)
    states = torch.randint(
        0, 3, (rows, agent.state_dim), generator=generator
    ).float()
    actions = torch.randint(0, agent.action_dim, (rows,), generator=generator)
    masks = torch.rand(rows, agent.action_dim, generator=generator) < 0.3
    masks[torch.arange(rows), actions] = True
    with torch.no_grad():
        logits, _ = agent.model(states.to(agent.model.card_emb.weight.device))
        old_lp = torch.log_softmax(
            logits.cpu().masked_fill(~masks, float("-inf")), dim=-1
        ).gather(1, actions.unsqueeze(1)).squeeze(1)
    returns = torch.rand(rows, generator=generator) * 2 - 1
    advs = torch.randn(rows, generator=generator)
    return [
        tensor.to(device)
        for tensor in (states, actions, old_lp, returns, advs, masks)
    ]


def _in_order(monkeypatch):
    """Make the minibatch shuffle the identity so batches are predictable."""
    def identity(n, *args, device=None, **kwargs):
        return torch.arange(n, device=device)

    monkeypatch.setattr(torch, "randperm", identity)


def test_cuda_agent_uses_fused_adam(card_setup):
    agent = _agent(card_setup, "cuda")
    assert agent._fused_optimizer
    assert agent.optimizer.param_groups[0]["fused"] is True
    result = agent.update(*_rollout(agent, 96, "cuda"))
    assert result["nan_detected"] is False
    assert all(
        value == value for key, value in result.items() if key != "nan_detected"
    )
    assert not hasattr(agent.optimizer, "found_inf")


def test_nonfinite_minibatch_skips_its_step_and_every_later_one(
    card_setup, monkeypatch
):
    _in_order(monkeypatch)
    agent = _agent(card_setup, "cuda")
    reference = copy.deepcopy(agent)
    states, actions, old_lp, returns, advs, masks = _rollout(agent, 128, "cuda")

    # Minibatches 0 and 1 are clean; minibatch 2 has a NaN advantage.
    poisoned = advs.clone()
    poisoned[2 * 32 + 5] = float("nan")
    result = agent.update(states, actions, old_lp, returns, poisoned, masks)
    assert result["nan_detected"] is True

    # The reference sees only the two clean minibatches, in the same order.
    clean = slice(0, 64)
    reference.update(
        states[clean], actions[clean], old_lp[clean], returns[clean],
        advs[clean], masks[clean],
    )
    for (name, param), (_, expected) in zip(
        agent.model.named_parameters(), reference.model.named_parameters()
    ):
        torch.testing.assert_close(param, expected, rtol=0, atol=0, msg=name)
    for param, expected in zip(
        agent.model.parameters(), reference.model.parameters()
    ):
        state = agent.optimizer.state[param]
        expected_state = reference.optimizer.state[expected]
        assert float(state["step"]) == float(expected_state["step"]) == 2.0
        torch.testing.assert_close(
            state["exp_avg"], expected_state["exp_avg"], rtol=0, atol=0
        )
        torch.testing.assert_close(
            state["exp_avg_sq"], expected_state["exp_avg_sq"], rtol=0, atol=0
        )


def test_nonfinite_first_minibatch_changes_nothing(card_setup):
    agent = _agent(card_setup, "cuda")
    before = copy.deepcopy(agent.model.state_dict())
    states, actions, old_lp, returns, advs, masks = _rollout(agent, 64, "cuda")
    result = agent.update(
        states, actions, old_lp, returns, torch.full_like(advs, float("nan")),
        masks,
    )
    assert result["nan_detected"] is True
    for name, value in agent.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
    for state in agent.optimizer.state.values():
        assert float(state["step"]) == 0.0


def test_unfused_optimizer_state_loads_into_the_fused_optimizer(
    card_setup, tmp_path
):
    from src.ai.ppo_agent import PPOAgent
    from src.config import save_checkpoint

    cpu_agent = _agent(card_setup, "cpu")
    cpu_agent.update(*_rollout(cpu_agent, 64, "cpu"))
    assert cpu_agent.optimizer.param_groups[0]["fused"] in (None, False)
    path = tmp_path / "unfused.pth"
    save_checkpoint(
        str(path),
        cpu_agent.model.state_dict(),
        ppo_config=cpu_agent.ppo_config,
        model_config=cpu_agent.model_config,
        optimizer_state_dict=cpu_agent.optimizer.state_dict(),
    )

    resumed = PPOAgent(
        "PPO",
        card_setup.card_names,
        ppo_config=PPOConfig(epochs=1, batch_size=32),
        device_config=DeviceConfig(main_device="cuda", simulation_device="cuda"),
        registry=card_setup,
        model_path=str(path),
        load_optimizer_state=True,
    )
    group = resumed.optimizer.param_groups[0]
    assert group["fused"] is True
    for param in resumed.model.parameters():
        step = resumed.optimizer.state[param]["step"]
        assert step.device.type == "cuda"
        assert float(step) == 2.0
    assert resumed.update(*_rollout(resumed, 64, "cuda"))["nan_detected"] is False


def test_captured_minibatches_match_eager_minibatches(card_setup, monkeypatch):
    from src.ai.ppo_agent import PPOAgent

    _in_order(monkeypatch)
    graphed = _agent(card_setup, "cuda", epochs=2)
    eager = copy.deepcopy(graphed)
    monkeypatch.setattr(
        eager, "_minibatch_graph", lambda *args, **kwargs: None
    )
    data = _rollout(graphed, 32 * 3 + 7, "cuda")  # three full minibatches + 7
    for _ in range(2):
        graphed_metrics = graphed.update(*data)
        eager_metrics = eager.update(*data)
    assert graphed._update_graph is not None
    assert getattr(eager, "_update_graph", None) is None
    for name, param in graphed.model.named_parameters():
        expected = dict(eager.model.named_parameters())[name]
        torch.testing.assert_close(param, expected, rtol=1e-6, atol=1e-7, msg=name)
    for key, value in graphed_metrics.items():
        if key == "nan_detected":
            assert value is eager_metrics[key] is False
        else:
            assert value == pytest.approx(eager_metrics[key], rel=1e-5, abs=1e-7), key
    for param in graphed.model.parameters():
        assert float(graphed.optimizer.state[param]["step"]) == 16.0
