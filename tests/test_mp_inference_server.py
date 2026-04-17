"""Tests for the multi-model InferenceServer control plane and error path.

These tests exercise register_model / unregister_model and the unknown-
model_id error response without spawning worker processes — they drive
the request_queue / response_queues directly from the main thread.
"""
import time

import numpy as np
import pytest
import torch

from src.config import ModelConfig
from src.encoding.action_encoder import get_action_space_size
from src.ppo.mp_inference_server import (
    InferenceServer,
    InferenceRequest,
    InferenceResponse,
)
from src.ppo.ppo_actor_critic import PPOActorCritic


@pytest.fixture
def dims():
    card_names = ["card_a", "card_b", "card_c"]
    num_cards = len(card_names)
    from src.encoding.state_encoder import get_state_size
    state_dim = get_state_size(card_names)
    action_dim = get_action_space_size(card_names)
    return state_dim, action_dim, num_cards


def _make_model(dims):
    state_dim, action_dim, num_cards = dims
    m = PPOActorCritic(
        state_dim=state_dim, action_dim=action_dim, num_cards=num_cards,
        model_config=ModelConfig(),
    )
    m.eval()
    return m


def _make_request(worker_id, request_id, dims, model_id, batch=2):
    state_dim, action_dim, _ = dims
    states = np.zeros((batch, state_dim), dtype=np.float32)
    masks = np.ones((batch, action_dim), dtype=np.uint8)
    return InferenceRequest(
        worker_id=worker_id, request_id=request_id,
        states=states, masks=masks, model_id=model_id,
    )


def _start_server(models, num_workers=1):
    # Use the default fork-ish context — these tests don't spawn workers,
    # they just drive the response queues directly from the main thread.
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    server = InferenceServer(
        models=models, device=torch.device("cpu"),
        num_workers=num_workers, ctx=ctx,
    )
    server.start()
    return server


def test_unknown_model_id_returns_error(dims):
    server = _start_server({"a": _make_model(dims)})
    try:
        server.request_queue.put(_make_request(0, 1, dims, "does-not-exist"))
        resp: InferenceResponse = server.response_queues[0].get(timeout=5.0)
        assert resp.error is not None
        assert "does-not-exist" in resp.error
        assert resp.request_id == 1
    finally:
        server.stop()


def test_known_model_id_returns_no_error(dims):
    server = _start_server({"a": _make_model(dims)})
    try:
        server.request_queue.put(_make_request(0, 7, dims, "a"))
        resp = server.response_queues[0].get(timeout=5.0)
        assert resp.error is None
        assert resp.request_id == 7
        assert resp.action_indices.shape == (2,)
    finally:
        server.stop()


def test_register_unregister_round_trip(dims):
    server = _start_server({"a": _make_model(dims)})
    try:
        # Newly registered model should now answer requests.
        server.register_model("b", _make_model(dims))
        server.request_queue.put(_make_request(0, 1, dims, "b"))
        resp = server.response_queues[0].get(timeout=5.0)
        assert resp.error is None

        # After unregister, requests for that id should fail fast.
        server.unregister_model("b")
        server.request_queue.put(_make_request(0, 2, dims, "b"))
        resp = server.response_queues[0].get(timeout=5.0)
        assert resp.error is not None
        assert "b" in resp.error
    finally:
        server.stop()


def test_duplicate_register_raises(dims):
    server = _start_server({"a": _make_model(dims)})
    try:
        with pytest.raises(ValueError, match="already registered"):
            server.register_model("a", _make_model(dims))
    finally:
        server.stop()


def test_unregister_unknown_raises(dims):
    server = _start_server({"a": _make_model(dims)})
    try:
        with pytest.raises(ValueError, match="not registered"):
            server.unregister_model("ghost")
    finally:
        server.stop()


def test_register_before_start_fails(dims):
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    server = InferenceServer(
        models={"a": _make_model(dims)},
        device=torch.device("cpu"), num_workers=1, ctx=ctx,
    )
    with pytest.raises(RuntimeError, match="not running"):
        server.register_model("b", _make_model(dims))


def test_empty_models_rejected(dims):
    with pytest.raises(ValueError, match="at least one"):
        InferenceServer(
            models={}, device=torch.device("cpu"), num_workers=1,
        )


def test_mixed_model_batch_routes_correctly(dims):
    """Two requests with different model_ids in one drain cycle should
    both succeed and each be served by its own model.
    """
    server = _start_server({"a": _make_model(dims), "b": _make_model(dims)}, num_workers=2)
    try:
        server.request_queue.put(_make_request(0, 10, dims, "a"))
        server.request_queue.put(_make_request(1, 20, dims, "b"))
        r0 = server.response_queues[0].get(timeout=5.0)
        r1 = server.response_queues[1].get(timeout=5.0)
        assert r0.error is None and r1.error is None
        assert r0.request_id == 10 and r1.request_id == 20
    finally:
        server.stop()
