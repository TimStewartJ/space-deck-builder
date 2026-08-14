"""Tests for the shared-memory inference transport.

Covers the round trip a rollout worker depends on: create in the parent,
attach in a child, write on one side, read on the other, and release without
leaking the block.
"""
import multiprocessing as mp

import numpy as np
import pytest

from src.ppo.shared_io import SharedInferenceBuffers, SharedIOSpec
from src.ppo.mp_inference_server import InferenceRequest, InferenceResponse


@pytest.fixture
def buffers():
    b = SharedInferenceBuffers.create(
        workers=3, slots=2, max_rows=8, state_size=5, action_dim=4,
    )
    yield b
    b.close()


class TestLayout:
    def test_view_shapes(self, buffers):
        assert buffers.states.shape == (3, 2, 8, 5)
        assert buffers.masks.shape == (3, 2, 8, 4)
        assert buffers.actions.shape == (3, 2, 8)
        assert buffers.log_probs.shape == (3, 2, 8)
        assert buffers.values.shape == (3, 2, 8)

    def test_dtypes_match_the_wire_format(self, buffers):
        # Masks stay uint8 end to end; widening them to float32 was measured
        # as the single largest cost in the old server staging path.
        assert buffers.states.dtype == np.float32
        assert buffers.masks.dtype == np.uint8
        assert buffers.actions.dtype == np.int64
        assert buffers.log_probs.dtype == np.float32
        assert buffers.values.dtype == np.float32

    def test_regions_do_not_alias(self, buffers):
        buffers.states[:] = 1.0
        buffers.masks[:] = 7
        buffers.actions[:] = 11
        buffers.log_probs[:] = 2.0
        buffers.values[:] = 3.0
        assert np.all(buffers.states == 1.0)
        assert np.all(buffers.masks == 7)
        assert np.all(buffers.actions == 11)
        assert np.all(buffers.log_probs == 2.0)
        assert np.all(buffers.values == 3.0)

    def test_slots_are_independent(self, buffers):
        buffers.states[1, 0] = 4.0
        buffers.states[1, 1] = 9.0
        assert np.all(buffers.states[1, 0] == 4.0)
        assert np.all(buffers.states[1, 1] == 9.0)

    def test_nbytes_matches_allocation(self):
        spec = SharedIOSpec(name="", workers=2, slots=2, max_rows=4,
                            state_size=3, action_dim=5)
        expected = (2 * 2 * 4 * 3 * 4) + (2 * 2 * 4 * 5) + (2 * 2 * 4 * 8) \
            + (2 * 2 * 4 * 4) + (2 * 2 * 4 * 4)
        assert spec.nbytes == expected


class TestAttach:
    def test_attach_sees_writes(self, buffers):
        buffers.states[2, 1, 0:3] = np.arange(15).reshape(3, 5)
        other = SharedInferenceBuffers.attach(buffers.spec)
        try:
            assert np.array_equal(
                other.states[2, 1, 0:3], np.arange(15).reshape(3, 5),
            )
        finally:
            other.close()

    def test_writes_from_attached_are_visible(self, buffers):
        other = SharedInferenceBuffers.attach(buffers.spec)
        try:
            other.actions[0, 0, 0:4] = [3, 1, 4, 1]
        finally:
            other.close()
        assert list(buffers.actions[0, 0, 0:4]) == [3, 1, 4, 1]

    def test_attach_does_not_unlink_on_close(self, buffers):
        other = SharedInferenceBuffers.attach(buffers.spec)
        other.close()
        # The creator still owns a live block.
        buffers.states[0, 0, 0, 0] = 42.0
        assert buffers.states[0, 0, 0, 0] == 42.0


def _child(spec, out_q):
    """Attach in a spawned process, echo a transformed row back."""
    buf = SharedInferenceBuffers.attach(spec)
    try:
        out_q.put(float(buf.states[1, 0, 0, 0]))
        buf.values[1, 0, 0] = 99.5
    finally:
        buf.close()


class TestCrossProcess:
    def test_round_trip_through_a_spawned_process(self, buffers):
        buffers.states[1, 0, 0, 0] = 5.25
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_child, args=(buffers.spec, q))
        p.start()
        seen = q.get(timeout=60)
        p.join(timeout=60)
        assert seen == pytest.approx(5.25)
        assert buffers.values[1, 0, 0] == pytest.approx(99.5)


class TestProtocol:
    def test_request_batch_size_prefers_inline_array(self):
        req = InferenceRequest(
            worker_id=0, request_id=1,
            states=np.zeros((6, 3), dtype=np.float32),
            masks=np.zeros((6, 4), dtype=np.uint8),
        )
        assert req.batch_size == 6

    def test_request_batch_size_falls_back_to_row_count(self):
        req = InferenceRequest(worker_id=0, request_id=1, n_rows=12)
        assert req.batch_size == 12

    def test_shared_request_carries_no_arrays(self):
        req = InferenceRequest(
            worker_id=2, request_id=9, slot=1, row_offset=4, n_rows=7,
        )
        assert req.states is None and req.masks is None
        assert (req.slot, req.row_offset, req.n_rows) == (1, 4, 7)

    def test_response_defaults_are_shared_memory_shaped(self):
        resp = InferenceResponse(request_id=3, slot=1, row_offset=2, n_rows=5)
        assert resp.action_indices is None
        assert resp.error is None
        assert (resp.slot, resp.row_offset, resp.n_rows) == (1, 2, 5)
