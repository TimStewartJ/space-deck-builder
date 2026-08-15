"""Correctness tests for the vector-engine primitive benchmark.

These use tiny populations. They validate that the benchmark measures the
same deterministic work in every backend; they do not measure throughput.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from scripts import bench_vec_primitives as bench


def test_real_dimensions_are_locked():
    assert bench.NUM_CARDS == 49
    assert bench.ACTION_DIM == 494
    assert bench.NUM_PLAYERS == 2
    assert bench.NUM_ZONES == 5


def test_state_shapes_and_dtypes():
    st = bench.VecState.create(17, seed=4)
    assert st.zones.shape == (17, 2, 5, 49)
    assert st.zones.dtype == np.int8
    assert st.scalars.shape == (17, 2, 8)
    assert st.scalars.dtype == np.int16
    assert st.trade_row.shape == (17, 5)
    assert st.trade_deck.shape == (17, 49)
    assert st.mode.shape == (17,)
    assert st.rng_state.dtype == np.int64


def test_state_copy_does_not_alias():
    original = bench.VecState.create(8, seed=1)
    copied = original.copy()
    copied.zones[0, 0, 0, 0] += 1
    copied.cost[0] += 1
    assert not np.array_equal(copied.zones, original.zones)
    assert not np.array_equal(copied.cost, original.cost)


def test_action_schedule_exercises_every_main_branch():
    st = bench.VecState.create(25, seed=2)
    actions = bench.make_action_schedule(st, 5)
    flat = set(actions.reshape(-1).tolist())
    assert 0 in flat
    assert bench.PLAY_OFFSET in flat
    assert bench.PLAY_OFFSET + 2 in flat
    assert bench.ATTACK_PLAYER_INDEX in flat
    assert any(
        bench.BUY_OFFSET <= action < bench.BUY_OFFSET + bench.NUM_CARDS
        for action in flat
    )


def test_intended_workload_keeps_all_expensive_branches_live():
    base = bench.VecState.create(256, seed=50_000)
    _, profile = bench.build_workload(base, 200)
    bench.validate_workload_profile(profile)
    assert profile.mode_1 > 0
    assert profile.mode_2 > 0
    assert profile.mode_3 > 0
    assert profile.mode_4 > 0
    assert profile.resets > 0


def test_numpy_backend_produces_valid_state_and_mask():
    base = bench.VecState.create(41, seed=3)
    actions = bench.make_action_schedule(base, 19)
    elapsed, snapshot = bench.execute_backend(
        "numpy", base, actions, timed=True
    )
    assert elapsed > 0
    bench.assert_snapshot_invariants(snapshot)
    assert snapshot.digest()


def test_verification_captures_every_pre_step_mask_and_rebuilds_final_mask():
    base = bench.VecState.create(37, seed=14)
    actions = bench.make_action_schedule(base, 19)
    _, snapshot = bench.execute_backend(
        "numpy",
        base,
        actions,
        timed=False,
        capture_mask_trace=True,
    )
    assert snapshot.mask_trace is not None
    assert snapshot.mask_trace.shape == (19, 37, bench.ACTION_DIM)

    reference = base.copy()
    expected = np.zeros((37, bench.ACTION_DIM), dtype=np.bool_)
    for step, action in enumerate(actions):
        bench.build_mask_numpy(reference, expected)
        assert np.array_equal(snapshot.mask_trace[step], expected)
        bench.step_numpy(reference, action)
    bench.build_mask_numpy(reference, expected)
    assert np.array_equal(snapshot.mask, expected)


def test_numpy_draw_is_deterministic_and_conserves_personal_cards():
    a = bench.VecState.create(13, seed=7)
    b = a.copy()
    idx = np.arange(a.games)
    before = a.zones.sum(axis=(2, 3)).copy()

    bench._draw_numpy(a, idx, 5)
    bench._draw_numpy(b, idx, 5)

    assert np.array_equal(a.zones, b.zones)
    assert np.array_equal(a.rng_state, b.rng_state)
    assert np.array_equal(a.zones.sum(axis=(2, 3)), before)


def test_numpy_mask_has_legal_action_for_every_game():
    st = bench.VecState.create(37, seed=9)
    mask = np.zeros((st.games, bench.ACTION_DIM), dtype=np.bool_)
    bench.build_mask_numpy(st, mask)
    assert mask.any(axis=1).all()
    assert mask[st.mode == 0, 0].all()
    assert mask[st.mode != 0, 1].all()


def test_numpy_and_torch_cpu_are_exact():
    pytest.importorskip("torch")
    digest = bench.verify_backends_agree(
        games=43,
        steps=17,
        backends=["numpy", "torch-cpu"],
    )
    assert digest


def test_torch_cpu_does_not_alias_numpy_source_state():
    pytest.importorskip("torch")
    base = bench.VecState.create(13, seed=5)
    before = base.copy()
    actions = bench.make_action_schedule(base, 5)
    bench.execute_backend("torch-cpu", base, actions, timed=False)
    for name in bench.MUTABLE_FIELDS:
        assert np.array_equal(getattr(base, name), getattr(before, name)), name


def test_numpy_and_numba_are_exact():
    pytest.importorskip("numba")
    digest = bench.verify_backends_agree(
        games=43,
        steps=17,
        backends=["numpy", "numba"],
    )
    assert digest


@pytest.mark.skipif(
    not bench.backend_availability("torch-cuda")[0],
    reason="CUDA/ROCm device unavailable",
)
def test_numpy_and_torch_cuda_are_exact():
    digest = bench.verify_backends_agree(
        games=43,
        steps=17,
        backends=["numpy", "torch-cuda"],
    )
    assert digest


def test_full_available_backend_equivalence():
    available = [
        name
        for name in bench.KNOWN_BACKENDS
        if bench.backend_availability(name)[0]
    ]
    digest = bench.verify_backends_agree(
        games=31,
        steps=11,
        backends=available,
    )
    assert digest


def test_snapshot_comparison_reports_first_divergence():
    expected = np.zeros((2, 3), dtype=np.int8)
    actual = expected.copy()
    actual[1, 2] = 1
    with pytest.raises(AssertionError, match=r"diverged at \(1, 2\)"):
        bench._assert_array_equal("bad", "zones", expected, actual)


def test_digest_detects_same_sum_permutations():
    base = bench.VecState.create(2, seed=1)
    actions = bench.make_action_schedule(base, 1)
    _, snapshot = bench.execute_backend("numpy", base, actions, timed=False)
    arrays = {name: value.copy() for name, value in snapshot.arrays.items()}
    zones = arrays["zones"]
    left = (0, 0, bench.ZONE_HAND, 0)
    right = (1, 0, bench.ZONE_HAND, 0)
    zones[left] -= 1
    zones[right] += 1
    altered = bench.Snapshot(arrays=arrays, mask=snapshot.mask.copy())
    assert zones.sum() == snapshot.arrays["zones"].sum()
    assert altered.digest() != snapshot.digest()


def test_interleaving_rotates_backend_order():
    names = ["a", "b", "c"]
    assert bench._rotated_order(names, 0) == ["a", "b", "c"]
    assert bench._rotated_order(names, 1) == ["b", "c", "a"]
    assert bench._rotated_order(names, 2) == ["c", "a", "b"]
    assert bench._rotated_order(names, 3) == ["c", "b", "a"]


def test_tiny_benchmark_runs_end_to_end():
    results = bench.run_benchmarks(
        games=32,
        steps=4,
        repetitions=2,
        backends=["numpy"],
    )
    result = results["numpy"]
    assert result.decisions_per_sec > 0
    assert result.step_ms > 0
    assert len(result.repetitions) == 2


def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="unknown backend"):
        bench.execute_backend(
            "imaginary",
            bench.VecState.create(2),
            np.zeros((1, 2), dtype=np.int32),
            timed=False,
        )


def test_cli_verify_only(tmp_path):
    # Verify-only intentionally does not write benchmark JSON because there
    # are no timings to serialize.
    rc = bench.main(
        [
            "--games",
            "16",
            "--steps",
            "3",
            "--backends",
            "numpy,torch-cpu",
            "--verify-only",
            "--json-out",
            str(tmp_path / "unused.json"),
        ]
    )
    assert rc == 0
    assert not (tmp_path / "unused.json").exists()


def test_result_payload_is_json_serializable():
    args = bench.parse_args(["--games", "8", "--steps", "2", "--reps", "1"])
    aggregate = bench.AggregateResult(
        backend="numpy",
        decisions_per_sec=1234.5,
        step_ms=1.0,
        digest="77",
        repetitions=(1234.5,),
    )
    payload = bench._result_payload(
        args,
        {"numpy": aggregate},
        {"numba": "not installed"},
        verification_digest="99",
        workload_profile=bench.WorkloadProfile(decisions=1, end_turns=1),
    )
    encoded = json.dumps(payload)
    assert '"verification_digest": "99"' in encoded
    assert payload["results"]["numpy"]["gate_eligible"] is True
    assert payload["results"]["numpy"]["execution_model"] == "array-simd"


def test_torch_json_result_is_explicitly_ineligible_for_gate():
    args = bench.parse_args(["--games", "8", "--steps", "2", "--reps", "1"])
    aggregate = bench.AggregateResult(
        backend="torch-cuda",
        decisions_per_sec=9_999_999,
        step_ms=1.0,
        digest="abc",
        repetitions=(9_999_999,),
    )
    payload = bench._result_payload(
        args,
        {"torch-cuda": aggregate},
        {},
        verification_digest="verified",
        workload_profile=bench.WorkloadProfile(decisions=1, end_turns=1),
    )
    result = payload["results"]["torch-cuda"]
    assert result["execution_model"] == "eager"
    assert result["gate_eligible"] is False
    assert result["gate_verdict"] == "EAGER ONLY"
