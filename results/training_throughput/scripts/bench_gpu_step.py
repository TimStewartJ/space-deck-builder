"""Pure GPU execution time of one rollout decision step.

Replays are queued behind a spin kernel, so the GPU runs them back to back
after the host has finished submitting: elapsed GPU time minus the spin's own
time is the GPU's work, free of host submission limits.
"""
import time, torch
from src.config import DataConfig
from src.encoding.action_encoder import get_action_space_size
from src.encoding.state_encoder import get_state_size
from src.ppo.ppo_actor_critic import PPOActorCritic
from src.vector_engine.graphed_policy import GraphedRolloutPolicy
data = DataConfig(); cards = data.load_cards(); reg = data.build_registry(cards)
dev = torch.device("cuda")
model = PPOActorCritic(get_state_size(reg.card_names), get_action_space_size(reg.card_names), reg.num_cards).to(dev)
merged = GraphedRolloutPolicy(model, 3, 2000, dev, storage_rows=320000)
learner = GraphedRolloutPolicy(model, 0, 2000, dev, storage_rows=320000)
opponents = GraphedRolloutPolicy(model, 3, 2000, dev)
for p in (merged, learner, opponents):
    p.load_learner(model)
for s in (1, 2, 3):
    merged.load_opponent(s, model); opponents.load_opponent(s, model)

def spin_ms(cycles):
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(); start.record(); torch.cuda._sleep(cycles); end.record(); torch.cuda.synchronize()
    return start.elapsed_time(end)

cycles = 50_000_000
for _ in range(3):
    spin = spin_ms(cycles)
while spin < 150:
    cycles *= 2; spin = spin_ms(cycles)
print(f"spin {cycles} cycles = {spin:.1f} ms", flush=True)

def gpu_ms(fn, reps=200):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record(); torch.cuda._sleep(cycles)
    t = time.perf_counter()
    for _ in range(reps): fn()
    host = (time.perf_counter() - t) * 1e3
    end.record(); torch.cuda.synchronize()
    total = start.elapsed_time(end)
    return (total - spin) / reps, host / reps, host < spin

for n0, n_opp in ((896, 160), (448, 80), (448, 220), (224, 60), (96, 30)):
    b = merged.bucket(max(n0, n_opp))
    g_merged = merged._graph(b, 4)
    g_learner = learner._graph(learner.bucket(n0), 1)
    g_opp = opponents._graph(opponents.bucket(n_opp), 4)
    m_gpu, m_host, m_ok = gpu_ms(lambda: g_merged.replay())
    s_gpu, s_host, s_ok = gpu_ms(lambda: (g_learner.replay(), g_opp.replay()))
    l_gpu, l_host, l_ok = gpu_ms(lambda: g_learner.replay())
    print(f"learner {n0:4d}, opponents 3x{n_opp:4d}: merged 4x{b} GPU {m_gpu:.3f} ms (host {m_host:.3f}) | "
          f"separate GPU {s_gpu:.3f} ms (host {s_host:.3f}) | learner alone GPU {l_gpu:.3f} ms | "
          f"queued in time: {m_ok and s_ok and l_ok}", flush=True)
