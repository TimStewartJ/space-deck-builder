"""GPU and host time of one decision step: merged slots versus separate row counts."""
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
def gpu_ms(fn, reps=400):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    t = time.perf_counter(); start.record()
    for _ in range(reps): fn()
    end.record(); host = (time.perf_counter() - t) / reps * 1e3
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps, host
for n0, n_opp in ((448, 80), (448, 220), (224, 60), (96, 30)):
    b = merged.bucket(max(n0, n_opp))
    g_merged = merged._graph(b, 4)
    g_learner = learner._graph(learner.bucket(n0), 1)
    g_opp = opponents._graph(opponents.bucket(n_opp), 4)
    m_gpu, m_host = gpu_ms(lambda: g_merged.replay())
    s_gpu, s_host = gpu_ms(lambda: (g_learner.replay(), g_opp.replay()))
    print(f"learner {n0:4d}, opponents 3x{n_opp:4d}: merged (4x{b}) GPU {m_gpu:.3f} ms host {m_host:.3f} ms | separate GPU {s_gpu:.3f} ms host {s_host:.3f} ms", flush=True)
