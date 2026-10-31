"""PPO update time under the BLAS backend chosen by the environment.

Run once as is and once with TORCH_BLAS_PREFER_HIPBLASLT=0 (rocBLAS);
ALLOW_TF32=1 additionally allows reduced-precision fp32 matrix multiplies.
"""
import os, time, torch
from src.config import DataConfig, DeviceConfig, PPOConfig
from src.ai.ppo_agent import PPOAgent

if os.environ.get("BLAS_API"):
    torch.backends.cuda.preferred_blas_library(os.environ["BLAS_API"])
if os.environ.get("ALLOW_TF32") == "1":
    torch.backends.cuda.matmul.allow_tf32 = True
data = DataConfig(); cards = data.load_cards(); reg = data.build_registry(cards)
dev = torch.device("cuda")
agent = PPOAgent("PPO", reg.card_names, ppo_config=PPOConfig(),
                 device_config=DeviceConfig(main_device="cuda", simulation_device="cuda"), registry=reg)
print(f"blas={torch.backends.cuda.preferred_blas_library()} "
      f"env={os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT')} tf32={torch.backends.cuda.matmul.allow_tf32}", flush=True)
N = 292000
g = torch.Generator(device="cpu").manual_seed(0)
states = (torch.randint(0, 3, (N, agent.state_dim), generator=g).float() * 0.125).to(dev)
masks = (torch.rand(N, agent.action_dim, generator=g) < 0.05).to(dev); masks[:, 1] = True
actions = torch.multinomial(masks.float(), 1).squeeze(1)
old_lp = torch.full((N,), -2.0, device=dev)
returns = torch.rand(N, device=dev) * 2 - 1
advs = torch.randn(N, device=dev)
times = []
for i in range(5):
    torch.cuda.synchronize(); t = time.perf_counter()
    agent.update(states, actions, old_lp, returns, advs, masks)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t)
print("update seconds: " + " ".join(f"{x:.3f}" for x in times) + f" | warm mean {sum(times[2:]) / 3:.3f}", flush=True)

def ev():
    e = torch.cuda.Event(enable_timing=True); e.record(); return e
bs = agent.batch_size
b = torch.randperm(N, device=dev)[:bs]
s, a, olp, R, A, m = states[b], actions[b], old_lp[b], returns[b], advs[b], masks[b]
totals = torch.zeros(len(agent._METRIC_NAMES), device=dev); nonfinite = torch.zeros((), device=dev)
for _ in range(5):
    agent.optimizer.zero_grad(set_to_none=False)
    agent._learn_minibatch(s, a, olp, R, A, m, totals, nonfinite)
torch.cuda.synchronize()
reps = 30
t0 = ev()
for _ in range(reps):
    agent.optimizer.zero_grad(set_to_none=False)
    agent._learn_minibatch(s, a, olp, R, A, m, totals, nonfinite)
t1 = ev(); torch.cuda.synchronize()
print(f"learn step per {bs}-row minibatch: {t0.elapsed_time(t1) / reps:.2f} ms", flush=True)
for name, mod in agent.model.named_modules():
    if isinstance(mod, torch.nn.Linear):
        x = torch.randn(bs, mod.in_features, device=dev)
        for _ in range(5): mod(x)
        torch.cuda.synchronize(); e0 = ev()
        for _ in range(50): mod(x)
        e1 = ev(); torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / 50
        print(f"  {name}: {bs}x{mod.in_features}->{mod.out_features} {ms:.3f} ms "
              f"({2 * bs * mod.in_features * mod.out_features / ms / 1e9:.1f} TFLOP/s)", flush=True)
