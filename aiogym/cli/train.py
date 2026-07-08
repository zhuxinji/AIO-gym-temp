#!/usr/bin/env python3
"""Parallel RL training for AIO-Gym — hardware-maximized.

Rollout is parallelized across CPU cores (SubprocVecEnv, one plant per worker) and
the policy network runs on the GPU (Apple MPS / CUDA) automatically. Defaults to the
*control-informed* setpoint action mode (RL sets targets, an inner PID regulates) —
the architecture from Bloor et al.'s "Control-informed RL for chemical processes",
which is the right one here (the plant is always controlled, RL only optimizes).

    # 14 parallel CSTR plants, SAC on MPS, economic objective, supervisory RL
    python -m aiogym.cli.train --scenario cstr --algo sac --n-envs 14 --steps 300000

Benchmarks the trained policy against PID / APC-MPC / the NMPC oracle on exit.
"""
from __future__ import annotations
import argparse
import os
import time


def make_env_fn(scenario, seed, **kw):
    def _f():
        from aiogym.env import AIOGymNativeEnv
        env = AIOGymNativeEnv(scenario, **kw)
        env.reset(seed=seed)
        return env
    return _f


def best_device():
    # For the small MLP policies here, CPU beats Apple MPS by ~3-10x (MPS per-op
    # overhead dominates tiny matmuls); the 16+ parallel CPU env workers are the real
    # accelerator. Use CUDA if present (real GPU), else CPU. Override with --device mps.
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cstr", choices=["cascade", "quadruple", "cstr", "hvac"])
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo"])
    ap.add_argument("--action-mode", default="setpoint", choices=["setpoint", "actuator"])
    ap.add_argument("--reward-mode", default="economic", choices=["economic", "kpi", "track"])
    ap.add_argument("--n-envs", type=int, default=min(16, max(1, (os.cpu_count() or 4) - 2)))
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--episode-steps", type=int, default=400)
    ap.add_argument("--grad-steps", type=int, default=4, help="SAC gradient steps per iter (UTD vs throughput)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    import torch
    from stable_baselines3 import SAC, PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

    torch.set_num_threads(2)                       # leave cores for the env workers
    kw = dict(reward_mode=args.reward_mode, action_mode=args.action_mode,
              episode_steps=args.episode_steps, randomize_plant=True, dynamic=True)
    venv = SubprocVecEnv([make_env_fn(args.scenario, 1000 + i, **kw) for i in range(args.n_envs)])
    venv = VecMonitor(venv)
    dev = args.device or best_device()
    print(f"[train] {args.algo.upper()} | {args.n_envs} parallel {args.scenario} plants | "
          f"device={dev} | action={args.action_mode} | reward={args.reward_mode}")

    if args.algo == "sac":
        model = SAC("MlpPolicy", venv, device=dev, verbose=0, learning_starts=2000,
                    train_freq=1, gradient_steps=args.grad_steps, batch_size=512,
                    policy_kwargs=dict(net_arch=[256, 256]))
    else:
        model = PPO("MlpPolicy", venv, device=dev, verbose=0, n_steps=512, batch_size=2048,
                    policy_kwargs=dict(net_arch=[256, 256]))

    t0 = time.time()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    dt = time.time() - t0
    print(f"[train] {args.steps} steps in {dt:.0f}s = {args.steps/dt:.0f} steps/s "
          f"({args.n_envs} envs x {args.steps/dt/args.n_envs:.0f}/env/s)")
    venv.close()

    out = args.out or f"aiogym/runs/{args.algo}_{args.scenario}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    model.save(out)
    print(f"[train] saved {out}.zip")

    if not args.no_bench:
        benchmark(args.scenario, model, args.reward_mode, args.action_mode, args.episode_steps)


def benchmark(scenario, model, reward_mode, action_mode, episode_steps):
    """Rank the trained policy vs PID / APC-MPC / NMPC oracle on the same env."""
    from aiogym.controllers import make_controller
    from aiogym.evaluation import BenchmarkProtocol, evaluate_controller, primary_metric_for_objective

    protocol_cls = {"economic": BenchmarkProtocol.economic,
                    "track": BenchmarkProtocol.tracking,
                    "kpi": BenchmarkProtocol.kpi}[reward_mode]
    baseline_protocol = protocol_cls(scenario, action_mode="actuator", episode_steps=episode_steps)
    metric = primary_metric_for_objective(baseline_protocol.objective)

    rows = {
        "PID": evaluate_controller(make_controller("pid", scenario=scenario), baseline_protocol.make_env(),
                                   6, protocol=baseline_protocol)[metric],
        "APC-MPC": evaluate_controller(make_controller("mpc", scenario=scenario), baseline_protocol.make_env(),
                                       6, protocol=baseline_protocol)[metric],
    }
    try:
        rows["NMPC-oracle"] = evaluate_controller(
            make_controller("oracle", scenario=scenario, config={"mode": reward_mode}),
            baseline_protocol.make_env(), 3, protocol=baseline_protocol
        )[metric]
    except Exception as e:
        print("[bench] oracle skipped:", e)
    rl_protocol = protocol_cls(scenario, action_mode=action_mode, episode_steps=episode_steps)
    rl_controller = make_controller(
        "policy", scenario=scenario, policy=model,
        config={"action_mode": action_mode, "control_structure": "trained_rl_policy"},
    )
    rl = evaluate_controller(rl_controller, rl_protocol.make_env(), 6, seed=7000, protocol=rl_protocol)
    rows[f"RL({model.__class__.__name__})"] = float(rl[metric])
    print(f"[bench] {scenario} {metric}:  " + " | ".join(f"{k} {v:.0f}" for k, v in rows.items()))


if __name__ == "__main__":
    main()
