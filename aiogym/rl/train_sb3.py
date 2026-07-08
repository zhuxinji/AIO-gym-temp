#!/usr/bin/env python3
"""Train, export, and evaluate Stable-Baselines3 baselines for AIO-Gym.

Rollout parallelism uses SubprocVecEnv by default: one plant per worker process.
For these small MLP policies, CPU is the default on Apple machines because MPS
overhead is usually larger than the network compute. CUDA is still used when
available.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from datetime import datetime, timezone

from aiogym.controllers import make_controller
from aiogym.env import AIOGymNativeEnv
from aiogym.evaluation import BenchmarkProtocol, evaluate_controller, rollout_controller
from aiogym.rl.artifacts import (
    learning_curve_point,
    result_row,
    rl_payload,
    write_rl_artifacts,
)


def protocol_factory(objective: str):
    return {
        "economic": BenchmarkProtocol.economic,
        "tracking": BenchmarkProtocol.tracking,
        "robustness": BenchmarkProtocol.robustness,
        "safety": BenchmarkProtocol.safety,
        "kpi": BenchmarkProtocol.kpi,
    }[objective]


def parse_seed_list(raw: str | None, seed: int, episodes: int):
    if not raw:
        return [seed + i for i in range(episodes)]
    seeds = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--eval-seed-list must contain at least one integer seed")
    return seeds


def make_training_env(args, rank: int = 0):
    def _init():
        env = AIOGymNativeEnv(
            args.scenario,
            reward_mode=args.reward_mode,
            action_mode=args.action_mode,
            control_dt=args.control_dt,
            episode_steps=args.train_episode_steps,
            dynamic=args.dynamic,
            randomize=args.randomize,
            randomize_setpoints=args.randomize_setpoints,
            randomize_plant=args.randomize_plant,
            plant_drift=args.plant_drift,
            integral_obs=args.integral_obs,
            terminate_on_runaway=args.terminate_on_runaway,
            noise=args.noise,
            noise_pct=args.noise_pct,
        )
        env.reset(seed=args.seed + rank)
        return env

    return _init


def default_n_envs():
    return min(16, max(1, (os.cpu_count() or 4) - 2))


def best_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_algo(args, env):
    try:
        from stable_baselines3 import PPO, SAC, TD3
    except ModuleNotFoundError as ex:
        raise SystemExit(
            "stable-baselines3 is not installed. Install the optional RL dependencies "
            "from aiogym/requirements.txt before running this script."
        ) from ex

    algo = args.algo.lower()
    common = dict(
        policy="MlpPolicy",
        env=env,
        verbose=args.verbose,
        seed=args.seed,
        device=args.device,
        learning_rate=args.learning_rate,
        tensorboard_log=args.tensorboard_log,
    )
    if algo == "sac":
        return SAC(
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            **common,
        )
    if algo == "td3":
        return TD3(
            batch_size=args.batch_size,
            train_freq=args.train_freq,
            gradient_steps=args.gradient_steps,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            **common,
        )
    if algo == "ppo":
        return PPO(
            n_steps=args.ppo_n_steps,
            batch_size=args.batch_size,
            **common,
        )
    raise ValueError(f"unsupported SB3 algorithm: {args.algo}")


def evaluate_checkpoint(args, checkpoint_path: str):
    protocol = protocol_factory(args.eval_objective)(
        args.scenario,
        action_mode=args.action_mode,
        episode_steps=args.eval_episode_steps,
        control_dt=args.control_dt,
    )
    controller = make_controller(
        "sb3",
        scenario=args.scenario,
        config={
            "path": checkpoint_path,
            "algo": args.algo,
            "action_mode": args.action_mode,
        },
    )
    seeds = parse_seed_list(args.eval_seed_list, args.eval_seed, args.eval_episodes)
    result = evaluate_controller(
        controller,
        protocol.make_env(),
        episodes=len(seeds),
        seed=seeds[0],
        seed_list=seeds,
        protocol=protocol,
        include_episodes=True,
    )
    rollout = None
    if args.save_rollout:
        rollout = rollout_controller(
            controller,
            protocol.make_env(),
            seed=seeds[0],
            max_steps=args.rollout_steps,
            protocol=protocol,
        )
    return protocol, result, rollout


def evaluate_training_policy(args, model, step: int, phase: str = "eval"):
    protocol = protocol_factory(args.eval_objective)(
        args.scenario,
        action_mode=args.action_mode,
        episode_steps=args.eval_episode_steps,
        control_dt=args.control_dt,
    )
    controller = make_controller(
        "sb3",
        scenario=args.scenario,
        policy=model,
        config={
            "algo": args.algo,
            "action_mode": args.action_mode,
            "name": f"SB3-{args.algo.upper()}",
        },
    )
    seeds = parse_seed_list(args.eval_seed_list, args.eval_seed, args.learning_curve_episodes)
    env = protocol.make_env()
    try:
        result = evaluate_controller(
            controller,
            env,
            episodes=len(seeds),
            seed=seeds[0],
            seed_list=seeds,
            protocol=protocol,
            include_episodes=False,
        )
    finally:
        if hasattr(env, "close"):
            env.close()
    return learning_curve_point(step, result, phase=phase)


def training_metadata(args, checkpoint_path: str):
    return {
        "algo": args.algo,
        "scenario": args.scenario,
        "action_mode": args.action_mode,
        "reward_mode": args.reward_mode,
        "total_timesteps": args.steps,
        "seed": args.seed,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env,
        "subproc_start_method": args.subproc_start_method if args.vec_env == "subproc" else None,
        "device": args.device,
        "torch_threads": args.torch_threads,
        "train_episode_steps": args.train_episode_steps,
        "env_kwargs": {
            "dynamic": args.dynamic,
            "randomize": args.randomize,
            "randomize_setpoints": args.randomize_setpoints,
            "randomize_plant": args.randomize_plant,
            "plant_drift": args.plant_drift,
            "integral_obs": args.integral_obs,
            "terminate_on_runaway": args.terminate_on_runaway,
            "noise": args.noise,
            "noise_pct": args.noise_pct,
            "control_dt": args.control_dt,
        },
        "checkpoint_path": checkpoint_path,
    }


def artifact_dir_for(args, run_name: str) -> str:
    return args.artifact_dir or os.path.join(args.out_dir, f"{run_name}_artifacts")


def make_learning_curve_callback(args):
    from stable_baselines3.common.callbacks import BaseCallback

    class LearningCurveCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history = []
            self._next_eval = max(1, int(args.learning_curve_every))

        def _on_step(self) -> bool:
            if args.learning_curve_every <= 0 or self.num_timesteps < self._next_eval:
                return True
            row = evaluate_training_policy(args, self.model, self.num_timesteps)
            row["timesteps"] = self.num_timesteps
            self.history.append(row)
            self._next_eval += max(1, int(args.learning_curve_every))
            return True

    return LearningCurveCallback()


def export_onnx(model, obs_dim: int, path: str):
    import torch

    class DeterministicPolicy(torch.nn.Module):
        def __init__(self, policy):
            super().__init__()
            self.policy = policy

        def forward(self, obs):
            return self.policy._predict(obs, deterministic=True)

    policy = model.policy
    policy.eval()
    device = next(policy.parameters()).device
    dummy = torch.zeros(1, obs_dim, device=device)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.onnx.export(
        DeterministicPolicy(policy),
        dummy,
        path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
    )


def require_onnx_export_dependencies():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError as ex:
        raise SystemExit(
            "torch is required for ONNX export. Install the optional export dependencies "
            "with `pip install -e .[export]` or install torch/onnx manually."
        ) from ex
    if importlib.util.find_spec("onnx") is None:
        raise SystemExit(
            "onnx is required for ONNX export. Install it with `pip install -e .[export]` "
            "or install `onnx` in the active environment."
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cstr")
    ap.add_argument("--algo", default="sac", choices=["sac", "ppo", "td3"])
    ap.add_argument("--action-mode", default="actuator", choices=["actuator", "setpoint"])
    ap.add_argument("--reward-mode", default="kpi", choices=["kpi", "economic", "track"])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--n-envs", type=int, default=default_n_envs())
    ap.add_argument("--vec-env", default="subproc", choices=["subproc", "dummy"],
                    help="parallel rollout backend; subproc gives one process per env")
    ap.add_argument("--subproc-start-method", default="fork", choices=["fork", "forkserver", "spawn"],
                    help="multiprocessing start method for SubprocVecEnv")
    ap.add_argument("--train-episode-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--control-dt", type=float, default=0.5)
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--randomize-setpoints", action="store_true")
    ap.add_argument("--randomize-plant", action="store_true")
    ap.add_argument("--plant-drift", action="store_true")
    ap.add_argument("--integral-obs", action="store_true")
    ap.add_argument("--terminate-on-runaway", action="store_true")
    ap.add_argument("--noise", action="store_true")
    ap.add_argument("--noise-pct", type=float, default=0.01)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--buffer-size", type=int, default=300000)
    ap.add_argument("--learning-starts", type=int, default=100)
    ap.add_argument("--train-freq", type=int, default=1)
    ap.add_argument("--gradient-steps", type=int, default=1)
    ap.add_argument("--ppo-n-steps", type=int, default=2048)
    ap.add_argument("--device", default=None,
                    help="SB3 policy device; defaults to CUDA if available, otherwise CPU. Use 'mps' explicitly if desired.")
    ap.add_argument("--torch-threads", type=int, default=2,
                    help="torch intra-op threads; keep low so SubprocVecEnv workers get CPU time")
    ap.add_argument("--verbose", type=int, default=1)
    ap.add_argument("--tensorboard-log", default=None)
    ap.add_argument("--out-dir", default="aiogym/runs/rl/sb3")
    ap.add_argument("--name", default=None)
    ap.add_argument("--artifact-dir", default=None,
                    help="standard benchmark artifact directory; defaults to <out-dir>/<name>_artifacts")
    ap.add_argument("--eval-objective", default="kpi", choices=["economic", "tracking", "robustness", "safety", "kpi"])
    ap.add_argument("--eval-episodes", type=int, default=3)
    ap.add_argument("--eval-episode-steps", type=int, default=80)
    ap.add_argument("--eval-seed", type=int, default=9000)
    ap.add_argument("--eval-seed-list", default=None)
    ap.add_argument("--learning-curve-every", type=int, default=0,
                    help="evaluate the in-memory policy every N timesteps; 0 records only the final point")
    ap.add_argument("--learning-curve-episodes", type=int, default=1)
    ap.add_argument("--save-rollout", action="store_true")
    ap.add_argument("--rollout-steps", type=int, default=None)
    ap.add_argument("--onnx", action="store_true", help="export deterministic policy to ONNX after training")
    ap.add_argument("--onnx-path", default=None, help="optional ONNX export path; defaults to checkpoint basename + .onnx")
    args = ap.parse_args()
    if args.onnx:
        require_onnx_export_dependencies()

    try:
        from stable_baselines3.common.env_util import make_vec_env
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        import torch
    except ModuleNotFoundError as ex:
        raise SystemExit(
            "stable-baselines3 is not installed. Install the optional RL dependencies "
            "from aiogym/requirements.txt before running this script."
        ) from ex
    args.device = args.device or best_device()
    torch.set_num_threads(max(1, int(args.torch_threads)))

    os.makedirs(args.out_dir, exist_ok=True)
    run_name = args.name or f"{args.algo}_{args.scenario}_{args.action_mode}_{args.reward_mode}_seed{args.seed}"
    checkpoint_path = os.path.join(args.out_dir, run_name)
    report_path = os.path.join(args.out_dir, f"{run_name}_report.json")

    vec_env_cls = SubprocVecEnv if args.vec_env == "subproc" else DummyVecEnv
    vec_env_kwargs = {"start_method": args.subproc_start_method} if args.vec_env == "subproc" else None
    env = make_vec_env(
        make_training_env(args),
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env_cls=vec_env_cls,
        vec_env_kwargs=vec_env_kwargs,
    )
    model = build_algo(args, env)
    print(
        f"training {args.algo.upper()} | {args.n_envs} {args.vec_env} envs | "
        f"device={args.device} | action={args.action_mode} | reward={args.reward_mode}"
    )
    t0 = time.time()
    curve_callback = make_learning_curve_callback(args)
    model.learn(total_timesteps=args.steps, progress_bar=False, callback=curve_callback)
    train_seconds = time.time() - t0
    model.save(checkpoint_path)
    checkpoint_zip = f"{checkpoint_path}.zip"
    onnx_path = None
    if args.onnx:
        onnx_path = args.onnx_path or f"{checkpoint_path}.onnx"
        export_onnx(model, env.observation_space.shape[0], onnx_path)
    env.close()

    protocol, result, rollout = evaluate_checkpoint(args, checkpoint_zip)
    training = training_metadata(args, checkpoint_zip)
    if onnx_path is not None:
        training["onnx_path"] = onnx_path
    training["legacy_report_path"] = report_path
    learning_curve = list(curve_callback.history)
    final_curve_point = learning_curve_point(args.steps, result, phase="final")
    final_curve_point["timesteps"] = args.steps
    learning_curve.append(final_curve_point)
    artifact_payload = rl_payload(
        kind="sb3_train_eval",
        scenario=args.scenario,
        objective=protocol.objective,
        action_mode=args.action_mode,
        training=training,
        protocol=protocol.metadata(),
        results=[result],
        rows=[result_row(
            result,
            scenario=args.scenario,
            action_mode=args.action_mode,
            controller=f"SB3-{args.algo.upper()}",
            suite_case=f"{protocol.objective}:{args.scenario}:sb3_{args.algo}",
        )],
        learning_curve=learning_curve,
        rollouts=[rollout] if rollout is not None else [],
        extra={
            "training_runtime": {
                "seconds": train_seconds,
                "steps_per_second": args.steps / train_seconds if train_seconds > 0 else None,
                "steps_per_second_per_env": (args.steps / train_seconds / args.n_envs) if train_seconds > 0 else None,
            },
        },
    )
    artifact_payload = write_rl_artifacts(artifact_dir_for(args, run_name), artifact_payload)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": "sb3_train_eval",
        "training": training,
        "training_runtime": {
            "seconds": train_seconds,
            "steps_per_second": args.steps / train_seconds if train_seconds > 0 else None,
            "steps_per_second_per_env": (args.steps / train_seconds / args.n_envs) if train_seconds > 0 else None,
        },
        "evaluation_protocol": protocol.metadata(),
        "evaluation": result,
        "learning_curve": learning_curve,
        "artifact_dir": artifact_payload.get("artifact_dir", artifact_dir_for(args, run_name)),
        "artifacts": artifact_payload.get("artifacts", {}),
    }
    if rollout is not None:
        payload["rollout"] = rollout
    with open(report_path, "w") as f:
        json.dump(payload, f, indent=2)

    metric = result["metric"]
    print(f"saved checkpoint {checkpoint_zip}")
    if onnx_path is not None:
        print(f"exported onnx {onnx_path}")
    print(f"saved report {report_path}")
    print(f"saved artifacts {artifact_dir_for(args, run_name)}")
    if train_seconds > 0:
        print(
            f"train throughput {args.steps / train_seconds:.1f} steps/s "
            f"({args.n_envs} envs x {args.steps / train_seconds / args.n_envs:.1f}/env/s)"
        )
    print(
        f"eval {metric}={result[metric]:.3f} kpi={result['kpi']:.3f} "
        f"profit={result['profit']:.3f} track={result['track']:.3f} "
        f"safety={result['constraint_violation_count']:.1f}"
    )


if __name__ == "__main__":
    main()
