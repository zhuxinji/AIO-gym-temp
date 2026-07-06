from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(ROOT, "tmp", "matplotlib"))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import matplotlib.pyplot as plt

from aiogym.env import AIOGymNativeEnv
from aiogym.control.baselines import make_meas
from aiogym.control.mpc_grid import FastGridCSTRMPC


def run_episode(seed: int, steps: int = 320, sp_step_at: int = 160):
    env = AIOGymNativeEnv(
        "cstr",
        reward_mode="track",
        control_dt=0.5,
        episode_steps=steps,
        dynamic=True,
        randomize=True,
        randomize_setpoints=False,
        action_mode="actuator",
        randomize_plant=True,
        plant_drift=True,
        integral_obs=False,
        terminate_on_runaway=False,
    )
    agent = FastGridCSTRMPC(horizon=4, grid=51, r_move=0.0, pump=0.5, nsub=2)
    env.reset(seed=seed)
    agent.reset()

    ca, temp, t_sp, cooling = [], [], [], []
    for k in range(steps):
        env.t_sp[0] = 60.0 if k < sp_step_at else 75.0
        sp = {"h_sp": env.h_sp, "t_sp": env.t_sp}
        act = agent.compute(make_meas(env), sp, env.control_dt)

        x = list(env.integ.x)
        ca.append(float(x[0]))
        temp.append(float(x[1]))
        t_sp.append(float(env.t_sp[0]))
        cooling.append(float(act["heaters"][0]))

        action = np.array(
            list(act["pumps"]) + list(act["valves"]) + list(act["heaters"]),
            dtype=np.float32,
        )
        _, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    return {
        "time": np.arange(len(ca), dtype=float) * env.control_dt / 60.0,
        "ca": np.asarray(ca),
        "temp": np.asarray(temp),
        "t_sp": np.asarray(t_sp),
        "cooling": np.asarray(cooling),
    }


def stack_runs(runs, key: str):
    min_len = min(len(r[key]) for r in runs)
    return np.vstack([r[key][:min_len] for r in runs])


def plot(out_path: str):
    runs = [run_episode(seed) for seed in range(10)]
    time = runs[0]["time"][: min(len(r["time"]) for r in runs)]

    ca = stack_runs(runs, "ca")
    temp = stack_runs(runs, "temp")
    t_sp = stack_runs(runs, "t_sp")
    cooling = stack_runs(runs, "cooling")

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.2), sharex=True)
    red = "#d62728"
    shade = "#f2a3a3"

    series = [
        (ca, None, "Ca", "Ca (Tracking MPC)"),
        (temp, t_sp[0], "T (deg C)", "T (Tracking MPC)"),
        (cooling, None, "cooling u", "Cooling u (Tracking MPC)"),
    ]

    for ax, (data, ref, ylabel, label) in zip(axes, series):
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(time, mean, color=red, lw=2.8, label=label)
        ax.fill_between(time, mean - std, mean + std, color=shade, alpha=0.35, lw=0)
        if ref is not None:
            ax.step(time, ref[: len(time)], where="post", color="black", ls="--", lw=1.6, label="Set Point")
        ax.set_ylabel(ylabel)
        ax.grid(True, color="#b0b0b0", lw=0.8)
        ax.legend(loc="best", frameon=True)

    axes[-1].set_xlabel("Time (min)")
    axes[0].set_title("CSTR Tracking MPC Rollout and Plotting", loc="left", fontsize=15, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")


if __name__ == "__main__":
    output = os.path.join(ROOT, "output", "cstr_tracking_mpc_rollout.png")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    plot(output)
    print(output)
