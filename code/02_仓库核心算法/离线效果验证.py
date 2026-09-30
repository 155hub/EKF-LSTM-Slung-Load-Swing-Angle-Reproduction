"""Repository-faithful offline EKF versus EKF-LSTM validation.

This script reuses the repository EKF, feature construction, LSTM architecture,
400-sample window, and loss definition.  It deliberately keeps windows lazy so
the highly overlapping training sequences do not need to be materialized in a
multi-gigabyte intermediate file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# Avoid mixing Ubuntu 20.04's older system mpl_toolkits with the project-local
# Matplotlib when this script is run with PYTHONPATH=Python依赖包_Ubuntu20.04_Py38.
if any("Python依赖包_Ubuntu20.04_Py38" in entry for entry in sys.path):
    sys.path[:] = [
        entry
        for entry in sys.path
        if not entry.rstrip("/").endswith("/usr/lib/python3/dist-packages")
    ]
    for module_name in list(sys.modules):
        if module_name == "mpl_toolkits" or module_name.startswith("mpl_toolkits."):
            del sys.modules[module_name]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset

from 吊载扩展卡尔曼滤波 import SlungLoadEKF
from LSTM修正模型 import LSTMCorrectionModel


M_DRONE = 1.5
M_LOAD = 0.5
L_ROPE = 2.0
MAX_THRUST = 23.5
DEFAULT_DT = 0.02
INPUT_DIM = 13
STATE_DIM = 7


@dataclass
class ReplayData:
    inputs: np.ndarray
    targets: np.ndarray
    timestamps: np.ndarray
    stats: Dict[str, int]


class WindowDataset(Dataset):
    """Return stride-one windows without storing duplicated window tensors."""

    def __init__(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        starts: np.ndarray,
        seq_len: int,
    ) -> None:
        self.inputs = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32))
        self.targets = torch.from_numpy(np.ascontiguousarray(targets, dtype=np.float32))
        self.starts = np.asarray(starts, dtype=np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(self.starts.size)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        start = int(self.starts[item])
        stop = start + self.seq_len
        return self.inputs[start:stop], self.targets[start:stop], stop - 1


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train the repository EKF-LSTM and compare it with EKF on the held-out CSV."
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=project_dir / "数据集" / "吊载摆角训练数据集.csv",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=project_dir / "数据集" / "吊载摆角测试数据集.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "测试结果" / "正式离线验证_50轮",
    )
    parser.add_argument("--seq-len", type=int, default=400)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="GPU micro-batch size; gradients are accumulated to the effective batch size.",
    )
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=2048,
        help="Matches the batch size in 源代码/训练LSTM模型.py.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-val-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replay_csv(path: Path) -> ReplayData:
    """Replay one CSV through the repository EKF and feature transform."""

    frame = pd.read_csv(path)
    if len(frame) < 10:
        raise ValueError("CSV is too short: {}".format(path))

    required = {
        "timestamp",
        "cmd_thrust",
        "drone_qx",
        "drone_qy",
        "drone_qz",
        "drone_qw",
        "imu_ax",
        "imu_ay",
        "imu_az",
        "gt_xi",
        "gt_zeta",
        "gt_xi_dot",
        "gt_zeta_dot",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("Missing columns in {}: {}".format(path, ", ".join(missing)))

    frame = frame.copy()
    frame["cmd_thrust"] = frame["cmd_thrust"].replace(0.0, np.nan).bfill().fillna(0.6)

    inputs: List[np.ndarray] = []
    targets: List[np.ndarray] = []
    timestamps: List[float] = []
    ekf = SlungLoadEKF(m_drone=M_DRONE, m_load=M_LOAD, l_rope=L_ROPE)
    last_t = float(frame.iloc[0]["timestamp"])
    reset_count = 0
    divergence_count = 0
    short_dt_skips = 0

    for i, row in frame.iterrows():
        t = float(row["timestamp"])
        if i == 0:
            dt = DEFAULT_DT
        else:
            dt = t - last_t
            if dt <= 0.0:
                # The train file concatenates independent flights whose local
                # timestamps restart.  Keep a hard boundary and reset the EKF;
                # the original loop also drops this first non-increasing row.
                inputs.append(np.full(INPUT_DIM, np.nan))
                targets.append(np.full(STATE_DIM, np.nan))
                timestamps.append(np.nan)
                ekf = SlungLoadEKF(m_drone=M_DRONE, m_load=M_LOAD, l_rope=L_ROPE)
                last_t = t
                reset_count += 1
                continue

        last_t = t
        if dt > 0.1:
            dt = 0.05
        if dt < 0.001:
            short_dt_skips += 1
            continue

        quat = [row["drone_qx"], row["drone_qy"], row["drone_qz"], row["drone_qw"]]
        rot_matrix = R.from_quat(quat).as_matrix()

        thrust_mag = float(row["cmd_thrust"]) * MAX_THRUST
        if thrust_mag < 0.1:
            thrust_mag = (M_DRONE + M_LOAD) * 9.81
        thrust_body = np.array([0.0, 0.0, thrust_mag])
        u_enu = rot_matrix @ thrust_body

        acc_body = np.array([row["imu_ax"], row["imu_ay"], row["imu_az"]])
        acc_enu = rot_matrix @ acc_body

        u_ned = np.array([u_enu[1], u_enu[0], -u_enu[2]])
        acc_ned_raw = np.array([acc_enu[1], acc_enu[0], -acc_enu[2]])
        z_ned = acc_ned_raw + np.array([0.0, 0.0, 9.81])

        ekf.set_current_control(u_ned)
        ekf.predict(u=u_ned, dt=dt)
        ekf.update(z_meas=z_ned)
        x_est = ekf.x.copy()

        # Preserve 生成序列训练数据.py's current sign convention exactly.
        x_true = np.zeros(STATE_DIM)
        x_true[0] = -float(row["gt_xi"])
        x_true[1] = float(row["gt_zeta"])
        x_true[2] = float(row["gt_xi_dot"])
        x_true[3] = float(row["gt_zeta_dot"])

        if abs(x_est[0]) > 2.0 or abs(x_est[1]) > 2.0:
            inputs.append(np.full(INPUT_DIM, np.nan))
            targets.append(np.full(STATE_DIM, np.nan))
            timestamps.append(np.nan)
            ekf = SlungLoadEKF(m_drone=M_DRONE, m_load=M_LOAD, l_rope=L_ROPE)
            divergence_count += 1
            continue

        inputs.append(np.concatenate([x_est, u_ned, z_ned]))
        targets.append(x_true - x_est)
        timestamps.append(t)

    # Match the repository's explicit end-of-file separator.
    inputs.append(np.full(INPUT_DIM, np.nan))
    targets.append(np.full(STATE_DIM, np.nan))
    timestamps.append(np.nan)

    inputs_array = np.asarray(inputs, dtype=np.float64)
    targets_array = np.asarray(targets, dtype=np.float64)
    valid_rows = int((~np.isnan(inputs_array).any(axis=1)).sum())
    stats = {
        "source_rows": int(len(frame)),
        "replayed_rows_including_breaks": int(len(inputs_array)),
        "valid_rows": valid_rows,
        "timestamp_resets": reset_count,
        "ekf_divergences": divergence_count,
        "short_dt_skips": short_dt_skips,
    }
    return ReplayData(
        inputs=inputs_array,
        targets=targets_array,
        timestamps=np.asarray(timestamps, dtype=np.float64),
        stats=stats,
    )


def valid_window_starts(inputs: np.ndarray, seq_len: int) -> np.ndarray:
    """Match the repository's range(N - seq_len) and reject every NaN window."""

    if len(inputs) <= seq_len:
        return np.empty(0, dtype=np.int64)
    invalid = np.isnan(inputs).any(axis=1).astype(np.int32)
    counts = np.convolve(invalid, np.ones(seq_len, dtype=np.int32), mode="valid")
    # 生成序列训练数据.py uses range(len(all_inputs) - SEQ_LEN), hence [:-1].
    return np.flatnonzero(counts[:-1] == 0).astype(np.int64)


def limit_indices(indices: np.ndarray, maximum: int) -> np.ndarray:
    if maximum is None or maximum >= len(indices):
        return indices
    return indices[:maximum]


def make_loader(
    dataset: WindowDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    effective_batch_size: int,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    accumulated_samples = 0
    loss_sum = 0.0
    sample_count = 0

    for batch_index, (batch_x, batch_y, _) in enumerate(loader):
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)
        pred_y, _ = model(batch_x)
        loss = loss_fn(pred_y[:, :, :4], batch_y[:, :, :4])
        batch_samples = int(batch_x.shape[0])
        (loss * batch_samples).backward()
        accumulated_samples += batch_samples
        loss_sum += float(loss.detach().cpu()) * batch_samples
        sample_count += batch_samples

        is_last = batch_index + 1 == len(loader)
        if accumulated_samples >= effective_batch_size or is_last:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.div_(float(accumulated_samples))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            accumulated_samples = 0

    return loss_sum / max(sample_count, 1)


def validation_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            pred_y, _ = model(batch_x)
            loss = loss_fn(pred_y[:, :, :4], batch_y[:, :, :4])
            count = int(batch_x.shape[0])
            loss_sum += float(loss.cpu()) * count
            sample_count += count
    return loss_sum / max(sample_count, 1)


def evaluate_endpoints(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Use the final output of each 400-sample context window once."""

    model.eval()
    predictions: List[np.ndarray] = []
    endpoints: List[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _, batch_endpoints in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            pred_y, _ = model(batch_x)
            predictions.append(pred_y[:, -1, :].cpu().numpy())
            endpoints.append(batch_endpoints.numpy())
    return np.concatenate(predictions, axis=0), np.concatenate(endpoints, axis=0)


def rmse(values: np.ndarray, axis=None):
    return np.sqrt(np.mean(np.square(values), axis=axis))


def save_plots(
    output_dir: Path,
    history: Sequence[Dict[str, float]],
    result_frame: pd.DataFrame,
) -> None:
    epochs = [entry["epoch"] for entry in history]
    plt.figure(figsize=(8, 4.5))
    plt.semilogy(epochs, [entry["train_loss"] for entry in history], label="Train")
    plt.semilogy(epochs, [entry["val_loss"] for entry in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE (first four states)")
    plt.title("Repository EKF-LSTM training loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "训练损失曲线.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for axis, name, label in zip(axes, ("xi", "zeta"), ("Xi", "Zeta")):
        axis.plot(result_frame["time_s"], result_frame["true_{}_deg".format(name)], label="Ground truth", linewidth=1.2)
        axis.plot(result_frame["time_s"], result_frame["ekf_{}_deg".format(name)], label="EKF", linewidth=0.9, alpha=0.85)
        axis.plot(
            result_frame["time_s"],
            result_frame["ekf_lstm_{}_deg".format(name)],
            label="EKF-LSTM",
            linewidth=0.9,
            alpha=0.85,
        )
        axis.set_ylabel("{} angle (deg)".format(label))
        axis.grid(True, alpha=0.25)
    axes[0].legend(ncol=3, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Independent test flight: EKF versus EKF-LSTM")
    fig.tight_layout()
    fig.savefig(output_dir / "EKF与EKF-LSTM测试对比.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.seq_len <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("seq-len, epochs, and batch-size must be positive")
    if args.effective_batch_size < args.batch_size:
        raise ValueError("effective-batch-size must be at least batch-size")

    args.train_csv = args.train_csv.resolve()
    args.test_csv = args.test_csv.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    print("Replaying training CSV through the repository EKF...", flush=True)
    train_replay = replay_csv(args.train_csv)
    print("Replaying independent test CSV through the repository EKF...", flush=True)
    test_replay = replay_csv(args.test_csv)

    valid_train_rows = ~np.isnan(train_replay.inputs).any(axis=1)
    mean = train_replay.inputs[valid_train_rows].mean(axis=0)
    std = train_replay.inputs[valid_train_rows].std(axis=0) + 1e-6

    train_inputs_norm = train_replay.inputs.copy()
    train_inputs_norm[valid_train_rows] = (train_inputs_norm[valid_train_rows] - mean) / std
    valid_test_rows = ~np.isnan(test_replay.inputs).any(axis=1)
    test_inputs_norm = test_replay.inputs.copy()
    test_inputs_norm[valid_test_rows] = (test_inputs_norm[valid_test_rows] - mean) / std

    train_all_starts = valid_window_starts(train_inputs_norm, args.seq_len)
    test_starts = valid_window_starts(test_inputs_norm, args.seq_len)
    if len(train_all_starts) == 0 or len(test_starts) == 0:
        raise RuntimeError("No valid windows were generated")

    split_generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(train_all_starts), generator=split_generator).numpy()
    train_count = int(0.8 * len(permutation))
    train_starts = train_all_starts[permutation[:train_count]]
    val_starts = train_all_starts[permutation[train_count:]]
    train_starts = limit_indices(train_starts, args.max_train_windows)
    val_starts = limit_indices(val_starts, args.max_val_windows)
    test_starts = limit_indices(test_starts, args.max_test_windows)

    train_dataset = WindowDataset(train_inputs_norm, train_replay.targets, train_starts, args.seq_len)
    val_dataset = WindowDataset(train_inputs_norm, train_replay.targets, val_starts, args.seq_len)
    test_dataset = WindowDataset(test_inputs_norm, test_replay.targets, test_starts, args.seq_len)
    train_loader = make_loader(train_dataset, args.batch_size, True, args.num_workers, device, args.seed)
    val_loader = make_loader(val_dataset, args.batch_size, False, args.num_workers, device, args.seed)
    test_loader = make_loader(test_dataset, args.batch_size, False, args.num_workers, device, args.seed)

    model = LSTMCorrectionModel(
        input_dim=INPUT_DIM,
        state_dim=STATE_DIM,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(
        "Device={} | windows: train={} val={} test={} | micro/effective batch={}/{}".format(
            device, len(train_dataset), len(val_dataset), len(test_dataset), args.batch_size, args.effective_batch_size
        ),
        flush=True,
    )
    history: List[Dict[str, float]] = []
    training_start = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, args.effective_batch_size
        )
        val_loss = validation_loss(model, val_loader, loss_fn, device)
        elapsed = time.time() - epoch_start
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": elapsed}
        )
        print(
            "Epoch {:02d}/{:02d} train={:.8f} val={:.8f} time={:.1f}s".format(
                epoch, args.epochs, train_loss, val_loss, elapsed
            ),
            flush=True,
        )

    training_seconds = time.time() - training_start
    predictions, endpoints = evaluate_endpoints(model, test_loader, device)
    target_errors = test_replay.targets[endpoints]
    x_est = test_replay.inputs[endpoints, :STATE_DIM]
    x_true = x_est + target_errors
    x_corrected = x_est + predictions

    ekf_angle_error_deg = np.rad2deg(target_errors[:, :2])
    corrected_angle_error_deg = np.rad2deg(target_errors[:, :2] - predictions[:, :2])
    ekf_axis_rmse = rmse(ekf_angle_error_deg, axis=0)
    corrected_axis_rmse = rmse(corrected_angle_error_deg, axis=0)
    ekf_combined_rmse = float(rmse(ekf_angle_error_deg))
    corrected_combined_rmse = float(rmse(corrected_angle_error_deg))
    improvement_percent = 100.0 * (1.0 - corrected_combined_rmse / ekf_combined_rmse)

    result_frame = pd.DataFrame(
        {
            "time_s": test_replay.timestamps[endpoints],
            "true_xi_deg": np.rad2deg(x_true[:, 0]),
            "true_zeta_deg": np.rad2deg(x_true[:, 1]),
            "ekf_xi_deg": np.rad2deg(x_est[:, 0]),
            "ekf_zeta_deg": np.rad2deg(x_est[:, 1]),
            "ekf_lstm_xi_deg": np.rad2deg(x_corrected[:, 0]),
            "ekf_lstm_zeta_deg": np.rad2deg(x_corrected[:, 1]),
            "ekf_error_xi_deg": ekf_angle_error_deg[:, 0],
            "ekf_error_zeta_deg": ekf_angle_error_deg[:, 1],
            "ekf_lstm_error_xi_deg": corrected_angle_error_deg[:, 0],
            "ekf_lstm_error_zeta_deg": corrected_angle_error_deg[:, 1],
        }
    )
    result_frame.to_csv(args.output_dir / "测试集逐窗口预测.csv", index=False)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "config": {
            "input_dim": INPUT_DIM,
            "state_dim": STATE_DIM,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "seq_len": args.seq_len,
            "sampling_hz": 50,
            "seed": args.seed,
        },
        "history": history,
    }
    torch.save(checkpoint, args.output_dir / "LSTM离线修正模型.pth")
    save_plots(args.output_dir, history, result_frame)

    cuda_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    peak_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else None
    )
    metrics = {
        "method": {
            "baseline": "repository-faithful model and preprocessing with explicit timestamp-reset boundaries",
            "test_evaluation": "independent test CSV; final output of each 400-sample window",
            "target_sign_convention": "matches 源代码/生成序列训练数据.py",
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "device": str(device),
            "cuda_device": cuda_name,
            "peak_cuda_memory_mb": peak_memory_mb,
        },
        "config": {
            "sampling_hz": 50,
            "seq_len": args.seq_len,
            "window_seconds": args.seq_len / 50.0,
            "input_dim": INPUT_DIM,
            "state_dim": STATE_DIM,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "micro_batch_size": args.batch_size,
            "effective_batch_size": args.effective_batch_size,
            "seed": args.seed,
        },
        "data": {
            "train_csv": str(args.train_csv),
            "test_csv": str(args.test_csv),
            "train_sha256": file_sha256(args.train_csv),
            "test_sha256": file_sha256(args.test_csv),
            "train_replay": train_replay.stats,
            "test_replay": test_replay.stats,
            "all_valid_train_windows": int(len(train_all_starts)),
            "used_train_windows": int(len(train_dataset)),
            "used_validation_windows": int(len(val_dataset)),
            "used_test_windows": int(len(test_dataset)),
        },
        "training": {
            "seconds": training_seconds,
            "final_train_loss": history[-1]["train_loss"],
            "final_validation_loss": history[-1]["val_loss"],
            "history": history,
        },
        "independent_test_angle_rmse_deg": {
            "ekf": {
                "xi": float(ekf_axis_rmse[0]),
                "zeta": float(ekf_axis_rmse[1]),
                "combined": ekf_combined_rmse,
            },
            "ekf_lstm": {
                "xi": float(corrected_axis_rmse[0]),
                "zeta": float(corrected_axis_rmse[1]),
                "combined": corrected_combined_rmse,
            },
            "combined_improvement_percent": float(improvement_percent),
        },
    }
    with (args.output_dir / "评估指标.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    print(
        "Independent test angle RMSE (deg): EKF={:.6f}, EKF-LSTM={:.6f}, improvement={:.2f}%".format(
            ekf_combined_rmse, corrected_combined_rmse, improvement_percent
        ),
        flush=True,
    )
    print("Results saved to {}".format(args.output_dir), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
