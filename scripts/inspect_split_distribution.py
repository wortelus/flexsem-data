from pathlib import Path
import sys

import joblib
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

repo_root = Path(__file__).resolve().parents[1]
run_dir = repo_root / Path(
    "rnn/outputs/"
    "inverse_gru_h64_l1_b0_seq16_delta_residual_delta_"
    "nh0_mse_bs16_lr0p0005_do0p0_seed10"
)

scaler = joblib.load(run_dir / "scalers/scaler.gz")

for split in ("train", "val", "test"):
    dataset = torch.load(
        run_dir / "dataset" / f"{split}.pt",
        map_location="cpu",
        weights_only=False,
    )

    X_scaled, y_scaled = dataset.tensors
    X_scaled = X_scaled.numpy()
    y_scaled = y_scaled.numpy()

    # Scaler je jeden globální scaler aplikovaný po jednotlivých hodnotách.
    X_nm = scaler.inverse_transform(
        X_scaled.reshape(-1, 1)
    ).reshape(X_scaled.shape)

    y_nm = scaler.inverse_transform(
        y_scaled.reshape(-1, 1)
    ).reshape(y_scaled.shape)

    # Pozorovaný pohyb je vstup inverse modelu.
    observed_delta = X_nm[:, -1, :2]

    # Ground-truth motorový command použitý v původních datech. Pro
    # residual_delta platí y = command_delta - observed_delta.
    command_delta = np.rint(observed_delta + y_nm)
    command_size = np.linalg.norm(command_delta, axis=1)

    bins = np.array([0, 100, 300, 1_000, 3_000, np.inf])
    labels = ["0–100", "100–300", "300–1000", "1–3 µm", "≥3 µm"]

    counts, _ = np.histogram(command_size, bins=bins)

    print(f"\n{split}: {len(command_size)} oken (podle velikosti command delta)")
    for label, count in zip(labels, counts):
        share = 100 * count / len(command_size)
        print(f"{label:10} {count:6d}  {share:6.2f} %")
