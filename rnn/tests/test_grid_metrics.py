import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

from rnn.run_2_fit_grid import (
    DatasetVariant,
    Experiment,
    evaluate_command_delta_metrics,
)


class IdentityScaler:
    def inverse_transform(self, values):
        return values


class ConstantModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.register_buffer("output", torch.tensor(output, dtype=torch.float32))

    def forward(self, sequences):
        return self.output.expand(len(sequences), -1)


def make_experiment(window_coord_mode, target_mode):
    variant = DatasetVariant(
        name="test",
        train_path=Path("train.pt"),
        val_path=Path("val.pt"),
        test_path=Path("test.pt"),
        scaler_path=Path("scaler.gz"),
        window_coord_mode=window_coord_mode,
        target_mode=target_mode,
        sequence_length=2,
    )
    return Experiment(
        dataset=variant,
        model_type="tcn",
        hidden_size=4,
        num_layers=1,
        tcn_kernel_size=2,
        bidirectional=False,
        dropout=0.0,
        batch_size=1,
        learning_rate=1e-3,
        loss_mode="mse",
        seed=1,
    )


class CommonCommandMetricTests(unittest.TestCase):
    def assert_representation_maps_to_same_step(
        self, window_coord_mode, target_mode, sequence, label
    ):
        experiment = make_experiment(window_coord_mode, target_mode)
        dataset = TensorDataset(
            torch.tensor([sequence], dtype=torch.float32),
            torch.tensor([label], dtype=torch.float32),
        )
        metrics = evaluate_command_delta_metrics(
            ConstantModel(label),
            dataset,
            experiment,
            IdentityScaler(),
            torch.device("cpu"),
        )
        self.assertEqual(metrics["rmse_nm"], 0.0)
        self.assertAlmostEqual(
            metrics["true_step_distance_mean_nm"],
            float(np.hypot(50.0, 100.0)),
            places=5,
        )
        self.assertEqual(
            metrics["by_step_size"]["gt0_le300_nm"]["samples"], 1
        )
        self.assertEqual(
            metrics["by_step_size"]["gt0_le300_nm"]["rmse_nm"], 0.0
        )

    def test_relative_actual_position(self):
        self.assert_representation_maps_to_same_step(
            "relative",
            "actual_delta",
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 100.0, 200.0]],
            [150.0, 300.0],
        )

    def test_direct_delta(self):
        self.assert_representation_maps_to_same_step(
            "delta",
            "actual_delta",
            [[0.0, 0.0, 0.0, 0.0], [30.0, 50.0, 0.0, 0.0]],
            [50.0, 100.0],
        )

    def test_residual_delta(self):
        self.assert_representation_maps_to_same_step(
            "delta",
            "residual_delta",
            [[0.0, 0.0, 0.0, 0.0], [30.0, 50.0, 0.0, 0.0]],
            [20.0, 50.0],
        )


if __name__ == "__main__":
    unittest.main()
