import unittest

import numpy as np
import torch

from rnn.models.model_tcn import HysteresisTCN
from rnn.preprocessing.double import quantize_command_positions


class HysteresisTCNTests(unittest.TestCase):
    def test_output_shape_and_receptive_field(self):
        model = HysteresisTCN(
            input_size=4,
            hidden_size=8,
            output_size=2,
            num_layers=3,
            dropout=0.0,
            tcn_kernel_size=3,
        )
        output = model(torch.randn(5, 16, 4))
        self.assertEqual(tuple(output.shape), (5, 2))
        self.assertEqual(model.receptive_field, 29)

    def test_temporal_features_are_causal(self):
        torch.manual_seed(3)
        model = HysteresisTCN(
            input_size=4,
            hidden_size=8,
            output_size=2,
            num_layers=3,
            dropout=0.0,
            tcn_kernel_size=3,
        ).eval()
        original = torch.randn(2, 16, 4)
        modified = original.clone()
        modified[:, 9:, :] += 1000.0
        with torch.no_grad():
            original_features = model.network(original.transpose(1, 2))
            modified_features = model.network(modified.transpose(1, 2))
        torch.testing.assert_close(
            original_features[:, :, :9],
            modified_features[:, :, :9],
            rtol=0.0,
            atol=0.0,
        )

    def test_bidirectional_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cannot be bidirectional"):
            HysteresisTCN(
                input_size=4,
                hidden_size=8,
                output_size=2,
                num_layers=3,
                dropout=0.0,
                bidirectional=True,
            )


class CommandQuantizationTests(unittest.TestCase):
    def test_q50_uses_absolute_positions_and_positive_tie_break(self):
        values = np.array([-75.0, -25.0, 24.9, 25.0, 74.9, 75.0])
        quantized = quantize_command_positions(values, 50.0)
        np.testing.assert_array_equal(
            quantized,
            np.array([-50.0, 0.0, 0.0, 50.0, 50.0, 100.0]),
        )


if __name__ == "__main__":
    unittest.main()
