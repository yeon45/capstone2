"""Unit tests for CPM-triangle ESN helpers."""

from __future__ import annotations

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

from src.cpm.core import turning_points_to_label_array
from src.esn.dataset import build_triangle_target, make_esn_dataset, split_esn_arrays
from src.esn.metrics import score_to_signal
from src.esn.model import EchoStateNetwork
from src.pipeline.run_esn_sweep import _is_better, _parameter_grid


class TriangleTargetTests(unittest.TestCase):
    """Tests for CPM turning-point triangle target generation."""

    def test_cpm_label_convention_is_bottom_negative_top_positive(self) -> None:
        labels = turning_points_to_label_array(
            5,
            [{"index": 1, "type": "bottom"}, {"index": 3, "type": "top"}],
        )

        np.testing.assert_array_equal(labels, np.array([0, -1, 0, 1, 0]))

    def test_build_triangle_target_interpolates_between_points(self) -> None:
        df = pd.DataFrame({"Close": np.arange(60.0)})
        turning_points = [
            {"index": 10, "type": "bottom"},
            {"index": 30, "type": "top"},
            {"index": 50, "type": "bottom"},
        ]

        target = build_triangle_target(df, turning_points)

        self.assertEqual(len(target), len(df))
        self.assertAlmostEqual(float(target.iloc[0]), -1.0)
        self.assertAlmostEqual(float(target.iloc[10]), -1.0)
        self.assertAlmostEqual(float(target.iloc[20]), 0.0)
        self.assertAlmostEqual(float(target.iloc[30]), 1.0)
        self.assertAlmostEqual(float(target.iloc[40]), 0.0)
        self.assertAlmostEqual(float(target.iloc[50]), -1.0)
        self.assertAlmostEqual(float(target.iloc[-1]), -1.0)
        self.assertEqual(target.name, "triangle_target")
        self.assertTrue(target.index.equals(df.index))
        self.assertTrue(target.between(-1.0, 1.0).all())

    def test_build_triangle_target_sorts_unsorted_points(self) -> None:
        df = pd.DataFrame({"Close": np.arange(40.0)})
        turning_points = [
            {"index": 30, "type": "top"},
            {"index": 10, "type": "bottom"},
        ]

        target = build_triangle_target(df, turning_points)

        self.assertAlmostEqual(float(target.iloc[10]), -1.0)
        self.assertAlmostEqual(float(target.iloc[20]), 0.0)
        self.assertAlmostEqual(float(target.iloc[30]), 1.0)


class ESNDatasetTests(unittest.TestCase):
    """Tests for ESN X/y construction and chronological splitting."""

    def test_make_esn_dataset_uses_signal_cols_and_triangle_target(self) -> None:
        df = pd.DataFrame(
            {
                "Close": [10.0, 11.0, 12.0, 13.0],
                "ma_signal": [-1, 0, 1, np.nan],
                "rsi_signal": [0, 1, -1, 0],
                "triangle_target": [-1.0, -0.3, 0.3, 1.0],
            },
            index=pd.Index([10, 11, 12, 13], name="row"),
        )

        X, y = make_esn_dataset(df, ["ma_signal", "rsi_signal"], "triangle_target")

        self.assertEqual(X.columns.tolist(), ["ma_signal", "rsi_signal"])
        self.assertNotIn("Close", X.columns)
        self.assertEqual(y.name, "triangle_target")
        self.assertTrue(X.index.equals(y.index))
        self.assertEqual(X.index.tolist(), [10, 11, 12])

    def test_split_esn_arrays_preserves_order_and_scales_from_train(self) -> None:
        X = np.arange(20, dtype=float).reshape(10, 2)
        y = np.arange(10, dtype=float)
        dates = np.array([f"d{i}" for i in range(10)])

        dataset = split_esn_arrays(
            X=X,
            y=y,
            dates=dates,
            feature_cols=["a", "b"],
            target_col="triangle_target",
            target_shift=0,
            input_path=Path("dummy.csv"),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        self.assertEqual(dataset.train.dates.tolist(), ["d0", "d1", "d2", "d3", "d4", "d5"])
        self.assertEqual(dataset.val.dates.tolist(), ["d6", "d7"])
        self.assertEqual(dataset.test.dates.tolist(), ["d8", "d9"])
        np.testing.assert_allclose(dataset.train.X.mean(axis=0), np.zeros(2), atol=1e-12)


class EchoStateNetworkTests(unittest.TestCase):
    """Tests for ESN fit/predict shape behavior."""

    def test_fit_predict_returns_matching_length(self) -> None:
        rng = np.random.default_rng(42)
        X = rng.normal(size=(80, 3))
        y = np.sin(np.linspace(0.0, 4.0, 80))
        model = EchoStateNetwork(
            input_dim=3,
            reservoir_size=20,
            washout=5,
            random_state=7,
        )

        model.fit(X, y)
        pred = model.predict(X)

        self.assertEqual(len(pred), len(X))
        self.assertIsNotNone(model.Wout)
        self.assertTrue(np.all(np.asarray(pred) <= 1.0))
        self.assertTrue(np.all(np.asarray(pred) >= -1.0))

    def test_fit_does_not_change_fixed_reservoir_weights(self) -> None:
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 2))
        y = np.linspace(-1.0, 1.0, 60)
        model = EchoStateNetwork(input_dim=2, reservoir_size=12, washout=3)
        win_before = model.Win.copy()
        w_before = model.W.copy()

        model.fit(X, y)

        np.testing.assert_allclose(model.Win, win_before)
        np.testing.assert_allclose(model.W, w_before)


class ScoreToSignalTests(unittest.TestCase):
    """Tests for ESN trading-score signal conversion."""

    def test_score_to_signal_numeric_and_labels(self) -> None:
        scores = np.array([-0.6, -0.2, 0.0, 0.2, 0.7])

        numeric = score_to_signal(scores, threshold=0.3, as_label=False)
        labels = score_to_signal(scores, threshold=0.3, as_label=True)

        np.testing.assert_array_equal(numeric, np.array([-1, 0, 0, 0, 1]))
        np.testing.assert_array_equal(
            labels,
            np.array(["buy", "hold", "hold", "hold", "sell"]),
        )

    def test_score_to_signal_rejects_negative_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "threshold must be non-negative"):
            score_to_signal(np.array([0.0]), threshold=-0.1)


class ESNSweepTests(unittest.TestCase):
    """Tests for ESN sweep configuration helpers."""

    def test_parameter_grid_expands_cartesian_product(self) -> None:
        grid = _parameter_grid(
            {
                "reservoir_size": [100, 200],
                "spectral_radius": [0.8, 0.9],
                "washout": [30],
            }
        )

        self.assertEqual(len(grid), 4)
        self.assertEqual(
            grid[0],
            {"reservoir_size": 100, "spectral_radius": 0.8, "washout": 30},
        )
        self.assertEqual(
            grid[-1],
            {"reservoir_size": 200, "spectral_radius": 0.9, "washout": 30},
        )

    def test_is_better_supports_max_and_min_modes(self) -> None:
        candidate = {"val_directional_accuracy": 0.6, "val_mse": 0.2}
        incumbent = {"val_directional_accuracy": 0.5, "val_mse": 0.3}

        self.assertTrue(
            _is_better(candidate, incumbent, "val_directional_accuracy", "max")
        )
        self.assertTrue(_is_better(candidate, incumbent, "val_mse", "min"))


if __name__ == "__main__":
    unittest.main()
