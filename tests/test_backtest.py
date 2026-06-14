"""Unit tests for simple ESN signal backtesting."""

from __future__ import annotations

import contextlib
import io
import unittest

with contextlib.redirect_stderr(io.StringIO()):
    import pandas as pd

from src.backtest.simple_backtest import buy_and_hold_summary, run_long_only_backtest


class LongOnlyBacktestTests(unittest.TestCase):
    """Tests for long-only all-in/all-out signal handling."""

    def test_buy_then_sell_updates_portfolio_value(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["d0", "d1", "d2"],
                "Close": [10.0, 15.0, 20.0],
                "signal": [-1, 0, 1],
            }
        )

        equity, trades, summary = run_long_only_backtest(df, initial_cash=100.0)

        self.assertEqual(trades["action"].tolist(), ["buy", "sell"])
        self.assertAlmostEqual(float(equity["portfolio_value"].iloc[-1]), 200.0)
        self.assertAlmostEqual(summary["total_return"], 1.0)

    def test_duplicate_buy_while_position_open_is_ignored(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["d0", "d1", "d2"],
                "Close": [10.0, 11.0, 12.0],
                "signal": ["buy", "buy", "sell"],
            }
        )

        _, trades, summary = run_long_only_backtest(df, initial_cash=100.0)

        self.assertEqual(trades["action"].tolist(), ["buy", "sell"])
        self.assertEqual(summary["num_buy"], 1)

    def test_sell_without_position_is_ignored(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["d0", "d1", "d2"],
                "Close": [10.0, 11.0, 12.0],
                "signal": [1, 0, 1],
            }
        )

        equity, trades, summary = run_long_only_backtest(df, initial_cash=100.0)

        self.assertTrue(trades.empty)
        self.assertAlmostEqual(float(equity["portfolio_value"].iloc[-1]), 100.0)
        self.assertEqual(summary["num_trades"], 0)

    def test_max_drawdown_is_computed_from_equity_curve(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["d0", "d1", "d2", "d3"],
                "Close": [10.0, 20.0, 10.0, 15.0],
                "signal": [-1, 0, 0, 0],
            }
        )

        _, _, summary = run_long_only_backtest(df, initial_cash=100.0)

        self.assertAlmostEqual(summary["max_drawdown"], -0.5)

    def test_buy_and_hold_baseline_is_created(self) -> None:
        df = pd.DataFrame(
            {
                "date": ["d0", "d1", "d2"],
                "Close": [10.0, 15.0, 20.0],
                "signal": [0, 0, 0],
            }
        )

        _, _, summary = run_long_only_backtest(df, initial_cash=100.0)
        baseline = buy_and_hold_summary(df, initial_cash=100.0)

        self.assertIn("buy_and_hold", summary)
        self.assertAlmostEqual(summary["buy_and_hold"]["final_value"], 200.0)
        self.assertAlmostEqual(baseline["total_return"], 1.0)


if __name__ == "__main__":
    unittest.main()
