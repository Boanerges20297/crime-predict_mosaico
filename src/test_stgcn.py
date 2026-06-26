import os
import sys
import unittest

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if os.path.join(BASE_DIR, "src") not in sys.path:
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from stgcn_forecaster import is_stgcn_available, run_stgcn_pipeline


@unittest.skipUnless(is_stgcn_available(), "PyTorch não disponível para o teste do ST-GCN.")
class TestSTGCNPipeline(unittest.TestCase):
    def _build_synthetic_weekly_series(self):
        weeks = pd.date_range("2024-01-01", periods=40, freq="W-MON")
        rows = []
        for region_id in range(4):
            base = region_id + 1
            values = np.maximum(
                0,
                np.round(
                    base
                    + 0.4 * np.sin(np.arange(len(weeks)) / 3.0 + region_id)
                    + 0.2 * np.arange(len(weeks)) / len(weeks)
                ),
            ).astype(int)
            for idx, week in enumerate(weeks):
                lag_1 = values[idx - 1] if idx >= 1 else 0
                lag_2 = values[idx - 2] if idx >= 2 else 0
                rows.append(
                    {
                        "semana": week,
                        "hex_id": region_id,
                        "crimes": int(values[idx]),
                        "lag_1": float(lag_1),
                        "lag_2": float(lag_2),
                        "lag_3": float(values[idx - 3] if idx >= 3 else 0),
                        "media_movel_4": float(np.mean(values[max(0, idx - 4):idx])) if idx > 0 else 0.0,
                        "tendencia_1": float(lag_1 - lag_2),
                        "semana_ano": int(week.isocalendar().week),
                        "mes": int(week.month),
                    }
                )
        return pd.DataFrame(rows)

    def test_stgcn_pipeline_returns_non_negative_forecasts(self):
        df_model = self._build_synthetic_weekly_series()
        result = run_stgcn_pipeline(
            df_model,
            region_col="hex_id",
            grid=None,
            train_split_date="2024-08-01",
            seq_len=8,
            ensemble_size=1,
        )
        self.assertEqual(result["model_name"], "stgcn_poisson_nll")
        self.assertGreater(result["node_count"], 0)
        self.assertFalse(result["forecasts_df"].empty)
        self.assertTrue((result["forecasts_df"]["previsao_proxima_semana"] >= 0).all())
        self.assertGreaterEqual(result["mse"], 0.0)
        self.assertGreaterEqual(result["mae"], 0.0)


if __name__ == "__main__":
    unittest.main()
