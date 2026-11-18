"""Multiplier API contract: complete products, individual factors, no HEP."""

import contextlib
import io
import math
import sys
import unittest
import warnings
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from config.catboost_mappings import CATBOOST_OPTION_VALUES
from predictors.psf_multipliers import multiplier_for_prediction, summarize_psf_results
from predictors.psf4_formula import predict_psf4_formula, build_psf4_forecast
from predictors.psf1_forecast import build_psf1_forecast, KST


def records(factors=None):
    factors = factors or [1] * 8
    return [{"psf_id": f"F.{i + 1}", "psf_multiplier": factor} for i, factor in enumerate(factors)]


class ProductTests(unittest.TestCase):
    def test_original_level_multipliers_are_preserved(self):
        expected = {
            1: [1, 14.45, 17.54], 2: [0.492, 1, 5.654], 3: [0.292, 1, 8.559],
            4: [0.421, 1, 7.229], 5: [1, 9.559], 6: [1, 11.576],
            7: [0.322, 1, 12.11], 8: [1, 4.654, 11.873],
        }
        for psf, factors in expected.items():
            for option, expected_factor in enumerate(factors, 1):
                self.assertEqual(multiplier_for_prediction(psf, f"F.{psf}.{option}"), expected_factor)
        for psf, code in [(6, "F.6.3"), (1, "보통"), (4, "F.1.1"), (9, "F.9.1")]:
            with self.assertRaises(ValueError):
                multiplier_for_prediction(psf, code)

    def test_product_is_unbounded_and_monotonic_at_old_hep_threshold(self):
        for factor in (0.33, 1, 2, 54, 55, 1_000_000):
            self.assertEqual(summarize_psf_results(records([factor] + [1] * 7))["psf_multiplier_product"], factor)
        all_bad = [17.54, 5.654, 8.559, 7.229, 9.559, 11.576, 12.11, 11.873]
        self.assertAlmostEqual(summarize_psf_results(records(all_bad))["psf_multiplier_product"], math.prod(all_bad))

    def test_partial_duplicate_missing_and_invalid_factors(self):
        partial = summarize_psf_results(records()[:2])
        self.assertIsNone(partial["psf_multiplier_product"])
        self.assertEqual(partial["missing_psfs"], [f"F.{i}" for i in range(3, 9)])
        self.assertIsNone(summarize_psf_results([])["psf_multiplier_product"])
        for invalid in (True, None, "1", 0, -1, float("nan"), float("inf")):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                summarize_psf_results(records([invalid] + [1] * 7))
        for invalid in [records() + records()[:1], [{"psf_id": "F.9", "psf_multiplier": 1}], [{"psf_id": "F.1"}], records([1e300] * 8)]:
            with self.assertRaises(ValueError):
                summarize_psf_results(invalid)

    def test_real_monthly_factors_do_not_drop_after_44_months(self):
        products = []
        for month in (44, 45):
            factor = predict_psf4_formula({"training_elapsed_months": month})["psf_multiplier"]
            products.append(summarize_psf_results(records([1, 1, 1, factor, 1, 11.576, 1, 1]))["psf_multiplier_product"])
        self.assertAlmostEqual(products[0], 54.314592)
        self.assertAlmostEqual(products[1], 54.453504)
        self.assertGreater(products[1], products[0])


class MultiplierApiTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def assert_no_hep(self, payload):
        if isinstance(payload, dict):
            for key, value in payload.items():
                self.assertNotIn("hep", key.lower())
                self.assert_no_hep(value)
        elif isinstance(payload, list):
            for value in payload:
                self.assert_no_hep(value)

    def test_actual_complete_and_partial_prediction_routes(self):
        request = {
            f"F.{psf}": {field: f"{field}.1" for field in CATBOOST_OPTION_VALUES if field.startswith(f"F.{psf}.S.")}
            for psf in range(1, 9)
        }
        request["F.4"]["training_elapsed_months"] = 44
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for route in ("/api/predict", "/api/catboost/predict", "/api/predict/catboost"):
                response = self.client.post(route, json=request)
                self.assertEqual(response.status_code, 200)
                data = response.get_json()
                self.assert_no_hep(data)
                self.assertEqual(data["missing_psfs"], [])
                self.assertEqual(len(data["results"]), 8)
                self.assertAlmostEqual(data["psf_multiplier_product"], math.prod(r["psf_multiplier"] for r in data["results"]))
                fourth = next(r for r in data["results"] if r["psf_id"] == "F.4")
                if route == "/api/predict":
                    self.assertEqual(fourth["psf_multiplier"], 4.692)
                else:
                    self.assertEqual(fourth["psf_multiplier"], multiplier_for_prediction(4, fourth["predicted_text"]))
                partial = self.client.post(route, json={"F.6": request["F.6"]}).get_json()
                self.assertIsNone(partial["psf_multiplier_product"])
                self.assertEqual(len(partial["missing_psfs"]), 7)
            for route in ("/api/predict/6", "/api/catboost/predict/6", "/api/predict/catboost/6", "/api/test/predict/6"):
                response = self.client.post(route, json=request["F.6"])
                self.assertEqual(response.status_code, 200)
                self.assertIn("psf_multiplier", response.get_json()["result"])
                self.assert_no_hep(response.get_json())

    def test_forecast_routes_supply_factors_without_hep(self):
        monthly = self.client.post('/api/psf4/forecast', json={"training_elapsed_months": 44})
        self.assertEqual(monthly.status_code, 200)
        data = monthly.get_json()
        self.assert_no_hep(data)
        factors = data["series"]["psf_multipliers"]
        self.assertEqual(len(factors), 601)
        self.assertEqual(factors[0], 4.692)
        self.assertEqual(factors[10], 4.704)
        self.assertTrue(all(a <= b for a, b in zip(factors, factors[1:])))
        fixture = build_psf1_forecast("2026-11-18T08:00:00+09:00", 0, datetime(2026, 11, 18, 8, tzinfo=KST))
        with patch('app.build_psf1_forecast', return_value=fixture):
            response = self.client.post('/api/psf1/predict', json={"work_started_at": "2026-11-18T08:00:00+09:00", "base_psf1": 0})
        self.assertEqual(response.status_code, 200)
        self.assert_no_hep(response.get_json())
        for result in fixture["segments"] + [fixture["base_prediction"], fixture["current_prediction"]]:
            self.assertEqual(result["psf_multiplier"], (1, 14.45, 17.54)[result["predicted_label"]])


if __name__ == '__main__':
    unittest.main()
