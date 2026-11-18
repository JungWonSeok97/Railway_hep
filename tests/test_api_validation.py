"""Reject incomplete/ambiguous input before inference on every API alias."""

import contextlib
import io
import sys
import unittest
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app
from config.catboost_mappings import CATBOOST_OPTION_VALUES
from preprocessing.catboost_features import normalize_catboost_conditions


BATCH_ROUTES = ("/api/predict", "/api/catboost/predict", "/api/predict/catboost")
SINGLE_ROUTES = ("/api/predict/{psf}", "/api/catboost/predict/{psf}",
                 "/api/predict/catboost/{psf}", "/api/test/predict/{psf}")
ALL_ROUTES = BATCH_ROUTES + tuple(route.format(psf=1) for route in SINGLE_ROUTES) + (
    "/api/psf1/predict", "/api/psf4/forecast",
)


def scenario(psf):
    return {field: f"{field}.1" for field in CATBOOST_OPTION_VALUES if field.startswith(f"F.{psf}.")}


def legacy(conditions):
    return {key.replace(".S.", "."): value.replace(".S.", ".") for key, value in conditions.items()}


class ApiValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.enterContext(contextlib.redirect_stdout(io.StringIO()))
        self.enterContext(warnings.catch_warnings())
        warnings.simplefilter("ignore", UserWarning)

    def assert_error(self, response, status=400, contains=None):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        body = response.get_json()
        self.assertFalse(body["success"])
        self.assertIsInstance(body["error"], str)
        self.assertNotIn("result", body)
        self.assertNotIn("results", body)
        if contains is not None:
            self.assertIn(contains, body["error"])

    def test_each_missing_question_is_rejected_before_loading_models(self):
        with patch("predictors.catboost.load_catboost_model") as cbm, patch("preprocessing.catboost_features.load_gmm_bundle") as gmm:
            for psf in range(1, 9):
                for missing in scenario(psf):
                    body = scenario(psf)
                    del body[missing]
                    for pattern in SINGLE_ROUTES:
                        route = pattern.format(psf=psf)
                        with self.subTest(route=route, missing=missing):
                            self.assert_error(self.client.post(route, json=body), contains=missing)
                    for route in BATCH_ROUTES:
                        if psf == 4 and route == "/api/predict":
                            continue  # The formula requires months, not seven survey answers.
                        with self.subTest(route=route, missing=missing):
                            self.assert_error(self.client.post(route, json={f"F.{psf}": body}), contains=missing)
            cbm.assert_not_called()
            gmm.assert_not_called()

    def test_supplied_empty_null_and_non_object_psfs_are_not_skipped(self):
        for psf in range(1, 9):
            for invalid in ({}, None, [], False, "", 0):
                for route in BATCH_ROUTES:
                    for key in (f"F.{psf}", f"psf{psf}"):
                        with self.subTest(route=route, key=key, value=invalid):
                            self.assert_error(self.client.post(route, json={key: invalid}))

    def test_unknown_foreign_questions_and_conflicting_field_aliases(self):
        for psf in range(1, 9):
            base = scenario(psf)
            field = next(iter(base))
            extras = ({field + "_typo": base[field]}, {"F.99.S.1": 0},
                      {field.replace(".S.", "."): f"{field}.2"})
            for extra in extras:
                body = {**base, **extra}
                for pattern in SINGLE_ROUTES:
                    with self.subTest(psf=psf, extra=extra, route=pattern):
                        self.assert_error(self.client.post(pattern.format(psf=psf), json=body))
                for route in BATCH_ROUTES:
                    payload = {**body, "training_elapsed_months": 12} if psf == 4 else body
                    self.assert_error(self.client.post(route, json={f"F.{psf}": payload}))

    def test_invalid_values_and_codes_are_client_errors(self):
        for psf in range(1, 9):
            body = scenario(psf)
            field = next(iter(body))
            for invalid in (None, True, False, -1, 3, 0.5, float("nan"), float("inf"), 10**400, [], {}, "unknown", "F.99.S.1.1"):
                with self.subTest(psf=psf, invalid=repr(invalid)):
                    self.assert_error(self.client.post(f"/api/predict/{psf}", json={**body, field: invalid}))

    def test_current_legacy_and_equal_field_aliases_preserve_predictions(self):
        for psf in range(1, 9):
            body = scenario(psf)
            expected_response = self.client.post(f"/api/predict/{psf}", json=body)
            self.assertEqual(expected_response.status_code, 200)
            expected = expected_response.get_json()
            for supplied in (legacy(body), {**body, **legacy(body)}):
                for pattern in SINGLE_ROUTES[:3]:
                    with self.subTest(psf=psf, route=pattern):
                        response = self.client.post(pattern.format(psf=psf), json=supplied)
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.get_json(), expected)
            for psf_key in (f"F.{psf}", f"psf{psf}"):
                response = self.client.post("/api/catboost/predict", json={psf_key: body})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.get_json()["results"], [expected["result"]])

    def test_equivalent_numeric_field_aliases_are_accepted(self):
        body = scenario(1)
        # Sleep code .1 means numeric 2; different syntax, same learned value.
        body["F.1.3"] = 2
        self.assertEqual(normalize_catboost_conditions(1, body)["F.1.S.3"], 2)
        self.assertEqual(self.client.post("/api/predict/1", json=body).status_code, 200)

    def test_top_level_aliases_compare_original_inputs_before_gmm(self):
        base = scenario(2)
        changed = {**base, "F.2.S.3": "F.2.S.3.2"}
        # This raw difference disappears in the deployed GMM; it must still conflict.
        for route in BATCH_ROUTES:
            with self.subTest(route=route):
                self.assert_error(self.client.post(route, json={"F.2": base, "psf2": changed}), contains="서로 다릅니다")
                response = self.client.post(route, json={"F.2": base, "psf2": legacy(base)})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(response.get_json()["results"]), 1)
                for invalid in (None, {}, []):
                    self.assert_error(self.client.post(route, json={"F.2": base, "psf2": invalid}))

    def test_unknown_psf_keys_and_all_batch_inputs_are_validated_before_prediction(self):
        with patch("app.predict_psf_catboost") as predict:
            for route in BATCH_ROUTES:
                for extra in ({"F.9": scenario(1)}, {"F.1_typo": scenario(1)}, {"psf01": scenario(1)}, {"F.8": {}}):
                    self.assert_error(self.client.post(route, json={"F.1": scenario(1), **extra}))
            predict.assert_not_called()

    def test_absent_psfs_and_empty_batch_keep_partial_product_contract(self):
        for route in BATCH_ROUTES:
            empty = self.client.post(route, json={})
            self.assertEqual(empty.status_code, 200)
            self.assertEqual(empty.get_json()["results"], [])
            self.assertIsNone(empty.get_json()["psf_multiplier_product"])
            response = self.client.post(route, json={"F.1": scenario(1)})
            self.assertEqual(response.status_code, 200)
            body = response.get_json()
            self.assertEqual(len(body["results"]), 1)
            self.assertIsNone(body["psf_multiplier_product"])
            self.assertEqual(body["missing_psfs"], [f"F.{i}" for i in range(2, 9)])

    def test_formula_psf4_only_requires_months_and_checks_supplied_survey_fields(self):
        months = {"training_elapsed_months": 12}
        expected = self.client.post("/api/predict", json={"F.4": months}).get_json()
        for fields in ({}, scenario(4), {"F.4.S.1": "F.4.S.1.1"}):
            response = self.client.post("/api/predict", json={"F.4": {**months, **fields}})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), expected)
        for extra in ({"F.4.S.1_typo": 0}, {"F.4.S.1": None}, {"F.4.S.1": "F.4.S.1.1", "F.4.1": 2}):
            self.assert_error(self.client.post("/api/predict", json={"F.4": {**months, **extra}}))
        self.assert_error(self.client.post("/api/predict", json={"F.4": months, "psf4": {"training_elapsed_months": 13}}))
        for invalid in (True, None, "12", -1, float("inf"), float("nan"), 10**400):
            self.assert_error(self.client.post("/api/predict", json={"F.4": {"training_elapsed_months": invalid}}))

    def test_catboost_psf4_accepts_formula_metadata_but_requires_all_questions(self):
        for pattern in SINGLE_ROUTES:
            route = pattern.format(psf=4)
            expected = self.client.post(route, json=scenario(4)).get_json()
            response = self.client.post(route, json={**scenario(4), "training_elapsed_months": 12})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), expected)
            self.assert_error(self.client.post(route, json={"training_elapsed_months": 12}))

    def test_bad_json_and_non_object_json_return_400_on_every_route(self):
        for route in ALL_ROUTES:
            for body in ("{bad", "", "null", "[]", '"text"', "false", "1"):
                with self.subTest(route=route, body=body):
                    self.assert_error(self.client.post(route, data=body, content_type="application/json"))

    def test_wrong_or_missing_content_type_returns_json_415_on_every_route(self):
        for route in ALL_ROUTES:
            for content_type in (None, "text/plain", "application/x-www-form-urlencoded"):
                with self.subTest(route=route, content_type=content_type):
                    self.assert_error(self.client.post(route, data="{}", content_type=content_type), status=415)

    def test_forecast_unknown_missing_and_invalid_fields(self):
        started = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        for route, valid in (("/api/psf1/predict", {"work_started_at": started, "base_psf1": 0}),
                             ("/api/psf4/forecast", {"training_elapsed_months": 12})):
            self.assertEqual(self.client.post(route, json=valid).status_code, 200)
            self.assert_error(self.client.post(route, json={**valid, "typo": 0}))
            for key in valid:
                self.assert_error(self.client.post(route, json={k: v for k, v in valid.items() if k != key}))
        self.assert_error(self.client.post("/api/psf4/forecast", json={"training_elapsed_months": 10**400}))

    def test_real_execution_errors_still_return_500(self):
        with patch("app.predict_psf_catboost", side_effect=RuntimeError("model failure")):
            self.assert_error(self.client.post("/api/predict/1", json=scenario(1)), status=500, contains="model failure")

    def test_unsupported_psf_and_non_string_question_keys(self):
        for psf in (0, 9):
            for pattern in SINGLE_ROUTES:
                self.assert_error(self.client.post(pattern.format(psf=psf), json=scenario(1)))
        for psf in range(1, 9):
            with self.assertRaises(ValueError):
                normalize_catboost_conditions(psf, {**scenario(psf), 3: 0})


if __name__ == "__main__":
    unittest.main()
