"""PSF6 schema, deployed model and HTTP regression checks (no retraining)."""

import contextlib
import io
import random
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.catboost_config import CATBOOST_CONFIG
from config.environment_schema import (
    ENVIRONMENT_FIELDS, ENVIRONMENT_FEATURES, ENVIRONMENT_SCHEMA_SHA256,
    ENVIRONMENT_SCHEMA_VERSION,
)
from preprocessing.catboost_features import GMM_PSF_IDS, build_catboost_features
from preprocessing.environment_features import normalize_environment_condition
from predictors.catboost import load_catboost_model, predict_psf_catboost, predict_psf_catboost_test
from test_catboost_encoding import SurveyOptions, normalize_text, workbook_rows


def baseline():
    return {field: f"{field}.1" for field in ENVIRONMENT_FIELDS}


class EnvironmentSchemaTests(unittest.TestCase):
    def test_all_32_ui_choices_match_original_workbook_text(self):
        survey = SurveyOptions()
        survey.feed((ROOT / "input.html").read_text(encoding="utf-8"))
        reference = {
            normalize_text(text): (column, text)
            for psf, column, text, _ in workbook_rows() if psf == "환경"
        }
        options = [option for option in survey.options if option["field"].startswith("F.6.S.")]
        self.assertEqual(len(options), 32)
        self.assertEqual(len(reference), 32)
        expected_baseline = {}
        for option in options:
            column, text = reference[normalize_text(option["text"])]
            if option["code"].endswith(".1"):
                expected_baseline[column] = text
        self.assertEqual(len(expected_baseline), 14)
        for option in options:
            field, code = option["field"], option["code"]
            column, text = reference[normalize_text(option["text"])]
            expected = {**expected_baseline, column: text}
            with self.subTest(code=code):
                self.assertEqual(normalize_environment_condition(code, field), text)
                self.assertEqual(normalize_environment_condition(option["text"], field), text)
                self.assertEqual(normalize_environment_condition(code.replace(".S.", "."), field), text)
                coded = {**baseline(), field: code}
                actual = build_catboost_features(6, coded)
                self.assertEqual(actual, [expected[name] for name in sorted(expected)])
                legacy = {key.replace(".S.", "."): value.replace(".S.", ".") for key, value in coded.items()}
                self.assertEqual(build_catboost_features(6, legacy), actual)

    def test_temperature_and_humidity_keep_three_levels(self):
        for field in ("F.6.S.10", "F.6.S.11"):
            categories = [normalize_environment_condition(f"{field}.{i}", field) for i in (1, 2, 3)]
            self.assertEqual(len(set(categories)), 3)
            self.assertEqual(normalize_environment_condition(0, field), categories[1])
            for ambiguous in (1, 1.0, "1", "1.0"):
                with self.subTest(field=field, value=ambiguous), self.assertRaisesRegex(ValueError, "낮음/높음"):
                    normalize_environment_condition(ambiguous, field)

    def test_legacy_numbers_are_only_accepted_when_unambiguous(self):
        from config.catboost_mappings import CATBOOST_OPTION_VALUES
        for field, (_, categories) in ENVIRONMENT_FIELDS.items():
            encoded = CATBOOST_OPTION_VALUES[field]
            for index, number in enumerate(encoded):
                if encoded.count(number) == 1:
                    for supplied in (number, float(number), str(number)):
                        self.assertEqual(normalize_environment_condition(supplied, field), categories[index])

    def test_missing_foreign_conflicting_and_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "14개"):
            build_catboost_features(6, {})
        for field in ENVIRONMENT_FIELDS:
            missing = baseline()
            missing.pop(field)
            with self.subTest(missing=field), self.assertRaises(ValueError):
                build_catboost_features(6, missing)
        for invalid in (None, True, False, -1, 2, 0.5, float("nan"), float("inf"), "nan", [], {}, "F.6.S.2.1", "F.6.S.1.3"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                build_catboost_features(6, {**baseline(), "F.6.S.1": invalid})
        for extra in ({"F.5.S.1": "F.5.S.1.1"}, {"F.6.1": "F.6.1.2"}, {3: 0}):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                build_catboost_features(6, {**baseline(), **extra})
        self.assertEqual(build_catboost_features(6, {**baseline(), "F.6.1": "F.6.1.1"}), build_catboost_features(6, baseline()))

    def test_environment_bypasses_gmm_and_other_gmm_psfs_remain(self):
        self.assertEqual(GMM_PSF_IDS, {2, 4, 8})
        with patch("preprocessing.catboost_features.load_gmm_bundle", side_effect=AssertionError("PSF6 must not use GMM")):
            self.assertEqual(len(build_catboost_features(6, baseline())), 14)


class EnvironmentModelAndApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()):
            cls.model = load_catboost_model(6)
        rng = random.Random(42)
        cls.scenarios = [
            {field: f"{field}.{rng.randrange(1, len(categories) + 1)}" for field, (_, categories) in ENVIRONMENT_FIELDS.items()}
            for _ in range(512)
        ]
        cls.frame = pd.DataFrame(
            [build_catboost_features(6, scenario) for scenario in cls.scenarios],
            columns=cls.model.feature_names_,
        )
        cls.probabilities = cls.model.predict_proba(cls.frame)
        from app import app
        cls.client = app.test_client()

    def test_saved_model_contract(self):
        self.assertEqual(CATBOOST_CONFIG[6]["features"], ENVIRONMENT_FEATURES)
        self.assertEqual(self.model.feature_names_, [name for name, _ in ENVIRONMENT_FEATURES])
        self.assertEqual(self.model.get_cat_feature_indices(), list(range(14)))
        self.assertEqual(set(self.model.classes_), {"보통", "나쁨"})
        self.assertEqual(self.model.get_metadata()["input_schema"], ENVIRONMENT_SCHEMA_VERSION)
        self.assertEqual(self.model.get_metadata()["input_schema_sha256"], ENVIRONMENT_SCHEMA_SHA256)

    def test_schema_mismatch_cannot_load_as_raw14(self):
        with patch.dict("predictors.catboost.catboost_models", {}, clear=True):
            with patch.dict(CATBOOST_CONFIG[6], {"input_schema_sha256": "incompatible"}):
                with self.assertRaisesRegex(ValueError, "스키마"):
                    load_catboost_model(6)

    def test_both_classes_and_probability_order_across_all_prediction_routes(self):
        predicted = np.asarray(self.model.classes_)[self.probabilities.argmax(axis=1)]
        self.assertEqual(set(predicted), {"보통", "나쁨"})
        for label in ("보통", "나쁨"):
            index = int(np.flatnonzero(predicted == label)[0])
            scenario = self.scenarios[index]
            with contextlib.redirect_stdout(io.StringIO()):
                expected = predict_psf_catboost(6, scenario)
                debug = predict_psf_catboost_test(6, scenario)
            self.assertEqual(expected["predicted_text"], "F.6.1" if label == "보통" else "F.6.2")
            for code, name in (("F.6.1", "보통"), ("F.6.2", "나쁨")):
                self.assertAlmostEqual(expected["probabilities"][code], self.probabilities[index, list(self.model.classes_).index(name)])
            self.assertEqual(debug["prediction"], label)
            self.assertEqual(debug["model_input"], self.frame.iloc[index].to_dict())
            for route in ("/api/predict/6", "/api/catboost/predict/6", "/api/predict/catboost/6"):
                with self.subTest(route=route, label=label), contextlib.redirect_stdout(io.StringIO()):
                    response = self.client.post(route, json=scenario)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.get_json()["result"], expected)
            for route in ("/api/predict", "/api/catboost/predict", "/api/predict/catboost"):
                for psf_key in ("F.6", "psf6"):
                    with self.subTest(route=route, key=psf_key, label=label), contextlib.redirect_stdout(io.StringIO()):
                        response = self.client.post(route, json={psf_key: scenario})
                        self.assertEqual(response.status_code, 200)
                        self.assertEqual(response.get_json()["results"], [expected])
            response = self.client.post("/api/test/predict/6", json=scenario)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["result"], debug)

    def test_invalid_environment_input_returns_400(self):
        invalid_inputs = ({}, [], {**baseline(), "F.6.S.10": 1}, {**baseline(), "F.6.S.11": "F.6.S.10.2"})
        for scenario in invalid_inputs:
            for route in ("/api/predict/6", "/api/catboost/predict/6", "/api/predict/catboost/6", "/api/test/predict/6"):
                with self.subTest(route=route, scenario=scenario):
                    response = self.client.post(route, json=scenario)
                    self.assertEqual(response.status_code, 400)
                    self.assertFalse(response.get_json()["success"])
            for route in ("/api/predict", "/api/catboost/predict", "/api/predict/catboost"):
                with self.subTest(route=route, scenario=scenario):
                    self.assertEqual(self.client.post(route, json={"F.6": scenario}).status_code, 400)
        self.assertEqual(self.client.post("/api/predict", json={}).get_json()["results"], [])
        for route in ("/api/predict", "/api/catboost/predict", "/api/predict/catboost"):
            for key in ("F.6", "psf6"):
                self.assertEqual(self.client.post(route, json={key: None}).status_code, 400)


if __name__ == "__main__":
    unittest.main()
