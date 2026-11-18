"""Regression checks against the shipped training workbook and model artifacts.

Run from Railway_hep: python -m unittest discover -s tests -p test_catboost_encoding.py
The XLSX reader uses the standard library; no Excel or openpyxl dependency is needed.
"""

import contextlib
import io
import sys
import types
import unittest
import warnings
from collections import defaultdict
from html.parser import HTMLParser
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.catboost_config import CATBOOST_CONFIG
from config.catboost_mappings import (
    CATBOOST_ALLOWED_VALUES,
    CATBOOST_CODE_TO_VALUE,
    CATBOOST_OPTION_VALUES,
)
from config.mappings import PSF_NAMES, code_to_value as cnn_code_to_value

# config.paths creates a CNN output directory at import time. These read-only
# tests need only the existing GMM directory, so do not import that side effect.
_paths = types.ModuleType("config.paths")
_paths.GMM_MODEL_DIR = str(ROOT / "models" / "gmm")
with patch.dict(sys.modules, {"config.paths": _paths}):
    from preprocessing.catboost_features import (
        build_catboost_features,
        normalize_catboost_condition,
    )


def normalize_text(text):
    """Only normalize the two known spelling variants in the UI/workbook."""
    return " ".join(
        text.replace("m³", "m^3")
        .replace("발생하지 않은 경우", "발생하지 않을 경우")
        .split()
    )


def workbook_rows():
    namespace = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(ROOT / "processed_data" / "subpsf_level_mapping.xlsx") as book:
        strings = []
        if "xl/sharedStrings.xml" in book.namelist():
            shared = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
            strings = [
                "".join(node.itertext())
                for item in shared.findall("s:si", namespace)
                for node in [item]
            ]
        sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall("s:sheetData/s:row", namespace):
            cells = {}
            for cell in row.findall("s:c", namespace):
                column = "".join(c for c in cell.attrib["r"] if c.isalpha())
                value_node = cell.find("s:v", namespace)
                value = value_node.text if value_node is not None else ""
                if cell.attrib.get("t") == "s":
                    value = strings[int(value)]
                elif cell.attrib.get("t") == "inlineStr":
                    value = "".join(
                        node.text or ""
                        for node in cell.findall("s:is//s:t", namespace)
                    )
                cells[column] = value
            if cells.get("A") and cells["A"] != "PSF_name":
                yield cells["A"], cells["B"], cells["C"], int(cells["D"])


class SurveyOptions(HTMLParser):
    """Associate options with the same house/select ordering used at submit."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.options = []
        self.fields = []
        self.house = 0
        self.house_depth = None
        self.div_depth = 0
        self.question = 0
        self.field = None
        self.option = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div":
            self.div_depth += 1
            if "house" in attrs.get("class", "").split():
                self.house += 1
                self.house_depth = self.div_depth
                self.question = 0
        elif tag == "select" and "room-select" in attrs.get("class", "").split():
            if self.house_depth is None:
                raise AssertionError("A survey select is outside its PSF house")
            self.question += 1
            self.field = f"F.{self.house}.S.{self.question}"
            self.fields.append(self.field)
        elif tag == "option" and self.field:
            self.option = {
                "field": self.field,
                "code": attrs.get("data-condition-code"),
                "value": attrs.get("value", ""),
                "disabled": "disabled" in attrs,
                "text": "",
            }

    def handle_data(self, data):
        if self.option is not None:
            self.option["text"] += data

    def handle_endtag(self, tag):
        if tag == "option" and self.option is not None:
            if self.option["value"] and not self.option["disabled"]:
                self.options.append(self.option)
            self.option = None
        elif tag == "select":
            self.field = None
        elif tag == "div":
            if self.div_depth == self.house_depth:
                self.house_depth = None
            self.div_depth -= 1


class CatBoostEncodingTests(unittest.TestCase):
    def setUp(self):
        # Keep existing model-version warnings visible. Production supplies
        # arrays in the saved column order, which the reference checks below
        # verify; that known feature-name warning adds no signal to this sweep.
        warning_context = warnings.catch_warnings()
        warning_context.__enter__()
        self.addCleanup(warning_context.__exit__, None, None, None)
        warnings.simplefilter("once", UserWarning)
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but GaussianMixture was fitted with feature names",
            category=UserWarning,
        )

    @classmethod
    def setUpClass(cls):
        cls.survey = SurveyOptions()
        cls.survey.feed((ROOT / "input.html").read_text(encoding="utf-8"))
        cls.reference = {}
        for psf, column, text, value in workbook_rows():
            key = (psf, normalize_text(text))
            if key in cls.reference:
                raise AssertionError(f"Ambiguous workbook option: {key}")
            cls.reference[key] = (column, value)
        cls.options_by_field = defaultdict(list)
        for option in cls.survey.options:
            cls.options_by_field[option["field"]].append(option)

    def reference_option(self, option):
        psf = int(option["field"].split(".")[1])
        return self.reference[(PSF_NAMES[psf], normalize_text(option["text"]))]

    def test_all_57_fields_and_157_options_match_training_workbook(self):
        self.assertEqual(len(self.survey.fields), 57)
        self.assertEqual(len(set(self.survey.fields)), 57)
        self.assertEqual(len(self.survey.options), 157)
        self.assertEqual(len(self.reference), 157)
        self.assertEqual(set(self.survey.fields), set(CATBOOST_OPTION_VALUES))
        seen_codes = set()
        seen_reference = set()
        for option in self.survey.options:
            field, code = option["field"], option["code"]
            with self.subTest(field=field, text=option["text"]):
                self.assertIsInstance(code, str)
                self.assertEqual(code.rsplit(".", 1)[0], field)
                self.assertNotIn(code, seen_codes)
                seen_codes.add(code)
                _, expected = self.reference_option(option)
                self.assertEqual(normalize_catboost_condition(code, field), expected)
                self.assertEqual(CATBOOST_CODE_TO_VALUE[code], expected)
                legacy_code = code.replace(".S.", ".")
                legacy_field = field.replace(".S.", ".")
                self.assertEqual(normalize_catboost_condition(legacy_code, field), expected)
                self.assertEqual(normalize_catboost_condition(code, legacy_field), expected)
                self.assertEqual(normalize_catboost_condition(legacy_code, legacy_field), expected)
                seen_reference.add((PSF_NAMES[int(field.split(".")[1])], normalize_text(option["text"])))
        self.assertEqual(seen_reference, set(self.reference))
        self.assertEqual(len(CATBOOST_CODE_TO_VALUE), 314)

    def test_numeric_inputs_are_training_values_in_each_fields_domain(self):
        for field, options in self.options_by_field.items():
            expected_domain = frozenset(self.reference_option(o)[1] for o in options)
            with self.subTest(field=field):
                self.assertEqual(CATBOOST_ALLOWED_VALUES[field], expected_domain)
                for value in expected_domain:
                    for supplied in (value, float(value), str(value), f"{value}.0"):
                        self.assertEqual(normalize_catboost_condition(supplied, field), value)
                for invalid in {0, 1, 2} - expected_domain:
                    for supplied in (invalid, str(invalid)):
                        with self.assertRaises(ValueError):
                            normalize_catboost_condition(supplied, field)

    def test_invalid_values_and_codes_from_other_questions_are_rejected(self):
        for invalid in (None, True, False, -1, 3, 0.5, float("nan"), float("inf"), [], {}, "nan", "inf", "", "not a code"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                normalize_catboost_condition(invalid, "F.1.S.3")
        for field, code in (
            ("F.1.S.3", "F.8.S.7.2"),
            ("F.1.3", "F.8.7.2"),
            ("F.2.S.1", "F.1.S.3.1"),
            ("F.1.S.3", "F.1.S.3.4"),
        ):
            with self.subTest(field=field, code=code), self.assertRaises(ValueError):
                normalize_catboost_condition(code, field)
        with self.assertRaises(ValueError):
            normalize_catboost_condition(0, "F.99.S.1")

    def test_cnn_encoding_is_not_changed_by_catboost_correction(self):
        # The legacy CNN has a separate training contract; this patch does not
        # silently change its input schema while retaining old CNN weights.
        self.assertEqual(cnn_code_to_value["F.2.S.1.2"], 2)
        self.assertEqual(CATBOOST_CODE_TO_VALUE["F.2.S.1.2"], 1)
        self.assertEqual(cnn_code_to_value["F.6.S.10.2"], 1)
        self.assertEqual(CATBOOST_CODE_TO_VALUE["F.6.S.10.2"], 0)

    def test_saved_gmm_features_match_workbook_encoded_inputs(self):
        for psf, config in CATBOOST_CONFIG.items():
            if psf == 6:
                # Raw PSF6 categories are covered by test_environment_raw14.py.
                continue
            fields = [f for f in self.survey.fields if f.startswith(f"F.{psf}.S.")]
            baseline = {field: self.options_by_field[field][0] for field in fields}
            bundle_path = ROOT / "models" / "gmm" / f"psf{psf}_gmm.joblib"
            bundle = joblib.load(bundle_path) if bundle_path.exists() else {}
            # Change each option once against a valid reference background.
            # Expected raw values come from XLSX text, not the production map.
            for field in fields:
                for option in self.options_by_field[field]:
                    choices = {**baseline, field: option}
                    expected_raw = {f: self.reference_option(o)[1] for f, o in choices.items()}
                    values_by_column = dict(self.reference_option(o) for o in choices.values())
                    expected_features = []
                    for feature, condition_keys in config["features"]:
                        if len(condition_keys) == 1:
                            expected_features.append(str(expected_raw[condition_keys[0]]))
                        else:
                            saved = bundle[feature]
                            raw = [[values_by_column[column] for column in saved["group_cols"]]]
                            cluster = int(saved["gmm"].predict(np.asarray(raw, dtype=float))[0])
                            expected_features.append(str(saved["label_map"][cluster]))
                    coded = {f: o["code"] for f, o in choices.items()}
                    legacy = {f.replace(".S.", "."): value for f, value in coded.items()}
                    with self.subTest(psf=psf, code=option["code"]), contextlib.redirect_stdout(io.StringIO()):
                        self.assertEqual(build_catboost_features(psf, coded), expected_features)
                        self.assertEqual(build_catboost_features(psf, legacy), expected_features)

    def test_equipment_code_reaches_the_category_used_by_saved_model(self):
        model = CatBoostClassifier()
        model.load_model(str(ROOT / "models" / "장비_및_기기_catboost.cbm"))
        rows = []
        for value in (0, 1, 2):
            row = {feature: "1" for feature in model.feature_names_}
            row["Equipment_Accessibility"] = str(value)
            rows.append(row)
        probabilities = model.predict_proba(pd.DataFrame(rows, columns=model.feature_names_))
        np.testing.assert_allclose(probabilities[0], probabilities[2])
        self.assertFalse(np.allclose(probabilities[0], probabilities[1]))
        corrected = normalize_catboost_condition("F.2.S.1.2", "F.2.S.1")
        self.assertEqual(corrected, 1)


if __name__ == "__main__":
    unittest.main()
