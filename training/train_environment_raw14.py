"""Train PSF6 on the original 14 categorical inputs, without GMM.

Run from Railway_hep: python -B training/train_environment_raw14.py --cross-validate
Only the new raw14 model and its aggregate metadata are written. The survey,
notebooks, other PSF models and legacy environment artifacts remain unchanged.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import sklearn
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.environment_schema import (
    ENVIRONMENT_FEATURES, ENVIRONMENT_FIELDS, ENVIRONMENT_MODEL_FILENAME,
    ENVIRONMENT_SCHEMA_SHA256, ENVIRONMENT_SCHEMA_VERSION,
)

PARAMS = dict(
    iterations=500, depth=6, learning_rate=0.05, l2_leaf_reg=20,
    grow_policy="Lossguide", max_leaves=32, min_data_in_leaf=2,
    random_strength=10, bagging_temperature=2, loss_function="MultiClass",
    eval_metric="TotalF1", random_seed=42, verbose=False,
    thread_count=2, allow_writing_files=False,
)
SURVEY = ROOT / "철도_코레일 종사자 설문조사 결과.txt"
NOTEBOOK = ROOT / "CatBoost_Result/preprocess_v1.4.ipynb"
NOTEBOOK_SHA256 = "e4b8fd330b1cc193656e535e0fc2146763d831256e11fe39f04d0cd545461d48"
PREPROCESS_CELLS = [4, 5, 11, 13, 16, 18, 23, 30, 35, 50, 51, 52, 53, 54, 55, 56]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_training_data():
    """Reproduce the reviewed quality_95 filter and original response order.

    Reuse only pinned, non-writing notebook cells to preserve the identity and
    survey-session rules. Do not execute notebook export/model-training cells.
    Identifiers stay in memory and are excluded from the returned data/report.
    """
    if sha256(NOTEBOOK) != NOTEBOOK_SHA256:
        raise RuntimeError("전처리 노트북이 변경되었습니다. 지정한 셀을 검토한 후 해시를 갱신하세요.")
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    frame = pd.DataFrame(json.loads(SURVEY.read_text(encoding="utf-8")))
    frame["questions"] = frame.questions.map(
        lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
    )
    namespace = {"pd": pd, "json": json, "df": frame}
    with contextlib.redirect_stdout(io.StringIO()):
        for index in PREPROCESS_CELLS:
            if index == 30:
                namespace["answer_df"] = namespace["df_answer"].copy()
            code = "".join(notebook["cells"][index]["source"])
            exec(compile(code, f"original-preprocessing-cell-{index}", "exec"), namespace)
    answers = namespace["answer_df_new_quality"]
    selected = answers.loc[
        (answers.PSF == 6) & (answers.quality_95 == "성실") & answers.answer.notna()
    ]
    raw = pd.DataFrame(selected.conditions.tolist()).reset_index(drop=True)
    columns = [name for name, _ in ENVIRONMENT_FEATURES]
    if set(raw.columns) != set(columns) or raw.isna().any().any():
        raise ValueError("환경 학습 데이터의 원변수 14개 또는 결측값을 확인하세요.")
    raw = raw[columns].astype(str)
    for name, categories in ENVIRONMENT_FIELDS.values():
        if not set(raw[name]).issubset(categories):
            raise ValueError(f"환경 {name}에 정의되지 않은 범주가 있습니다.")
    labels = selected.answer.reset_index(drop=True).astype(str)
    if set(labels) != {"보통", "나쁨"}:
        raise ValueError("환경 학습 데이터에 보통/나쁨 두 클래스가 필요합니다.")
    # A tuple avoids separator collisions and retains the original factorization order.
    groups = pd.factorize(pd.Series(list(raw.itertuples(index=False, name=None))), sort=False)[0]
    return raw, labels, groups


def evaluate(labels, probabilities, classes, groups):
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    matrix = confusion_matrix(labels, predicted, labels=classes)
    counts = {label: int((predicted == label).sum()) for label in classes}
    first = np.unique(groups, return_index=True)[1]
    return {
        "n_responses": len(labels), "classes": classes,
        "prediction_counts": counts,
        "prediction_ratios": {label: count / len(labels) for label, count in counts.items()},
        "unique_scenario_prediction_counts": {
            label: int((predicted[first] == label).sum()) for label in classes
        },
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "confusion_matrix_actual_rows_predicted_columns": matrix.tolist(),
        "recall": dict(zip(classes, (matrix.diagonal() / matrix.sum(axis=1)).tolist())),
    }


def train(output_dir, cross_validate=False):
    raw, labels, groups = load_training_data()
    classes = sorted(labels.unique().tolist())
    print(f"PSF6: {len(labels)} responses, {len(set(groups))} original scenarios, 14 categorical features", flush=True)
    report = {
        "psf": 6, "model_filename": ENVIRONMENT_MODEL_FILENAME,
        "input_schema": ENVIRONMENT_SCHEMA_VERSION,
        "input_schema_sha256": ENVIRONMENT_SCHEMA_SHA256,
        "source_sha256": sha256(SURVEY), "preprocessing_notebook_sha256": sha256(NOTEBOOK),
        "preprocessing_cells": PREPROCESS_CELLS, "filter": "PSF == 6, quality_95 == 성실, answer present",
        "params": PARAMS,
        "versions": {"python": platform.python_version(), "catboost": catboost.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__},
        "n_responses": len(labels), "n_original_scenarios": len(set(groups)),
        "label_counts": {key: int(value) for key, value in labels.value_counts().items()},
        "features": list(raw.columns), "categories": ENVIRONMENT_FIELDS,
        "observed_categories": {name: sorted(raw[name].unique().tolist()) for name in raw},
        "unobserved_categories": {
            name: sorted(set(categories) - set(raw[name]))
            for name, categories in ENVIRONMENT_FIELDS.values() if set(categories) - set(raw[name])
        },
        "dataset_fingerprint": hashlib.sha256(
            json.dumps({"x": raw.values.tolist(), "y": labels.tolist()}, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
    }
    if cross_validate:
        probabilities = np.full((len(labels), len(classes)), np.nan)
        folds = []
        for fold, (train_indices, test_indices) in enumerate(GroupKFold(n_splits=5).split(raw, labels, groups), 1):
            if set(groups[train_indices]) & set(groups[test_indices]):
                raise RuntimeError("학습/평가 시나리오가 중복되었습니다.")
            model = CatBoostClassifier(**PARAMS)
            model.fit(raw.iloc[train_indices], labels.iloc[train_indices], cat_features=list(raw.columns))
            order = [list(model.classes_).index(label) for label in classes]
            probabilities[test_indices] = model.predict_proba(raw.iloc[test_indices])[:, order]
            folds.append({"fold": fold, "n_train": len(train_indices), "n_test": len(test_indices), "scenario_overlap": 0})
            print(f"GroupKFold {fold}/5 complete", flush=True)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("OOF 평가에 누락된 예측이 있습니다.")
        report["group_kfold_oof"] = {
            "split": "GroupKFold(n_splits=5, shuffle=False); group=original 14-variable scenario",
            "scope": "response-weighted, unseen scenarios; not respondent-held-out or an external test set",
            "folds": folds, **evaluate(labels, probabilities, classes, groups),
        }

    model = CatBoostClassifier(**PARAMS, metadata={
        "input_schema": ENVIRONMENT_SCHEMA_VERSION,
        "input_schema_sha256": ENVIRONMENT_SCHEMA_SHA256,
        "source_sha256": report["source_sha256"],
    })
    model.fit(raw, labels, cat_features=list(raw.columns))
    report["full_fit_in_sample_only"] = evaluate(labels, model.predict_proba(raw), list(model.classes_), groups)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ENVIRONMENT_MODEL_FILENAME
    temporary = target.with_suffix(".cbm.tmp")
    model.save_model(str(temporary))
    restored = CatBoostClassifier().load_model(str(temporary))
    if list(restored.feature_names_) != list(raw.columns):
        raise RuntimeError("저장된 모델의 특성 순서가 일치하지 않습니다.")
    np.testing.assert_allclose(restored.predict_proba(raw), model.predict_proba(raw), rtol=0, atol=0)
    temporary.replace(target)
    report["model_sha256"] = sha256(target)
    report_path = output_dir / "environment_raw14_metadata.json"
    temporary_report = report_path.with_suffix(".json.tmp")
    temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary_report.replace(report_path)
    print(json.dumps({
        "model": str(target), "in_sample_prediction_counts": report["full_fit_in_sample_only"]["prediction_counts"],
        "oof_prediction_counts": report.get("group_kfold_oof", {}).get("prediction_counts"),
    }, ensure_ascii=False), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--cross-validate", action="store_true", help="Run 5-fold scenario-grouped OOF evaluation before final fitting")
    arguments = parser.parse_args()
    train(arguments.output_dir, arguments.cross_validate)
