import os

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from config.paths import CATBOOST_MODEL_DIR
from config.mappings import mappings
from config.catboost_config import CATBOOST_CONFIG
from preprocessing.catboost_features import build_catboost_features


# 로드된 모델 캐시
catboost_models = {}


def load_catboost_model(psf_id):
    """PSF별 CatBoost 모델 로드"""

    if psf_id not in CATBOOST_CONFIG:
        raise ValueError(
            f"지원하지 않는 PSF ID입니다: {psf_id}"
        )

    # 이미 메모리에 있으면 재사용
    if psf_id in catboost_models:
        return catboost_models[psf_id]

    config = CATBOOST_CONFIG[psf_id]

    model_path = os.path.join(
        CATBOOST_MODEL_DIR,
        config["filename"],
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"CatBoost 모델 파일을 찾을 수 없습니다: "
            f"{model_path}"
        )

    model = CatBoostClassifier()
    model.load_model(model_path)

    # ----------------------------------------
    # 학습 Feature와 API Feature 이름 검증
    # ----------------------------------------

    expected_features = [
        feature_name
        for feature_name, _
        in config["features"]
    ]

    actual_features = list(
        model.feature_names_
    )

    if actual_features != expected_features:
        raise ValueError(
            f"PSF{psf_id} CatBoost Feature 불일치\n"
            f"expected={expected_features}\n"
            f"actual={actual_features}"
        )

    catboost_models[psf_id] = model

    print(
        f"PSF{psf_id} CatBoost 로드 완료: "
        f"{model_path}"
    )

    return model


def predict_psf_catboost(
    psf_id,
    conditions,
):
    """단일 PSF CatBoost 예측"""

    model = load_catboost_model(
        psf_id
    )

    config = CATBOOST_CONFIG[
        psf_id
    ]

    # ----------------------------------------
    # API 입력 -> 모델 Feature
    # ----------------------------------------

    feature_names = [
        feature_name
        for feature_name, _
        in config["features"]
    ]

    feature_values = (
        build_catboost_features(
            psf_id,
            conditions,
        )
    )

    model_input = pd.DataFrame(
        [feature_values],
        columns=feature_names,
    )

    # 디버깅할 때 유용
    print(
        f"\n[PSF{psf_id}] CatBoost 입력"
    )
    print(model_input)

    # ----------------------------------------
    # 확률 예측
    # ----------------------------------------

    raw_probabilities = np.asarray(
        model.predict_proba(
            model_input
        )
    )[0]

    model_classes = [
        str(cls)
        for cls in model.classes_
    ]

    label_map = mappings[
        psf_id
    ]["label_map"]

    probabilities = {}
    canonical_probabilities = []

    # ----------------------------------------
    # 모델 클래스 순서를 API 출력 순서로 정렬
    # ----------------------------------------

    for label_index, label_text in label_map.items():

        if label_text not in model_classes:
            raise ValueError(
                f"PSF{psf_id} 모델에 "
                f"'{label_text}' 클래스가 없습니다.\n"
                f"model.classes_={model_classes}"
            )

        model_class_index = (
            model_classes.index(
                label_text
            )
        )

        probability = float(
            raw_probabilities[
                model_class_index
            ]
        )

        output_code = (
            f"F.{psf_id}."
            f"{label_index + 1}"
        )

        probabilities[
            output_code
        ] = probability

        canonical_probabilities.append(
            probability
        )

    # ----------------------------------------
    # 최종 예측
    # ----------------------------------------

    predicted_label = int(
        np.argmax(
            canonical_probabilities
        )
    )

    predicted_text = (
        f"F.{psf_id}."
        f"{predicted_label + 1}"
    )

    return {
        "psf_id": f"F.{psf_id}",
        "predicted_label": predicted_label,
        "predicted_text": predicted_text,
        "probabilities": probabilities,
    }


from config.mappings import (
    mappings,
    PSF_NAMES,
)


def predict_psf_catboost_test(
    psf_id,
    conditions,
):
    """
    모델 동작 확인용 예측.

    - 최종 예측값을 한글로 반환
    - 클래스별 확률을 한글 + %로 반환
    - 실제 CatBoost 입력 Feature도 반환
    """

    model = load_catboost_model(
        psf_id
    )

    config = CATBOOST_CONFIG[
        psf_id
    ]

    # ----------------------------------------
    # API 입력 -> 실제 CatBoost 입력
    # ----------------------------------------

    feature_names = [
        feature_name
        for feature_name, _
        in config["features"]
    ]

    feature_values = (
        build_catboost_features(
            psf_id,
            conditions,
        )
    )

    model_input = pd.DataFrame(
        [feature_values],
        columns=feature_names,
    )

    # ----------------------------------------
    # 예측
    # ----------------------------------------

    raw_probabilities = np.asarray(
        model.predict_proba(
            model_input
        )
    )[0]

    model_classes = [
        str(cls)
        for cls in model.classes_
    ]

    label_map = mappings[
        psf_id
    ]["label_map"]

    # ----------------------------------------
    # 사람이 보기 쉬운 확률
    # ----------------------------------------

    probabilities = {}

    probability_values = []

    for _, korean_label in label_map.items():

        if korean_label not in model_classes:
            raise ValueError(
                f"PSF{psf_id} 모델에 "
                f"'{korean_label}' 클래스가 없습니다."
            )

        class_index = model_classes.index(
            korean_label
        )

        probability = float(
            raw_probabilities[
                class_index
            ]
        )

        probabilities[
            korean_label
        ] = f"{probability * 100:.2f}%"

        probability_values.append(
            probability
        )

    # ----------------------------------------
    # 최종 예측
    # ----------------------------------------

    predicted_label = int(
        np.argmax(
            probability_values
        )
    )

    predicted_korean = label_map[
        predicted_label
    ]

    # ----------------------------------------
    # CatBoost 실제 입력값
    # ----------------------------------------

    model_input_dict = {
        name: int(value)
        for name, value
        in zip(
            feature_names,
            feature_values,
        )
    }

    return {
        "psf_id": f"F.{psf_id}",
        "psf": PSF_NAMES[psf_id],

        "prediction": predicted_korean,

        "probabilities": probabilities,

        "model_input": model_input_dict,
    }