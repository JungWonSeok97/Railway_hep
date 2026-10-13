import os

import joblib
import numpy as np

from config.paths import GMM_MODEL_DIR
from config.mappings import (
    code_to_value,
    get_condition_value,
)
from config.catboost_config import CATBOOST_CONFIG


# =====================================================
# GMM 사용하는 PSF
# =====================================================

GMM_PSF_IDS = {
    2,
    4,
    6,
    8,
}


# =====================================================
# 로드된 GMM bundle 캐시
# =====================================================

gmm_bundles = {}


# =====================================================
# API 입력값 -> 0 / 1 / 2
# =====================================================

def normalize_catboost_condition(
    value,
    condition_key,
):
    """
    다음 형식을 모두 0 / 1 / 2로 변환한다.

    예:
        F.6.S.10.1
        F.6.10.1
        0
        1
        2
        "0"
        "1"
        "2"
    """

    # ---------------------------------
    # F.x.S.x.x 코드
    # ---------------------------------

    if (
        isinstance(value, str)
        and value in code_to_value
    ):
        value = code_to_value[value]

    # ---------------------------------
    # 숫자 문자열
    # ---------------------------------

    elif isinstance(value, str):

        try:
            value = float(value)

        except ValueError as exc:

            raise ValueError(
                f"{condition_key} 값은 "
                f"F.n.S.n.n 코드 또는 "
                f"0/1/2 숫자여야 합니다: "
                f"{value}"
            ) from exc

    # ---------------------------------
    # 형식 검증
    # ---------------------------------

    if (
        isinstance(value, bool)
        or not isinstance(
            value,
            (
                int,
                float,
                np.integer,
                np.floating,
            ),
        )
    ):
        raise ValueError(
            f"{condition_key} 값의 형식이 "
            f"올바르지 않습니다: {value}"
        )

    if (
        not np.isfinite(value)
        or float(value)
        not in (0.0, 1.0, 2.0)
    ):
        raise ValueError(
            f"{condition_key} 값은 "
            f"0, 1, 2 중 하나여야 합니다: "
            f"{value}"
        )

    return int(value)


# =====================================================
# GMM bundle 로드
# =====================================================

def load_gmm_bundle(psf_id):
    """
    PSF별 저장된 GMM bundle을 로드한다.

    예:
        models/gmm/psf6_gmm.joblib
    """

    if psf_id not in GMM_PSF_IDS:
        return None

    # 이미 로드했다면 재사용
    if psf_id in gmm_bundles:
        return gmm_bundles[psf_id]

    model_path = os.path.join(
        GMM_MODEL_DIR,
        f"psf{psf_id}_gmm.joblib",
    )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"PSF{psf_id} GMM 파일을 "
            f"찾을 수 없습니다: {model_path}"
        )

    bundle = joblib.load(
        model_path
    )

    if not isinstance(bundle, dict):
        raise ValueError(
            f"PSF{psf_id} GMM 파일 형식이 "
            f"올바르지 않습니다."
        )

    gmm_bundles[psf_id] = bundle

    print(
        f"PSF{psf_id} GMM 로드 완료: "
        f"{model_path}"
    )

    return bundle


# =====================================================
# 복합 Feature 생성
# =====================================================

def predict_composite_feature(
    psf_id,
    feature_name,
    values,
):
    """
    저장된 GMM으로 API 입력 한 건의
    복합 Feature를 계산한다.

    예:
        [실내온도, 실내습도, 통풍]
        -> GMM cluster
        -> label_map
        -> 0 / 1 / 2
    """

    bundle = load_gmm_bundle(
        psf_id
    )

    if feature_name not in bundle:

        raise KeyError(
            f"PSF{psf_id} GMM bundle에 "
            f"{feature_name}이 없습니다. "
            f"저장된 Feature: "
            f"{list(bundle.keys())}"
        )

    feature_bundle = bundle[
        feature_name
    ]

    gmm = feature_bundle.get(
        "gmm"
    )

    label_map = feature_bundle.get(
        "label_map"
    )

    group_cols = feature_bundle.get(
        "group_cols",
        [],
    )

    if gmm is None:
        raise ValueError(
            f"{feature_name}은 "
            f"GMM 모델이 없습니다."
        )

    if label_map is None:
        raise ValueError(
            f"{feature_name}의 "
            f"label_map이 없습니다."
        )

    # ---------------------------------
    # 학습 당시 변수 개수 검증
    # ---------------------------------

    if (
        group_cols
        and len(values) != len(group_cols)
    ):
        raise ValueError(
            f"{feature_name} 입력 변수 개수 "
            f"불일치: "
            f"expected={len(group_cols)}, "
            f"actual={len(values)}, "
            f"group_cols={group_cols}"
        )

    # ---------------------------------
    # 개별 API 요청 1건 예측
    # ---------------------------------

    X = np.asarray(
        [values],
        dtype=float,
    )

    cluster = int(
        gmm.predict(X)[0]
    )

    # joblib 저장 시 int key이므로
    # 우선 int로 검색
    if cluster in label_map:

        state = label_map[
            cluster
        ]

    # 혹시 문자열 key로 저장된 경우 대응
    elif str(cluster) in label_map:

        state = label_map[
            str(cluster)
        ]

    else:
        raise KeyError(
            f"{feature_name}의 "
            f"cluster {cluster}에 대한 "
            f"label_map이 없습니다."
        )

    return int(state)


# =====================================================
# API 입력 -> CatBoost 입력 Feature
# =====================================================

def build_catboost_features(
    psf_id,
    conditions,
):
    """
    기존 API 입력을
    CatBoost 학습 당시 Feature 형태로 변환한다.

    단일 변수:
        0 / 1 / 2 직접 사용

    복합 변수:
        저장된 GMM.predict()
        -> label_map
        -> 0 / 1 / 2
    """

    if not isinstance(
        conditions,
        dict,
    ):
        raise ValueError(
            f"F.{psf_id} 입력은 "
            f"JSON 객체여야 합니다."
        )

    if psf_id not in CATBOOST_CONFIG:

        raise ValueError(
            f"지원하지 않는 PSF ID입니다: "
            f"{psf_id}"
        )

    config = CATBOOST_CONFIG[
        psf_id
    ]

    feature_values = []

    # =================================
    # CatBoost Feature별 처리
    # =================================

    for (
        feature_name,
        condition_keys,
    ) in config["features"]:

        values = []

        # -----------------------------
        # API 입력 -> 0 / 1 / 2
        # -----------------------------

        for condition_key in condition_keys:

            raw_value = get_condition_value(
                conditions,
                condition_key,
                1,
            )

            value = (
                normalize_catboost_condition(
                    raw_value,
                    condition_key,
                )
            )

            values.append(
                value
            )

        # =============================
        # 단일 Feature
        # =============================

        if len(condition_keys) == 1:

            feature_value = values[
                0
            ]

        # =============================
        # GMM 복합 Feature
        # =============================

        else:

            if psf_id not in GMM_PSF_IDS:

                raise ValueError(
                    f"PSF{psf_id}의 "
                    f"{feature_name}은 "
                    f"복합 Feature이지만 "
                    f"GMM 설정이 없습니다."
                )

            feature_value = (
                predict_composite_feature(
                    psf_id=psf_id,
                    feature_name=feature_name,
                    values=values,
                )
            )

        # ---------------------------------
        # CatBoost에서 범주형으로 학습했으므로
        # 문자열로 전달
        # ---------------------------------

        feature_values.append(
            str(feature_value)
        )

    return feature_values