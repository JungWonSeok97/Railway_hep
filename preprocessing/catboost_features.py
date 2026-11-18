import os

import joblib
import numpy as np

from config.paths import GMM_MODEL_DIR
from config.catboost_mappings import (
    CATBOOST_ALLOWED_VALUES,
    CATBOOST_CODE_TO_VALUE,
    canonical_condition_key,
)
from config.catboost_config import CATBOOST_CONFIG
from preprocessing.environment_features import normalize_environment_conditions


# =====================================================
# GMM 사용하는 PSF
# =====================================================

GMM_PSF_IDS = {
    2,
    4,
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
    PSF6 이외의 선택 코드를 학습 당시 숫자로 변환한다.
    PSF6의 기존 숫자 매핑은 검증·호환용이며 실제 추론은 원래 범주를 사용한다.
    숫자 입력은 이미 학습 인코딩된 값이며 문항별 허용 범위를 검사한다.

    예:
        F.2.S.1.1
        F.2.1.1
        0
        1
        2
        "0"
        "1"
        "2"
    """

    # PSF6 runtime uses raw categories via environment_features instead.
    condition_key = canonical_condition_key(condition_key)
    if condition_key not in CATBOOST_ALLOWED_VALUES:
        raise ValueError(f"지원하지 않는 문항입니다: {condition_key}")

    # ---------------------------------
    # F.x.S.x.x 코드
    # ---------------------------------

    if (
        isinstance(value, str)
        and value in CATBOOST_CODE_TO_VALUE
    ):
        code_field = canonical_condition_key(value.rsplit(".", 1)[0])
        if code_field != condition_key:
            raise ValueError(
                f"{condition_key}에 다른 문항의 선택 코드를 사용할 수 없습니다: {value}"
            )
        value = CATBOOST_CODE_TO_VALUE[value]

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
                f"문항별 학습 인코딩 숫자여야 합니다: "
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

    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{condition_key} 값이 허용 범위를 벗어났습니다.") from exc

    if not np.isfinite(numeric) or numeric not in CATBOOST_ALLOWED_VALUES[condition_key]:
        allowed = ", ".join(
            str(number) for number in sorted(CATBOOST_ALLOWED_VALUES[condition_key])
        )
        raise ValueError(
            f"{condition_key}의 학습 인코딩 값은 {allowed} 중 하나여야 합니다: "
            f"{value}. 숫자의 의미가 불확실하면 해당 문항의 선택 코드를 사용하세요."
        )

    return int(value)


# =====================================================
# GMM bundle 로드
# =====================================================

def load_gmm_bundle(psf_id):
    """
    PSF별 저장된 GMM bundle을 로드한다.

    예:
        models/gmm/psf2_gmm.joblib
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
        [작업 도구의 적절성, 장비의 사용성]
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

def normalize_catboost_conditions(psf_id, conditions, *, require_all=True):
    """Validate original questions before GMM or model inference.

    Equivalent current/legacy aliases are accepted after value normalization.
    PSF4's optional training_elapsed_months is formula metadata, not a CatBoost
    feature; accepting it preserves requests shared with the default endpoint.
    """
    if psf_id not in CATBOOST_CONFIG:
        raise ValueError(f"지원하지 않는 PSF ID입니다: {psf_id}")
    if not isinstance(conditions, dict):
        raise ValueError(f"F.{psf_id} 입력은 JSON 객체여야 합니다.")
    if psf_id == 6:
        return normalize_environment_conditions(conditions)

    expected = list(dict.fromkeys(
        key for _, keys in CATBOOST_CONFIG[psf_id]["features"] for key in keys
    ))
    resolved = {}
    for key, value in conditions.items():
        if not isinstance(key, str):
            raise ValueError(f"F.{psf_id} 문항 키는 문자열이어야 합니다.")
        if psf_id == 4 and key == "training_elapsed_months":
            continue
        field = canonical_condition_key(key)
        if field not in expected:
            raise ValueError(f"F.{psf_id}에 지원하지 않는 문항 키입니다: {key}")
        normalized = normalize_catboost_condition(value, field)
        if field in resolved and resolved[field] != normalized:
            raise ValueError(f"{field}의 기본 키와 별칭 값이 서로 다릅니다.")
        resolved[field] = normalized
    missing = [field for field in expected if field not in resolved]
    if require_all and missing:
        raise ValueError(
            f"F.{psf_id}에는 {len(expected)}개 문항이 모두 필요합니다. 누락: "
            + ", ".join(missing)
        )
    return resolved


def build_catboost_features(
    psf_id,
    conditions,
):
    """
    기존 API 입력을
    CatBoost 학습 당시 Feature 형태로 변환한다.

    PSF6은 GMM 없이 원래 14개 응답 범주를 반환한다.

    단일 변수:
        0 / 1 / 2 직접 사용

    복합 변수:
        저장된 GMM.predict()
        -> label_map
        -> 0 / 1 / 2
    """

    conditions = normalize_catboost_conditions(psf_id, conditions)
    if psf_id == 6:
        return [conditions[keys[0]] for _, keys in CATBOOST_CONFIG[6]["features"]]

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

            values.append(conditions[condition_key])

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
