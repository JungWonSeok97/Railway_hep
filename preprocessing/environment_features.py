"""Convert PSF6 choices directly to the 14 original categorical values."""

import math
from numbers import Real

from config.catboost_mappings import CATBOOST_OPTION_VALUES, canonical_condition_key
from config.environment_schema import ENVIRONMENT_FIELDS, ENVIRONMENT_FEATURES


def _text_key(value):
    return " ".join(
        value.replace("m³", "m^3")
        .replace("발생하지 않을 경우", "발생하지 않은 경우").split()
    )


def normalize_environment_condition(value, condition_key):
    """Resolve a choice code/text, or an unambiguous legacy numeric value.

    Legacy indoor temperature/humidity value 1 merged low and high levels.
    It cannot be converted back without guessing, so callers must send a code.
    """
    if not isinstance(condition_key, str):
        raise ValueError("환경 문항 키는 문자열이어야 합니다.")
    field = canonical_condition_key(condition_key)
    if field not in ENVIRONMENT_FIELDS:
        raise ValueError(f"지원하지 않는 환경 문항입니다: {field}")
    _, categories = ENVIRONMENT_FIELDS[field]

    if isinstance(value, str):
        for index, category in enumerate(categories, 1):
            codes = (f"{field}.{index}", f"{field.replace('.S.', '.')}.{index}")
            if value in codes or _text_key(value) == _text_key(category):
                return category
        try:
            value = float(value)
        except ValueError as exc:
            raise ValueError(f"{field}에 해당하는 선택 코드 또는 원래 응답 문구가 필요합니다.") from exc

    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field}의 환경 입력값이 올바르지 않습니다.")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field}의 환경 입력값이 허용 범위를 벗어났습니다.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{field}의 환경 입력값이 올바르지 않습니다.")
    matches = [
        category for category, encoded in zip(categories, CATBOOST_OPTION_VALUES[field])
        if value == encoded
    ]
    if len(matches) > 1:
        raise ValueError(
            f"{field}의 기존 숫자 {value:g}은 낮음/높음을 구분할 수 없습니다. "
            "선택 코드(F.6.S.n.n) 또는 원래 응답 문구를 보내세요."
        )
    if not matches:
        raise ValueError(f"{field}에 유효한 선택 코드 또는 원래 응답 문구를 보내세요.")
    return matches[0]


def normalize_environment_conditions(conditions):
    """Validate all 14 fields and resolve aliases to original categories."""
    if not isinstance(conditions, dict):
        raise ValueError("F.6 입력은 JSON 객체여야 합니다.")
    resolved = {}
    for key, value in conditions.items():
        if not isinstance(key, str):
            raise ValueError("환경 문항 키는 문자열이어야 합니다.")
        field = canonical_condition_key(key)
        category = normalize_environment_condition(value, field)
        if field in resolved and resolved[field] != category:
            raise ValueError(f"{field}의 기본 키와 별칭 값이 서로 다릅니다.")
        resolved[field] = category
    missing = [field for field in ENVIRONMENT_FIELDS if field not in resolved]
    if missing:
        raise ValueError("F.6에는 14개 문항이 모두 필요합니다. 누락: " + ", ".join(missing))
    return resolved


def build_environment_features(conditions):
    """Validate all 14 fields and return raw strings in saved model order."""
    resolved = normalize_environment_conditions(conditions)
    return [resolved[keys[0]] for _, keys in ENVIRONMENT_FEATURES]
