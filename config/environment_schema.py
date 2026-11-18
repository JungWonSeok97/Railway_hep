"""PSF6 raw categorical inputs, shared by training and inference.

Option order is the survey code suffix, not the current HTML display order.
Keep the original survey wording (including its rain-unit typo) as model tokens.
"""

import hashlib
import json


ENVIRONMENT_FIELDS = {
    "F.6.S.1": ("풍속", (
        "풍속이 10m/s 미만인 경우", "풍속 10m/s 이상인 경우",
    )),
    "F.6.S.2": ("비", (
        "비가 10mm/h(눈의 경우 1cm/h) 미만",
        "비가 10mm/s(눈의 경우 1cm/s) 이상",
    )),
    "F.6.S.3": ("실외 습도", (
        "실외 습도가 90% 이하인 경우", "실외 습도가 90% 이상인 경우",
    )),
    "F.6.S.4": ("실외 온도", (
        "실외 온도가 35℃(체감 온도) 이하인 경우",
        "실외 온도가 35℃(체감 온도) 이상인 경우",
    )),
    "F.6.S.5": ("낙뢰", (
        "낙뢰가 발생하지 않은 경우", "낙뢰가 발생할 경우",
    )),
    "F.6.S.6": ("안개", (
        "안개가 발생하지 않은 경우", "안개가 발생할 경우",
    )),
    "F.6.S.7": ("황사", (
        "황사로 인해 작업자가 확실히 사물을 볼 수 없는 경우",
        "황사로 인해 작업자가 숨 쉬기가 불편할 경우",
    )),
    "F.6.S.8": ("작업공간 크기", (
        "작업공간의 크기가 10m^3 이상인 경우",
        "작업공간의 크기가 10m^3 이하인 경우",
    )),
    "F.6.S.9": ("마찰계수", (
        "바닥 (정지)마찰계수(SCOF)가 < 0.63인 경우",
        "바닥 (정지)마찰계수(SCOF)가 ≥ 0.63인 경우",
    )),
    "F.6.S.10": ("실내 온도", (
        "실내 온도가 17℃ 이하인 경우", "실내 온도가 17~28℃인 경우",
        "실내 온도가 28℃ 이상인 경우",
    )),
    "F.6.S.11": ("실내 습도", (
        "실내 습도가 40% 이하인 경우", "실내 습도가 40~75%인 경우",
        "실내 습도가 75% 이상인 경우",
    )),
    "F.6.S.12": ("통풍", (
        "통풍이 잘 될 경우", "통풍이 잘 되지 않을 경우",
    )),
    "F.6.S.13": ("소음", (
        "소음이 85dB 이하인 경우", "소음이 85~115dB인 경우",
        "소음이 115dB 이상인 경우",
    )),
    "F.6.S.14": ("업무공간 가시성", (
        "보통작업을 수행할 정도의 조도가 형성된 경우",
        "정밀작업을 수행할 정도의 조도가 형성된 경우",
        "초정밀작업을 수행할 정도의 조도가 형성된 경우",
    )),
}

# Match the sorted original columns used in the controlled comparison.
ENVIRONMENT_FEATURES = sorted(
    (name, [field]) for field, (name, _) in ENVIRONMENT_FIELDS.items()
)
ENVIRONMENT_SCHEMA_VERSION = "psf6-raw-text-14-v1"
ENVIRONMENT_SCHEMA_SHA256 = hashlib.sha256(
    json.dumps(
        {"fields": ENVIRONMENT_FIELDS, "features": ENVIRONMENT_FEATURES},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
).hexdigest()
ENVIRONMENT_MODEL_FILENAME = "환경_raw14_catboost.cbm"
