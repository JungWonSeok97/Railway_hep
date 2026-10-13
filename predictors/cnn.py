import os

# TensorFlow import 전에 환경변수 지정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault(
    "LOKY_MAX_CPU_COUNT",
    "1",
)

import json
import random

import numpy as np
import tensorflow as tf

from tensorflow.keras.models import (
    Sequential,
    load_model,
)
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.utils import (
    to_categorical,
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
)

from sklearn.model_selection import (
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
)

from config.paths import (
    SURVEY_DATA_PATH,
    MODEL_DIR,
)
from config.mappings import (
    mappings,
    code_to_value,
    get_condition_value,
    get_num_classes,
)


# =====================================================
# 재현성
# =====================================================

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# =====================================================
# CNN 모델 캐시
# =====================================================

models = {}


# =====================================================
# CNN 구조
# =====================================================

def build_cnn_model(
    n_features,
    num_classes,
):
    model = Sequential([
        Conv1D(
            filters=64,
            kernel_size=2,
            activation="relu",
            input_shape=(
                n_features,
                1,
            ),
            padding="same",
        ),

        BatchNormalization(),

        Conv1D(
            filters=128,
            kernel_size=2,
            activation="relu",
            padding="same",
        ),

        BatchNormalization(),

        Conv1D(
            filters=64,
            kernel_size=2,
            activation="relu",
            padding="same",
        ),

        BatchNormalization(),

        Flatten(),

        Dense(
            128,
            activation="relu",
        ),

        Dropout(0.4),

        Dense(
            64,
            activation="relu",
        ),

        Dropout(0.3),

        Dense(
            num_classes,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# =====================================================
# 모델 호환성
# =====================================================

def is_model_compatible(
    model,
    psf_id,
):
    expected_features = len(
        mappings[
            psf_id
        ]["english_cols"]
    )

    expected_classes = (
        get_num_classes(
            psf_id
        )
    )

    actual_features = (
        model.input_shape[1]
    )

    actual_classes = (
        model.output_shape[-1]
    )

    return (
        actual_features
        == expected_features
        and actual_classes
        == expected_classes
    )


# =====================================================
# Class Weight
# =====================================================

def compute_class_weights(y):

    from collections import Counter

    counter = Counter(y)

    total = len(y)
    n_classes = len(counter)

    class_weights = {}

    for cls, count in counter.items():

        class_weights[cls] = (
            total
            / (
                n_classes
                * count
            )
        )

    return class_weights


# =====================================================
# CNN 학습
# =====================================================

def train_model_for_psf(psf_id):

    print(
        f"\n=== PSF{psf_id} "
        f"모델 학습 시작 ==="
    )

    num_classes = (
        get_num_classes(
            psf_id
        )
    )

    # ---------------------------------
    # 정답 Label
    # ---------------------------------

    if psf_id == 1:
        answer_to_label012 = {
            "보통": 0,
            "나쁨": 1,
            "매우나쁨": 2,
        }

    elif psf_id in (2, 3, 4, 7):
        answer_to_label012 = {
            "좋음": 0,
            "보통": 1,
            "나쁨": 2,
        }

    elif psf_id in (5, 6):
        answer_to_label012 = {
            "보통": 0,
            "나쁨": 1,
        }

    elif psf_id == 8:
        answer_to_label012 = {
            "보통": 0,
            "약간 높음": 1,
            "매우 높음": 2,
        }

    else:
        raise ValueError(
            f"지원하지 않는 PSF ID: "
            f"{psf_id}"
        )

    # ---------------------------------
    # 데이터 로드
    # ---------------------------------

    with open(
        SURVEY_DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    rows = []

    for resp in data:

        for q in resp["questions"]:

            if q["id"] != psf_id:
                continue

            rows.append({
                "answer_text": q["answer"],
                "conditions": q["conditions"],
            })

    if not rows:

        print(
            f"PSF{psf_id}에 대한 "
            f"데이터가 없습니다."
        )

        return None

    psf_map = mappings[
        psf_id
    ]

    X_list = []
    y_list = []

    for row in rows:

        if (
            row["answer_text"]
            not in answer_to_label012
        ):
            continue

        y_list.append(
            answer_to_label012[
                row["answer_text"]
            ]
        )

        conds = row[
            "conditions"
        ]

        # ---------------------------------
        # PSF1
        # ---------------------------------

        if psf_id == 1:

            x_row = [
                psf_map["work_time_map"].get(
                    conds.get("근무시간"),
                    1,
                ),
                psf_map["sleep_time_map"].get(
                    conds.get("수면시간"),
                    1,
                ),
                psf_map["task_time_map"].get(
                    conds.get("작업시간"),
                    1,
                ),
                psf_map["shift_map"].get(
                    conds.get("근무시간대"),
                    1,
                ),
                psf_map["allowed_time_map"].get(
                    conds.get("작업 허용 시간"),
                    1,
                ),
            ]

        # ---------------------------------
        # PSF2
        # ---------------------------------

        elif psf_id == 2:

            x_row = [
                psf_map["equip_use_map"].get(
                    conds.get("장비의 사용성"),
                    1,
                ),
                psf_map["maint_map"].get(
                    conds.get("유지보수의 적절성"),
                    1,
                ),
                psf_map["tool_map"].get(
                    conds.get("작업 도구의 적절성"),
                    1,
                ),
                psf_map["link_map"].get(
                    conds.get("타 기기와의 연계성"),
                    1,
                ),
                psf_map["avail_map"].get(
                    conds.get("장비 가용성(접근성)"),
                    1,
                ),
                psf_map["manual_map"].get(
                    conds.get("장비 사용 매뉴얼의 질"),
                    1,
                ),
                psf_map["itsys_map"].get(
                    conds.get("컴퓨터 지원 시스템의 적절성"),
                    1,
                ),
            ]

        # ---------------------------------
        # PSF3
        # ---------------------------------

        elif psf_id == 3:

            x_row = [
                psf_map["consistency_map"].get(
                    conds.get("(직무)절차의 일관성"),
                    1,
                ),
                psf_map["traceability_map"].get(
                    conds.get("(직무)절차의 추적성"),
                    1,
                ),
                psf_map["availability_map"].get(
                    conds.get("(직무)절차의 유무(가용성)"),
                    1,
                ),
                psf_map["clarity_map"].get(
                    conds.get("(직무)절차의 명료성(구체성)"),
                    1,
                ),
                psf_map["adequacy_map"].get(
                    conds.get("(직무)절차의 적절성(완전성 포함)"),
                    1,
                ),
                psf_map["complexity_map"].get(
                    conds.get("(직무)절차 기술(description)의 복잡성"),
                    1,
                ),
            ]

        # ---------------------------------
        # PSF4
        # ---------------------------------

        elif psf_id == 4:

            x_row = [
                psf_map["edu_period_map"].get(
                    conds.get("교육 주기(빈도)"),
                    1,
                ),
                psf_map["eval_freq_map"].get(
                    conds.get("직무능력 평가 빈도"),
                    1,
                ),
                psf_map["proc_skill_map"].get(
                    conds.get("절차적 지식의 숙련도"),
                    1,
                ),
                psf_map["equip_skill_map"].get(
                    conds.get("장비 사용 지식의 숙련도"),
                    1,
                ),
                psf_map["program_map"].get(
                    conds.get("교육/훈련 프로그램 적절성"),
                    1,
                ),
                psf_map["system_skill_map"].get(
                    conds.get("시스템 지식(원리적 지식)의 숙련도"),
                    1,
                ),
                psf_map["experience_map"].get(
                    conds.get(
                        "상황에 대한 경험 정도(자주 또는 드물게 겪는지)"
                    ),
                    1,
                ),
            ]

        # ---------------------------------
        # PSF5
        # ---------------------------------

        elif psf_id == 5:

            x_row = [
                psf_map["msg_len_map"].get(
                    conds.get("의사소통 메시지의 길이"),
                    1,
                ),
                psf_map["comm_autonomy_map"].get(
                    conds.get("팀/조직 의사소통 자율성"),
                    1,
                ),
                psf_map["protocol_specificity_map"].get(
                    conds.get("의사소통 프로토콜의 구체성"),
                    1,
                ),
                psf_map["situation_share_map"].get(
                    conds.get("의사소통자 간 상황인식 공유 정도"),
                    1,
                ),
                psf_map["device_adequacy_map"].get(
                    conds.get("의사소통을 위한 장비/기기의 적절성"),
                    1,
                ),
                psf_map["jargon_need_map"].get(
                    conds.get("의사소통을 위한 전문용어 숙지 필요성"),
                    1,
                ),
            ]

        # ---------------------------------
        # PSF6
        # ---------------------------------

        elif psf_id == 6:

            x_row = [
                psf_map["rain_map"].get(
                    conds.get("비"),
                    1,
                ),
                psf_map["noise_map"].get(
                    conds.get("소음"),
                    1,
                ),
                psf_map["fog_map"].get(
                    conds.get("안개"),
                    1,
                ),
                psf_map["vent_map"].get(
                    conds.get("통풍"),
                    1,
                ),
                psf_map["wind_map"].get(
                    conds.get("풍속"),
                    1,
                ),
                psf_map["dust_map"].get(
                    conds.get("황사"),
                    1,
                ),
                psf_map["friction_map"].get(
                    conds.get("마찰계수"),
                    1,
                ),
                psf_map["indoor_hum_map"].get(
                    conds.get("실내 습도"),
                    1,
                ),
                psf_map["indoor_temp_map"].get(
                    conds.get("실내 온도"),
                    1,
                ),
                psf_map["outdoor_hum_map"].get(
                    conds.get("실외 습도"),
                    1,
                ),
                psf_map["outdoor_temp_map"].get(
                    conds.get("실외 온도"),
                    1,
                ),
                psf_map["workspace_size_map"].get(
                    conds.get("작업공간 크기"),
                    1,
                ),
                psf_map["visibility_map"].get(
                    conds.get("업무공간 가시성"),
                    1,
                ),
                psf_map["thunder_map"].get(
                    conds.get("낙뢰"),
                    1,
                ),
            ]

        # ---------------------------------
        # PSF7
        # ---------------------------------

        elif psf_id == 7:

            x_row = [
                psf_map["role_division_map"].get(
                    conds.get("역할 분담 명료화"),
                    1,
                ),
                psf_map["plan_adequacy_map"].get(
                    conds.get("작업 계획 적절성"),
                    1,
                ),
                psf_map["procedure_rule_map"].get(
                    conds.get("작업절차 및 규정"),
                    1,
                ),
                psf_map["collaboration_map"].get(
                    conds.get("작업자간 협업 정도"),
                    1,
                ),
                psf_map["supervision_map"].get(
                    conds.get("작업 관리 및 감독 적절성"),
                    1,
                ),
            ]

        # ---------------------------------
        # PSF8
        # ---------------------------------

        elif psf_id == 8:

            x_row = [
                psf_map["info_uncertainty_map"].get(
                    conds.get("정보의 불확실성"),
                    1,
                ),
                psf_map["observed_info_amount_map"].get(
                    conds.get("관측되는 정보의 양"),
                    1,
                ),
                psf_map["decision_alternatives_map"].get(
                    conds.get("의사결정 대안의 수"),
                    1,
                ),
                psf_map["goal_clarity_map"].get(
                    conds.get("의사결정 목표의 명확성"),
                    1,
                ),
                psf_map["criteria_count_map"].get(
                    conds.get("의사결정 판단 기준의 수"),
                    1,
                ),
                psf_map["simul_decisions_map"].get(
                    conds.get("(동시에 처리 되어야 하는)의사결정 수"),
                    1,
                ),
                psf_map["inferred_info_amount_map"].get(
                    conds.get(
                        "(관측이 불가능하여)추론해야하는 정보의 양"
                    ),
                    1,
                ),
            ]

        X_list.append(
            x_row
        )

    X = np.array(
        X_list,
        dtype="float32",
    )

    y = np.array(
        y_list
    )

    n_features = X.shape[1]

    class_weights = (
        compute_class_weights(y)
    )

    print(
        f"PSF{psf_id} 클래스 가중치: "
        f"{class_weights}"
    )

    # ---------------------------------
    # Train / Test
    # ---------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train_cnn = X_train.reshape(
        -1,
        n_features,
        1,
    )

    X_test_cnn = X_test.reshape(
        -1,
        n_features,
        1,
    )

    y_train_cat = to_categorical(
        y_train,
        num_classes=num_classes,
    )

    # 기존 코드와 동일하게 유지
    y_test_cat = to_categorical(
        y_test,
        num_classes=num_classes,
    )

    model = build_cnn_model(
        n_features,
        num_classes,
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=0,
    )

    model.fit(
        X_train_cnn,
        y_train_cat,
        validation_split=0.2,
        epochs=150,
        batch_size=16,
        callbacks=[
            early_stopping
        ],
        class_weight=class_weights,
        verbose=0,
    )

    y_pred = np.argmax(
        model.predict(
            X_test_cnn,
            verbose=0,
        ),
        axis=1,
    )

    acc = (
        accuracy_score(
            y_test,
            y_pred,
        )
        * 100
    )

    print(
        f"PSF{psf_id} 모델 학습 완료 "
        f"- 정확도: {acc:.2f}%"
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"psf{psf_id}_model.h5",
    )

    model.save(
        model_path
    )

    print(
        f"PSF{psf_id} 모델 저장됨: "
        f"{model_path}"
    )

    return model


# =====================================================
# CNN 모델 로드
# =====================================================

def load_or_train_model(psf_id):

    model_path = os.path.join(
        MODEL_DIR,
        f"psf{psf_id}_model.h5",
    )

    model_is_current = (
        os.path.exists(
            model_path
        )
        and
        os.path.getmtime(
            model_path
        )
        >=
        os.path.getmtime(
            SURVEY_DATA_PATH
        )
    )

    if model_is_current:

        print(
            f"PSF{psf_id} 저장된 모델 "
            f"로드 중: {model_path}"
        )

        model = load_model(
            model_path
        )

        if not is_model_compatible(
            model,
            psf_id,
        ):
            print(
                f"PSF{psf_id} 저장 모델 구조 "
                f"불일치 - 재학습 시작"
            )

            return train_model_for_psf(
                psf_id
            )

        print(
            f"PSF{psf_id} 모델 로드 완료"
        )

        return model

    reason = (
        "저장된 모델 없음"
        if not os.path.exists(model_path)
        else "학습 데이터 변경됨"
    )

    print(
        f"PSF{psf_id} {reason} "
        f"- 새로 학습 시작"
    )

    return train_model_for_psf(
        psf_id
    )


# =====================================================
# CNN 예측
# =====================================================

def predict_psf(
    psf_id,
    conditions,
):

    if psf_id not in models:

        models[psf_id] = (
            load_or_train_model(
                psf_id
            )
        )

    model = models[
        psf_id
    ]

    if model is None:
        return {
            "error": (
                f"PSF{psf_id} 모델을 "
                f"로드할 수 없습니다."
            )
        }

    psf_map = mappings[
        psf_id
    ]

    english_cols = psf_map.get(
        "english_cols",
        [],
    )

    x_new = []

    for col in english_cols:

        val = get_condition_value(
            conditions,
            col,
            1,
        )

        if (
            isinstance(val, str)
            and val in code_to_value
        ):
            val = code_to_value[
                val
            ]

        x_new.append(
            val
        )

    n_features = len(
        x_new
    )

    x_input = np.array(
        x_new,
        dtype="float32",
    ).reshape(
        1,
        n_features,
        1,
    )

    pred_proba = model.predict(
        x_input,
        verbose=0,
    )

    probs = pred_proba[
        0
    ]

    pred_label = int(
        np.argmax(probs)
    )

    label_map = psf_map[
        "label_map"
    ]

    output_code_map = {
        i: f"F.{psf_id}.{i + 1}"
        for i in range(
            len(label_map)
        )
    }

    probabilities = {
        output_code_map[i]: float(
            probs[i]
        )
        for i in range(
            len(label_map)
        )
    }

    return {
        "psf_id": f"F.{psf_id}",
        "predicted_label": pred_label,
        "predicted_text": output_code_map[
            pred_label
        ],
        "probabilities": probabilities,
    }