import math


PSF4_PREDICTED_TEXT = {
    0: "F.4.1",
    1: "F.4.2",
    2: "F.4.3",
}


def _calculate_hep_multiplier(
    training_elapsed_months,
):
    x = float(training_elapsed_months)

    # 같은 공식을 큰 입력값에서도 overflow 없이 계산한다.
    if x == 0:
        fraction = 0.0
    else:
        normalized = x / 18.555

        if normalized <= 1:
            normalized_power = normalized ** 4
            fraction = (
                normalized_power
                / (1 + normalized_power)
            )
        else:
            inverse_power = (
                1 / normalized
            ) ** 4
            fraction = 1 / (1 + inverse_power)

    return 0.33 + 4.5 * fraction


def _validate_training_elapsed_months(
    training_elapsed_months,
):
    if (
        isinstance(training_elapsed_months, bool)
        or not isinstance(
            training_elapsed_months,
            (int, float),
        )
    ):
        raise ValueError(
            "training_elapsed_months는 "
            "숫자여야 합니다."
        )

    training_elapsed_months = float(
        training_elapsed_months
    )

    if (
        not math.isfinite(training_elapsed_months)
        or training_elapsed_months < 0
    ):
        raise ValueError(
            "training_elapsed_months는 "
            "0 이상의 유한한 숫자여야 합니다."
        )

    return training_elapsed_months


def predict_psf4_formula(psf_input):
    if not isinstance(psf_input, dict):
        raise ValueError(
            "F.4는 JSON 객체여야 합니다."
        )

    if "training_elapsed_months" not in psf_input:
        raise ValueError(
            "F.4.training_elapsed_months는 "
            "필수입니다."
        )

    training_elapsed_months = (
        _validate_training_elapsed_months(
            psf_input.get(
                "training_elapsed_months"
            )
        )
    )

    hep_multiplier = _calculate_hep_multiplier(
        training_elapsed_months
    )

    if hep_multiplier < 1:
        predicted_label = 0
    elif hep_multiplier < 2:
        predicted_label = 1
    else:
        predicted_label = 2

    return {
        "predicted_label": predicted_label,
        "predicted_text": (
            PSF4_PREDICTED_TEXT[predicted_label]
        ),
        "probabilities": None,
        "psf_id": "F.4",
        "hep_multiplier": round(
            hep_multiplier,
            3,
        ),
    }


def build_psf4_forecast(
    training_elapsed_months,
):
    training_elapsed_months = (
        _validate_training_elapsed_months(
            training_elapsed_months
        )
    )

    hep_multipliers = []

    for step_index in range(601):
        future_months = step_index / 10
        formula_months = (
            training_elapsed_months
            + future_months
        )
        hep_multipliers.append(
            round(
                _calculate_hep_multiplier(
                    formula_months
                ),
                3,
            )
        )

    if training_elapsed_months.is_integer():
        response_elapsed_months = int(
            training_elapsed_months
        )
    else:
        response_elapsed_months = (
            training_elapsed_months
        )

    return {
        "psf_id": "F.4",
        "training_elapsed_months": (
            response_elapsed_months
        ),
        "series": {
            "x_start": 0.0,
            "x_end": 60.0,
            "x_step": 0.1,
            "hep_multipliers": hep_multipliers,
        },
    }
