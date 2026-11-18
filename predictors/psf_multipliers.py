"""PSF multipliers only: no occupation coefficients or HEP calculation."""

import math


# Tuple positions follow the API output code suffix (F.n.1, F.n.2, ...).
PSF_MULTIPLIERS = {
    1: (1.0, 14.45, 17.54),
    2: (0.492, 1.0, 5.654),
    3: (0.292, 1.0, 8.559),
    4: (0.421, 1.0, 7.229),
    5: (1.0, 9.559),
    6: (1.0, 11.576),
    7: (0.322, 1.0, 12.11),
    8: (1.0, 4.654, 11.873),
}
EXPECTED_PSF_IDS = tuple(f"F.{psf}" for psf in PSF_MULTIPLIERS)


def multiplier_for_prediction(psf_id, predicted_text):
    for index, multiplier in enumerate(PSF_MULTIPLIERS.get(psf_id, ()), 1):
        if predicted_text == f"F.{psf_id}.{index}":
            return multiplier
    raise ValueError(f"PSF{psf_id}의 예측 코드가 올바르지 않습니다: {predicted_text}")


def summarize_psf_results(results):
    """Return a full product only when each of the eight PSFs is present once.

    Partial prediction requests remain supported, but missing PSFs are never
    silently assigned 1. Validate multipliers produced by the prediction layer.
    """
    seen = set()
    factors = {}
    for result in results:
        psf_id = result.get("psf_id")
        if psf_id not in EXPECTED_PSF_IDS or psf_id in seen:
            raise ValueError(f"PSF 결과에 중복 또는 잘못된 ID가 있습니다: {psf_id}")
        seen.add(psf_id)
        factor = result.get("psf_multiplier")
        if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"{psf_id} 배수는 0보다 큰 유한한 숫자여야 합니다.")
        factors[psf_id] = factor
    missing = [psf_id for psf_id in EXPECTED_PSF_IDS if psf_id not in seen]
    product = None if missing else math.prod(factors[psf_id] for psf_id in EXPECTED_PSF_IDS)
    if product is not None and (not math.isfinite(product) or product <= 0):
        raise ValueError("PSF 배수의 곱이 유효한 숫자 범위를 벗어났습니다.")
    return {
        "results": results,
        "psf_multiplier_product": product,
        "missing_psfs": missing,
    }
