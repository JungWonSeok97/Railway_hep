"""PSF output labels, display names, and supported request field aliases."""


PSF_LABELS = {
    1: {0: '보통', 1: '나쁨', 2: '매우나쁨'},
    2: {0: '좋음', 1: '보통', 2: '나쁨'},
    3: {0: '좋음', 1: '보통', 2: '나쁨'},
    4: {0: '좋음', 1: '보통', 2: '나쁨'},
    5: {0: '보통', 1: '나쁨'},
    6: {0: '보통', 1: '나쁨'},
    7: {0: '좋음', 1: '보통', 2: '나쁨'},
    8: {0: '보통', 1: '약간 높음', 2: '매우 높음'},
}

PSF_NAMES = {
    1: '작업부하',
    2: '장비 및 기기',
    3: '절차 및 지침',
    4: '교육 및 훈련',
    5: '의사소통',
    6: '환경',
    7: '작업절차',
    8: '복잡성',
}


def get_legacy_condition_key(condition_key):
    """
    F.1.S.1 -> F.1.1
    """
    return condition_key.replace(
        ".S.",
        ".",
        1,
    )


def get_condition_value(
    conditions,
    condition_key,
    default=1,
):
    """
    신규 키와 기존 키를 모두 지원한다.
    """

    if condition_key in conditions:
        return conditions[condition_key]

    legacy_key = get_legacy_condition_key(
        condition_key
    )

    return conditions.get(
        legacy_key,
        default,
    )
