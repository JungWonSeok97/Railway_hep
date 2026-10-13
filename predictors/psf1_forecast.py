from datetime import (
    datetime,
    time,
    timedelta,
    timezone,
)


KST = timezone(
    timedelta(hours=9),
    name="Asia/Seoul",
)

PSF1_PREDICTED_TEXT = {
    0: "F.1.1",
    1: "F.1.2",
    2: "F.1.3",
}

WORK_TIME_BOUNDARIES = (
    1.3,
    5.6,
    10.6,
    12.8,
)


def _prediction(predicted_label):
    return {
        "predicted_label": predicted_label,
        "predicted_text": (
            PSF1_PREDICTED_TEXT[predicted_label]
        ),
    }


def _parse_work_started_at(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "work_started_at은 필수입니다."
        )

    try:
        parsed = datetime.fromisoformat(
            value.strip().replace("Z", "+00:00")
        )
    except ValueError as error:
        raise ValueError(
            "work_started_at은 ISO 8601 형식이어야 "
            "합니다."
        ) from error

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=KST)

    return parsed.astimezone(KST)


def _elapsed_time_floor(elapsed_hours):
    if elapsed_hours <= 1.3:
        return 0

    if elapsed_hours < 5.6:
        return 1

    if elapsed_hours < 10.6:
        return 0

    if elapsed_hours < 12.8:
        return 1

    return 2


def _time_of_day_floor(target_datetime):
    minutes = (
        target_datetime.hour * 60
        + target_datetime.minute
        + target_datetime.second / 60
    )

    daytime_start = 6 * 60 + 24
    daytime_end = 18 * 60 + 24

    if daytime_start <= minutes < daytime_end:
        return 0

    return 1


def _corrected_psf1(
    base_psf1,
    work_started_at,
    target_datetime,
):
    elapsed_hours = (
        target_datetime - work_started_at
    ).total_seconds() / 3600

    return max(
        base_psf1,
        _elapsed_time_floor(elapsed_hours),
        _time_of_day_floor(target_datetime),
    )


def _candidate_boundaries(
    work_started_at,
    generated_at,
):
    final_boundary = (
        work_started_at
        + timedelta(hours=12.8)
    )

    boundaries = {generated_at}

    if generated_at >= final_boundary:
        return [generated_at]

    for elapsed_hours in WORK_TIME_BOUNDARIES:
        boundary = (
            work_started_at
            + timedelta(hours=elapsed_hours)
        )

        if generated_at < boundary <= final_boundary:
            boundaries.add(boundary)

    boundary_date = generated_at.date()
    final_date = final_boundary.date()

    while boundary_date <= final_date:
        for boundary_time in (
            time(hour=6, minute=24),
            time(hour=18, minute=24),
        ):
            boundary = datetime.combine(
                boundary_date,
                boundary_time,
                tzinfo=KST,
            )

            if generated_at < boundary < final_boundary:
                boundaries.add(boundary)

        boundary_date += timedelta(days=1)

    return sorted(boundaries)


def _build_segments(
    base_psf1,
    work_started_at,
    generated_at,
):
    boundaries = _candidate_boundaries(
        work_started_at,
        generated_at,
    )
    segments = []

    for index, start_at in enumerate(boundaries):
        if index + 1 < len(boundaries):
            end_at = boundaries[index + 1]
            probe_at = start_at + (
                end_at - start_at
            ) / 2
        else:
            end_at = None
            probe_at = start_at

        predicted_label = _corrected_psf1(
            base_psf1,
            work_started_at,
            probe_at,
        )

        start_elapsed_hours = (
            start_at - generated_at
        ).total_seconds() / 3600

        if end_at is None:
            end_elapsed_hours = None
        else:
            end_elapsed_hours = (
                end_at - generated_at
            ).total_seconds() / 3600

        segment = {
            "start_elapsed_hours": round(
                start_elapsed_hours,
                2,
            ),
            "end_elapsed_hours": (
                round(end_elapsed_hours, 2)
                if end_elapsed_hours is not None
                else None
            ),
            "predicted_label": predicted_label,
            "start_at": start_at.isoformat(),
            "end_at": (
                end_at.isoformat()
                if end_at is not None
                else None
            ),
        }

        if (
            segments
            and segments[-1]["predicted_label"]
            == predicted_label
        ):
            segments[-1]["end_at"] = segment["end_at"]
            segments[-1]["end_elapsed_hours"] = (
                segment["end_elapsed_hours"]
            )
        else:
            segments.append(segment)

    return segments


def build_psf1_forecast(
    work_started_at_value,
    base_psf1,
    generated_at=None,
):
    if type(base_psf1) is not int or base_psf1 not in (
        0,
        1,
        2,
    ):
        raise ValueError(
            "base_psf1은 0, 1, 2 중 하나여야 "
            "합니다."
        )

    work_started_at = _parse_work_started_at(
        work_started_at_value
    )

    if generated_at is None:
        generated_at = datetime.now(KST)
    elif generated_at.tzinfo is None:
        generated_at = generated_at.replace(
            tzinfo=KST
        )
    else:
        generated_at = generated_at.astimezone(KST)

    generated_at = generated_at.replace(
        microsecond=0
    )

    if work_started_at > generated_at:
        raise ValueError(
            "근무 시작 일시는 현재 시각보다 "
            "미래일 수 없습니다."
        )

    elapsed_hours = (
        generated_at - work_started_at
    ).total_seconds() / 3600
    current_label = _corrected_psf1(
        base_psf1,
        work_started_at,
        generated_at,
    )

    return {
        "psf_id": "F.1",
        "generated_at": generated_at.isoformat(),
        "work_started_at": (
            work_started_at.isoformat()
        ),
        "elapsed_hours": round(elapsed_hours, 2),
        "base_prediction": _prediction(base_psf1),
        "current_prediction": _prediction(
            current_label
        ),
        "segments": _build_segments(
            base_psf1,
            work_started_at,
            generated_at,
        ),
    }
