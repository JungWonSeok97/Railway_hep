"""Numeric input encoding for PSF1/2/3/4/5/7/8 and legacy PSF6.

Source: processed_data/subpsf_level_mapping.xlsx (Sub-PSF LV. -> level).
Tuple positions are stable selection-code suffixes, NOT display order or risk
rank. In particular, binary values are 0/1 and indoor temperature/humidity
encode both extremes as 1. Keep the existing CNN encoding in mappings.py
separate: those models have their own training history.

The active PSF6 model uses environment_schema.py raw categories instead.
Its old numbers here are only for unambiguous backwards-compatible inputs
and validation of the original workbook; they do not feed an environment GMM.
"""


CATBOOST_OPTION_VALUES = {
    # Workload: sleep codes retain .1=short, .2=middle, .3=long.
    "F.1.S.1": (0, 1, 2),
    "F.1.S.2": (0, 1, 2),
    "F.1.S.3": (2, 1, 0),
    "F.1.S.4": (0, 1, 2),
    "F.1.S.5": (0, 1, 2),
    # Equipment and devices.
    "F.2.S.1": (0, 1),
    "F.2.S.2": (0, 1, 2),
    "F.2.S.3": (0, 1, 2),
    "F.2.S.4": (0, 1, 2),
    "F.2.S.5": (0, 1, 2),
    "F.2.S.6": (0, 1, 2),
    "F.2.S.7": (0, 1, 2),
    # Procedures: complexity codes retain .1=high, .2=middle, .3=low.
    "F.3.S.1": (0, 1),
    "F.3.S.2": (2, 1, 0),
    "F.3.S.3": (0, 1, 2),
    "F.3.S.4": (0, 1, 2),
    "F.3.S.5": (0, 1, 2),
    "F.3.S.6": (0, 1, 2),
    # Training: preserve the deployed model's 6-month/1-year/3-year encoding.
    # The default F.4 formula does not use these survey fields.
    "F.4.S.1": (0, 1, 2),
    "F.4.S.2": (0, 1, 2),
    "F.4.S.3": (0, 1, 2),
    "F.4.S.4": (1, 0, 2),
    "F.4.S.5": (0, 1, 2),
    "F.4.S.6": (0, 1, 2),
    "F.4.S.7": (0, 1, 2),
    # Communication.
    "F.5.S.1": (0, 1, 2),
    "F.5.S.2": (0, 1, 2),
    "F.5.S.3": (0, 1, 2),
    "F.5.S.4": (0, 1, 2),
    "F.5.S.5": (0, 1, 2),
    "F.5.S.6": (0, 1, 2),
    # Environment: raw inputs to the saved GMM, not its composite outputs.
    "F.6.S.1": (0, 1),
    "F.6.S.2": (0, 1),
    "F.6.S.3": (0, 1),
    "F.6.S.4": (0, 1),
    "F.6.S.5": (0, 1),
    "F.6.S.6": (0, 1),
    "F.6.S.7": (0, 1),  # .1=objects not visible, .2=breathing discomfort
    "F.6.S.8": (0, 1),
    "F.6.S.9": (1, 0),  # .1=friction <0.63, .2=friction >=0.63
    "F.6.S.10": (1, 0, 1),
    "F.6.S.11": (1, 0, 1),
    "F.6.S.12": (0, 1),
    "F.6.S.13": (0, 1, 2),
    "F.6.S.14": (2, 1, 0),
    # Work process.
    "F.7.S.1": (0, 1, 2),
    "F.7.S.2": (0, 1, 2),
    "F.7.S.3": (0, 1, 2),
    "F.7.S.4": (0, 1, 2),
    "F.7.S.5": (0, 1, 2),
    # Complexity.
    "F.8.S.1": (0, 1),
    "F.8.S.2": (0, 1, 2),
    "F.8.S.3": (0, 1, 2),
    "F.8.S.4": (0, 1, 2),
    "F.8.S.5": (0, 1, 2),
    "F.8.S.6": (0, 1, 2),
    "F.8.S.7": (0, 1),
}

CATBOOST_CODE_TO_VALUE = {}
for field, values in CATBOOST_OPTION_VALUES.items():
    for option_number, value in enumerate(values, start=1):
        code = f"{field}.{option_number}"
        CATBOOST_CODE_TO_VALUE[code] = value
        CATBOOST_CODE_TO_VALUE[code.replace(".S.", ".", 1)] = value

CATBOOST_ALLOWED_VALUES = {
    field: frozenset(values)
    for field, values in CATBOOST_OPTION_VALUES.items()
}


def canonical_condition_key(key):
    """Accept both F.1.S.3 and the legacy F.1.3 field key."""
    if isinstance(key, str):
        parts = key.split(".")
        if len(parts) == 3 and parts[0] == "F":
            return f"F.{parts[1]}.S.{parts[2]}"
    return key
