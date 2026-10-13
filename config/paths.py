import os


# 프로젝트 루트
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# 설문 원본 데이터
SURVEY_DATA_PATH = os.path.join(
    BASE_DIR,
    "4.20 설문조사 결과.txt"
)

# CatBoost 모델
CATBOOST_MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

GMM_MODEL_DIR = os.path.join(
    CATBOOST_MODEL_DIR,
    "gmm"
)

# 기존 CNN 모델
MODEL_DIR = os.path.join(
    BASE_DIR,
    "saved_models"
)

os.makedirs(MODEL_DIR, exist_ok=True)