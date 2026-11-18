import os


# 프로젝트 루트
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
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
