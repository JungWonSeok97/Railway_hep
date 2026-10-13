import os

from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
)
from flask_cors import CORS

from config.paths import (
    MODEL_DIR,
    SURVEY_DATA_PATH,
)

from predictors.cnn import (
    models as cnn_models,
    predict_psf,
    train_model_for_psf,
)

from predictors.catboost import (
    predict_psf_catboost,
    predict_psf_catboost_test,
)

from predictors.psf1_forecast import (
    build_psf1_forecast,
)

from predictors.psf4_formula import (
    build_psf4_forecast,
    predict_psf4_formula,
)


# =====================================================
# Flask
# =====================================================

app = Flask(__name__)
app.json.sort_keys = False
CORS(app)


# =====================================================
# HTML / 이미지
# =====================================================

@app.route("/input.html")
@app.route("/input")
@app.route("/")
def index():

    return send_from_directory(
        app.root_path,
        "input.html",
    )


@app.route("/detail_result.html")
@app.route("/detail_result")
def detail_result():

    return send_from_directory(
        app.root_path,
        "detail_result.html",
    )


@app.route("/result.html")
@app.route("/result")
def result():

    return send_from_directory(
        app.root_path,
        "result.html",
    )


@app.route(
    "/images/<path:filename>"
)
def serve_images(filename):

    return send_from_directory(
        "images",
        filename,
    )


# =====================================================
# 기존 CNN API
# =====================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if not isinstance(data, dict):
            raise ValueError(
                "요청 본문은 JSON 객체여야 합니다."
            )
        
        results = []
        
        for psf_id in range(1, 9):
            request_key = f"F.{psf_id}"
            legacy_key = f"psf{psf_id}"
            psf_input = data.get(request_key)

            if psf_input is None:
                psf_input = data.get(legacy_key)

            if psf_input is None:
                continue

            if psf_id == 4:
                result = predict_psf4_formula(
                    psf_input
                )
            elif psf_input:
                # 모델 교체 위치
                # result = predict_psf(
                #     psf_id,
                #     psf_input,
                # )
                result = predict_psf_catboost(
                    psf_id,
                    psf_input,
                )
            else:
                continue

            results.append(result)
        
        return jsonify({
            "success": True,
            "results": results
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =====================================================
# PSF1 시간 변화 예측 API
# =====================================================

@app.route(
    "/api/psf1/predict",
    methods=["POST"],
)
def predict_psf1_over_time():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            raise ValueError(
                "요청 본문은 JSON 객체여야 합니다."
            )

        result = build_psf1_forecast(
            data.get("work_started_at"),
            data.get("base_psf1"),
        )

        return jsonify({
            "success": True,
            **result,
        })

    except (TypeError, ValueError) as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =====================================================
# PSF4 미래 HEP 배수 API
# =====================================================

@app.route(
    "/api/psf4/forecast",
    methods=["POST"],
)
def forecast_psf4():
    try:
        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            raise ValueError(
                "요청 본문은 JSON 객체여야 합니다."
            )

        result = build_psf4_forecast(
            data.get("training_elapsed_months")
        )

        return jsonify({
            "success": True,
            **result,
        })

    except (TypeError, ValueError) as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route(
    "/api/predict/<int:psf_id>",
    methods=["POST"],
)
def predict_single(psf_id):
    """단일 PSF CatBoost 예측"""

    try:
        conditions = request.get_json()

        result = predict_psf_catboost(
            psf_id,
            conditions,
        )

        return jsonify({
            "success": True,
            "result": result,
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =====================================================
# CatBoost API
# =====================================================

@app.route(
    "/api/catboost/predict",
    methods=["POST"],
)
@app.route(
    "/api/predict/catboost",
    methods=["POST"],
)
def predict_catboost():

    try:
        data = request.get_json()

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "요청 본문은 "
                "JSON 객체여야 합니다."
            )

        results = []

        for psf_id in range(
            1,
            9,
        ):

            request_key = (
                f"F.{psf_id}"
            )

            legacy_key = (
                f"psf{psf_id}"
            )

            psf_input = data.get(
                request_key
            )

            if psf_input is None:
                psf_input = data.get(
                    legacy_key
                )

            if psf_input:

                result = (
                    predict_psf_catboost(
                        psf_id,
                        psf_input,
                    )
                )

                results.append(
                    result
                )

        return jsonify({
            "success": True,
            "results": results,
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


@app.route(
    "/api/catboost/predict/<int:psf_id>",
    methods=["POST"],
)
@app.route(
    "/api/predict/catboost/<int:psf_id>",
    methods=["POST"],
)
def predict_catboost_single(
    psf_id,
):

    try:
        conditions = (
            request.get_json()
        )

        result = (
            predict_psf_catboost(
                psf_id,
                conditions,
            )
        )

        return jsonify({
            "success": True,
            "result": result,
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =====================================================
# 기존 CNN 재학습
# =====================================================

@app.route(
    "/api/retrain",
    methods=["POST"],
)
def retrain_models():

    try:
        # 메모리 모델 초기화
        cnn_models.clear()

        # 기존 파일 삭제
        for psf_id in range(
            1,
            9,
        ):

            model_path = os.path.join(
                MODEL_DIR,
                f"psf{psf_id}_model.h5",
            )

            if os.path.exists(
                model_path
            ):
                os.remove(
                    model_path
                )

                print(
                    f"PSF{psf_id} "
                    f"기존 모델 삭제됨"
                )

        results = []

        # 모든 PSF 재학습
        for psf_id in range(
            1,
            9,
        ):

            model = (
                train_model_for_psf(
                    psf_id
                )
            )

            cnn_models[
                psf_id
            ] = model

            results.append(
                f"PSF{psf_id} 재학습 완료"
            )

        return jsonify({
            "success": True,
            "message": (
                "모든 모델이 "
                "재학습되었습니다."
            ),
            "results": results,
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =====================================================
# CNN 모델 상태
# =====================================================

@app.route(
    "/api/model-status",
    methods=["GET"],
)
def model_status():

    status = {}

    for psf_id in range(
        1,
        9,
    ):

        model_path = os.path.join(
            MODEL_DIR,
            f"psf{psf_id}_model.h5",
        )

        status[
            f"PSF{psf_id}"
        ] = {
            "saved": os.path.exists(
                model_path
            ),

            "loaded": (
                psf_id
                in cnn_models
            ),

            "current": (
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
            ),
        }

    return jsonify({
        "success": True,
        "status": status,
    })

@app.route(
    "/api/test/predict/<int:psf_id>",
    methods=["POST"],
)
def test_predict_single(psf_id):
    """
    모델 검증용 API

    기존 API와 동일한 입력을 받고,
    사람이 확인하기 쉬운 형태로 반환한다.
    """

    try:
        conditions = request.get_json()

        if not isinstance(
            conditions,
            dict,
        ):
            raise ValueError(
                "요청 본문은 JSON 객체여야 합니다."
            )

        result = (
            predict_psf_catboost_test(
                psf_id,
                conditions,
            )
        )

        return jsonify({
            "success": True,
            "result": result,
        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =====================================================
# 실행
# =====================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        debug=True,
        port=8000,
    )
