from flask import (
    Flask,
    request,
    jsonify,
    send_from_directory,
)
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

from predictors.catboost import (
    predict_psf_catboost,
    predict_psf_catboost_test,
)

from predictors.psf_multipliers import summarize_psf_results

from predictors.psf1_forecast import (
    build_psf1_forecast,
)

from predictors.psf4_formula import (
    build_psf4_forecast,
    normalize_psf4_formula_input,
    predict_psf4_formula,
)
from preprocessing.catboost_features import normalize_catboost_conditions


# =====================================================
# Flask
# =====================================================

app = Flask(__name__)
app.json.sort_keys = False
CORS(app)


def read_json_object(allowed_keys=None):
    """Preserve malformed JSON (400) and unsupported media type (415)."""
    data = request.get_json()
    if not isinstance(data, dict):
        raise ValueError("요청 본문은 JSON 객체여야 합니다.")
    if allowed_keys is not None:
        unknown = [key for key in data if key not in allowed_keys]
        if unknown:
            raise ValueError("지원하지 않는 요청 키입니다: " + ", ".join(unknown))
    return data


def normalize_batch_inputs(data, *, formula_psf4=False):
    """Validate every supplied PSF before running any prediction.

    An absent PSF is allowed. A supplied empty/null/incomplete PSF is not.
    Compare aliases before GMM compression, which can erase differences.
    """
    allowed = {key for psf in range(1, 9) for key in (f"F.{psf}", f"psf{psf}")}
    unknown = [key for key in data if key not in allowed]
    if unknown:
        raise ValueError("지원하지 않는 PSF 키입니다: " + ", ".join(unknown))
    resolved = []
    for psf_id in range(1, 9):
        aliases = [key for key in (f"F.{psf_id}", f"psf{psf_id}") if key in data]
        if not aliases:
            continue
        values = [
            normalize_psf4_formula_input(data[key])
            if formula_psf4 and psf_id == 4
            else normalize_catboost_conditions(psf_id, data[key])
            for key in aliases
        ]
        if any(value != values[0] for value in values[1:]):
            raise ValueError(f"F.{psf_id}와 psf{psf_id}의 값이 서로 다릅니다.")
        resolved.append((psf_id, values[0]))
    return resolved


def http_error_response(error):
    return jsonify({"success": False, "error": error.description}), error.code


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
# 기본 PSF 예측·배수 API
# =====================================================

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = read_json_object()
        
        results = []
        
        for psf_id, psf_input in normalize_batch_inputs(data, formula_psf4=True):
            if psf_id == 4:
                result = predict_psf4_formula(
                    psf_input
                )
            else:
                result = predict_psf_catboost(
                    psf_id,
                    psf_input,
                )
            results.append(result)
        
        return jsonify({
            "success": True,
            **summarize_psf_results(results),
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

    except HTTPException as e:
        return http_error_response(e)

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
        data = read_json_object({"work_started_at", "base_psf1"})

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

    except HTTPException as e:
        return http_error_response(e)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


# =====================================================
# PSF4 미래 배수 API
# =====================================================

@app.route(
    "/api/psf4/forecast",
    methods=["POST"],
)
def forecast_psf4():
    try:
        data = read_json_object({"training_elapsed_months"})

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

    except HTTPException as e:
        return http_error_response(e)

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
        conditions = read_json_object()

        result = predict_psf_catboost(
            psf_id,
            conditions,
        )

        return jsonify({
            "success": True,
            "result": result,
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400

    except HTTPException as e:
        return http_error_response(e)

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
        data = read_json_object()

        results = []

        for psf_id, psf_input in normalize_batch_inputs(data):
            results.append(predict_psf_catboost(psf_id, psf_input))

        return jsonify({
            "success": True,
            **summarize_psf_results(results),
        })

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400

    except HTTPException as e:
        return http_error_response(e)

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
        conditions = read_json_object()

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

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400

    except HTTPException as e:
        return http_error_response(e)

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e),
        }), 500


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
        conditions = read_json_object()

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

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": str(e),
        }), 400

    except HTTPException as e:
        return http_error_response(e)

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
