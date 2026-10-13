# Railway HEP — 철도 작업 인적 오류 예측 웹앱

철도 작업자의 설문 응답을 바탕으로 수행영향요인(PSF, Performance Shaping Factor)을 예측하고, 직종별 기본 계수와 PSF 배수를 사용해 인적 오류 확률(HEP, Human Error Probability)을 계산하는 Flask 웹 애플리케이션입니다.

기관사·관제사·정비사·선로작업자를 대상으로 설문, 결과 요약, 항목별 상세 분석, 시간별·월별 HEP 변화 그래프를 제공합니다. 현재 기본 예측 경로는 **CatBoost 분류 모델과 교육·훈련 계산식**을 사용합니다.

## 주요 기능

- 8개 PSF 설문 입력 및 수준 예측
- 직종별 진단 HEP·수행 HEP·최종 HEP 계산 및 게이지 표시
- PSF별 분류 확률과 교육·훈련 배수 표시
- 근무 시작 일시를 반영한 조회 시점부터 24시간의 HEP 변화 그래프
- 마지막 직무 교육 후 경과 개월 수를 반영한 향후 60개월의 HEP 변화 그래프
- 슬라이더를 통한 시점별 HEP와 PSF 값 확인
- 개별 그래프 팝업, 입력 검증, 로딩 및 오류 메시지

## 기술 구성과 처리 흐름

| 구분 | 기술 및 역할 |
| --- | --- |
| 서버 | Python, Flask, Flask-Cors |
| 기본 분류 | CatBoost |
| 전처리 | NumPy, pandas, scikit-learn 기반 GMM, joblib |
| 교육·훈련 예측 | Python 계산식 |
| 기존 CNN 기능 | TensorFlow / Keras |
| 화면 | HTML, CSS, JavaScript |
| 그래프 | 외부 차트 라이브러리 없이 SVG로 렌더링 |
| 테스트 | Node.js 내장 테스트 러너 |

```text
설문 입력 → Flask API → 문항 정규화 / 복합 특성 생성
                     → CatBoost 예측 + F.4 계산식
                     → 브라우저에 결과 저장
                     → 직종별 HEP 계산 → 결과 및 그래프 표시
```

PSF 예측과 시간·월별 변화 데이터는 서버에서 생성합니다. 최종 HEP 계산과 그래프 표시는 브라우저에서 수행하며 별도 프런트엔드 빌드 과정은 없습니다.

## 설치 및 실행

### 저장소 내려받기

```powershell
git clone https://github.com/JungWonSeok97/Railway_hep.git
cd Railway_hep
```

### 가상환경과 의존성 설치

Python과 pip가 설치된 환경에서 프로젝트 루트를 기준으로 실행합니다. Python 버전은 사용하는 TensorFlow와 CatBoost가 지원하는 버전에 맞춰 선택합니다. 현재 `requirements.txt`에는 패키지 버전이 고정되어 있지 않습니다.

Windows / PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

PowerShell에서 가상환경 활성화가 제한되면 가상환경의 Python을 직접 실행할 수 있습니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

### 접속

브라우저에서 **http://127.0.0.1:8000**에 접속합니다. HTML 파일을 직접 더블클릭하면 API와 정적 파일 경로가 연결되지 않으므로 Flask 주소를 사용합니다.

현재 실행 설정은 `host="0.0.0.0"`, `port=8000`, `debug=True`입니다. 개발용 설정이므로 운영 배포 시 디버그 모드와 서버 구성을 별도로 변경해야 합니다.

기본 설문 예측에는 저장소의 CatBoost·GMM 모델을 사용합니다. `app.py`가 기존 CNN 모듈도 가져오기 때문에 현재 구성에서는 TensorFlow 설치가 필요합니다.

## 사용 방법

1. 첫 화면에서 직종을 선택합니다.
2. 8개 PSF 설문에 응답하고 마지막 직무 교육 후 경과 개월 수를 입력합니다.
3. **분석하기**를 누르면 `/api/predict`로 요청하고 결과 요약 화면으로 이동합니다.
4. **상세보기**에서 항목별 수준, 분류 확률, 교육·훈련 배수와 HEP 결과를 확인합니다.
5. **시간별 HEP 예측** 또는 **월별 HEP 예측**으로 해당 그래프를 엽니다. 필요한 값이 저장되어 있으면 자동 조회하며, 값이 없으면 팝업에서 입력 후 조회합니다.
6. 그래프 아래 슬라이더로 특정 시점의 수치를 확인합니다. 닫기 버튼, Esc 키 또는 팝업 바깥 클릭으로 닫습니다.
7. **결과 요약**으로 돌아가거나 **다시 설문하기**로 새로운 분석을 시작합니다.

예측 결과는 `sessionStorage`의 `psfResults`, 직종은 `selectedRole`에 저장됩니다. 결과 화면은 같은 탭의 저장값을 사용하므로 새 탭에서 결과 URL만 열거나 탭을 닫은 뒤에는 설문부터 다시 진행해야 합니다. 서버 측 결과 저장 기능은 구현되어 있지 않습니다.

## 예측 항목

| ID | 항목 | 기본 `/api/predict` 처리 | 출력 수준 |
| --- | --- | --- | --- |
| F.1 | 작업부하 | CatBoost | 보통 / 나쁨 / 매우나쁨 |
| F.2 | 장비 및 기기 | GMM 복합 특성 + CatBoost | 좋음 / 보통 / 나쁨 |
| F.3 | 절차 및 지침 | CatBoost | 좋음 / 보통 / 나쁨 |
| F.4 | 교육 및 훈련 | 교육 후 경과 개월 수 계산식 | 좋음 / 보통 / 나쁨 |
| F.5 | 의사소통 | CatBoost | 보통 / 나쁨 |
| F.6 | 환경 | GMM 복합 특성 + CatBoost | 보통 / 나쁨 |
| F.7 | 작업절차 | CatBoost | 좋음 / 보통 / 나쁨 |
| F.8 | 복잡성 | GMM 복합 특성 + CatBoost | 보통 / 약간 높음 / 매우 높음 |

CatBoost 입력은 `config/catboost_config.py`의 특성 순서에 맞춰 생성하며, 로드한 모델의 특성 이름·순서와 일치하는지 검사합니다. 모델은 최초 사용 시 로드하고 메모리에 캐시합니다.

### 교육·훈련 배수

교육 후 경과 개월 수를 `x`라고 할 때 다음 식을 사용합니다.

```text
m4(x) = 0.33 + 4.5 × ((x / 18.555)^4 / (1 + (x / 18.555)^4))
```

- `x`는 0 이상의 유한한 숫자여야 하며 숫자 문자열과 불리언은 허용하지 않습니다.
- 응답 배수는 소수점 셋째 자리까지 반올림합니다.
- 반올림 전 배수가 1 미만이면 수준 0, 1 이상 2 미만이면 수준 1, 2 이상이면 수준 2입니다.
- `probabilities`는 `null`이며 HEP 계산에는 `hep_multiplier`를 사용합니다.
- 기존 교육·훈련 선택 문항은 화면에 유지되지만 기본 F.4 계산에는 경과 개월 수만 사용합니다.

### HEP 계산 방식

현재 코드의 직종별 기본 계수는 다음과 같습니다.

| 직종 | 진단 기본 계수 | 수행 기본 계수 |
| --- | ---: | ---: |
| 기관사 | 0.01225 | 0.00615 |
| 관제사 | 0.0055 | 0.00352 |
| 정비사 | 0.00279 | 0.00347 |
| 선로작업자 | 0.01225 | 0.0035 |

각 PSF의 수준을 배수로 변환하고 F.4에는 계산된 배수를 적용합니다. 전체 PSF 배수의 곱을 `M`, 진단·수행 기본 계수를 각각 `d`, `e`라고 하면 다음과 같이 계산합니다.

```text
진단 HEP = d × M
수행 HEP = e × M
최종 HEP = 진단 HEP + 수행 HEP

위 합계가 1을 초과하면 아래 보정값을 합산:
보정 진단 HEP = (d × M) / (d × (M - 1) + 1)
보정 수행 HEP = (e × M) / (e × (M - 1) + 1)
```

계산 함수는 `result.html`과 `detail_result.html`에 있습니다. 두 파일의 계수·배수 매핑은 함께 관리해야 합니다. 그래프는 계산 결과를 그대로 표시하며 100을 곱하지 않습니다. 현재 상세 화면의 일부 값에는 `%` 표기가 붙지만 계산 자체에는 백분율 변환이 없으므로 수치 해석 시 이 구현 차이를 확인해야 합니다. 보정 후 합계에 별도의 1 상한 처리는 없습니다.

## 예측 그래프

### 시간별 HEP

- 입력: 근무 시작 일시 `work_started_at`, 설문에서 받은 원본 F.1 수준 `base_psf1`
- 시간대: 한국시간(UTC+09:00); 시간대 없는 입력도 한국시간으로 해석
- F.1 보정: 원본 수준, 근무 경과 시간 수준, 시간대 수준 중 최댓값 사용
- 근무 경과 시간 경계: 1.3 / 5.6 / 10.6 / 12.8시간
- 시간대 경계: 06:24 / 18:24
- 표시 범위: **API 조회 시점부터 24시간**; 가로축 0은 근무 시작 시각이 아님
- F.1만 변화시키고 F.2–F.8은 현재 설문 결과로 고정
- 종료값이 `null`인 마지막 구간은 해당 수준이 계속 유지되는 것으로 처리

### 월별 HEP

- 입력: 현재 교육 후 경과 개월 수 `training_elapsed_months`
- 표시 범위: 현재부터 60개월 후까지, 0.1개월 간격의 총 601개 값
- 현재 경과 개월 수에 미래 개월 수를 더해 F.4 배수 계산
- F.4만 변화시키고 나머지 PSF는 현재 설문 결과로 고정
- 추가 교육에 따른 경과 개월 수 초기화는 자동 반영하지 않음

두 그래프는 서로 다른 요인을 변화시킵니다. 시간별·월별 변화를 동시에 결합한 시나리오는 현재 제공하지 않습니다.

## API

POST 요청은 `Content-Type: application/json`을 사용합니다.

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| GET | `/`, `/input`, `/input.html` | 설문 화면 |
| GET | `/result`, `/result.html` | 결과 요약 |
| GET | `/detail_result`, `/detail_result.html` | 상세 결과 및 그래프 |
| POST | `/api/predict` | 기본 예측: F.4 공식, 나머지 CatBoost |
| POST | `/api/psf1/predict` | 시간에 따른 F.1 수준·구간 반환 |
| POST | `/api/psf4/forecast` | 향후 60개월의 F.4 배수 반환 |
| POST | `/api/predict/<psf_id>` | 단일 PSF CatBoost 예측 |
| POST | `/api/catboost/predict`, `/api/predict/catboost` | CatBoost 일괄 예측; F.4도 CatBoost 사용 |
| POST | `/api/catboost/predict/<psf_id>`, `/api/predict/catboost/<psf_id>` | 단일 CatBoost 예측 별칭 |
| POST | `/api/test/predict/<psf_id>` | 한글 예측·확률과 실제 모델 입력 반환 |
| GET | `/api/model-status` | 기존 CNN 모델 파일·메모리 상태 조회 |
| POST | `/api/retrain` | 기존 CNN 모델 파일 삭제 후 재학습 |

`<psf_id>`는 정수 1–8입니다. CNN 상태 조회와 재학습은 기본 CatBoost 모델을 관리하는 API가 아닙니다.

### 기본 예측 요청 예시

아래는 F.1과 F.4만 요청하는 예시입니다. 전체 화면 분석에는 F.1–F.8을 모두 전달해야 합니다.

```json
{
  "F.1": {
    "F.1.S.1": 0,
    "F.1.S.2": 0,
    "F.1.S.3": 0,
    "F.1.S.4": 0,
    "F.1.S.5": 0
  },
  "F.4": {
    "training_elapsed_months": 12
  }
}
```

최상위 키는 `F.1`–`F.8`이며 기존 `psf1`–`psf8`도 지원합니다. CatBoost 문항 값은 0/1/2, 해당 숫자 문자열 또는 `F.1.S.1.1` 같은 등록된 선택 코드로 전달할 수 있습니다. 문항 키의 기존 형식인 `F.1.1`도 지원합니다. 선택 코드와 특성 정의는 `config/mappings.py`, `config/catboost_config.py`를 참고합니다.

API는 요청에 없는 PSF를 건너뛰지만 웹 화면은 결과 8개를 요구합니다. CatBoost 하위 문항이 누락되면 전처리에서 기본값 1을 사용하므로 실제 분석에는 모든 문항을 명시적으로 전달하는 것이 좋습니다.

성공 응답은 `success: true`와 `results` 배열로 구성됩니다. 각 결과에는 다음 필드가 있습니다.

| 필드 | 의미 |
| --- | --- |
| `psf_id` | `F.1` 같은 항목 ID |
| `predicted_label` | 0부터 시작하는 수준 번호 |
| `predicted_text` | `F.1.1` 같은 출력 코드 |
| `probabilities` | 출력 코드별 분류 확률; 기본 F.4는 `null` |
| `hep_multiplier` | 기본 F.4에서 반환하는 교육·훈련 배수 |

분류 확률은 PSF 수준에 대한 모델 확률이며 최종 HEP와는 별도 값입니다. 기본 API는 최종 HEP를 반환하지 않습니다.

### 월별 예측 호출 — PowerShell

```powershell
$body = @{ training_elapsed_months = 12 } | ConvertTo-Json
$response = Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/api/psf4/forecast' `
  -ContentType 'application/json' -Body $body
$response.series.hep_multipliers.Count  # 601
```

응답은 `success`, `psf_id`, `training_elapsed_months`, `series`를 포함합니다. `series`에는 `x_start: 0.0`, `x_end: 60.0`, `x_step: 0.1`, `hep_multipliers` 배열이 있습니다.

### 시간별 예측 호출 — PowerShell

```powershell
$body = @{
  work_started_at = (Get-Date).AddHours(-2).ToString('o')
  base_psf1 = 0
} | ConvertTo-Json
Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/api/psf1/predict' `
  -ContentType 'application/json' -Body $body
```

`base_psf1`은 정수 0/1/2만 허용합니다. 근무 시작 일시는 ISO 8601 형식이어야 하며 미래일 수 없습니다. 예시의 날짜는 실행 컴퓨터의 시각·시간대를 사용합니다.

응답은 `generated_at`, `work_started_at`, `elapsed_hours`, `base_prediction`, `current_prediction`, `segments`를 포함합니다. 각 구간에는 `start_elapsed_hours`, `end_elapsed_hours`, `start_at`, `end_at`, `predicted_label`이 있습니다.

### 오류 응답

```json
{
  "success": false,
  "error": "F.4.training_elapsed_months는 필수입니다."
}
```

기본 `/api/predict`의 `ValueError`와 시간·월별 API의 입력 검증 오류는 HTTP 400, 기타 실행 오류는 HTTP 500을 반환합니다. 단일 CatBoost·기존 보조 API는 입력 오류도 HTTP 500으로 반환할 수 있으므로 상태 코드와 `success`, `error`를 함께 확인합니다.

## 프로젝트 구조

```text
Railway_hep/
├── app.py                         # Flask 화면 및 API 라우트
├── requirements.txt               # Python 의존성
├── input.html                     # 직종 선택 및 설문
├── result.html                    # 결과 요약, HEP 계산
├── detail_result.html             # 상세 결과, HEP 계산, 그래프 팝업
├── config/
│   ├── paths.py                   # 모델 및 기존 설문 원본 경로
│   ├── mappings.py                # 선택 코드·수준 매핑
│   └── catboost_config.py         # 모델 파일·특성 정의
├── preprocessing/
│   └── catboost_features.py       # 문항 정규화 및 GMM 특성 생성
├── predictors/
│   ├── catboost.py                # CatBoost 로드 및 예측
│   ├── psf4_formula.py            # 교육·훈련 배수 및 월별 예측
│   ├── psf1_forecast.py           # 작업부하 시간별 예측
│   └── cnn.py                     # 기존 CNN 학습·예측
├── models/
│   ├── *_catboost.cbm             # PSF별 CatBoost 모델
│   └── gmm/*.joblib               # 복합 특성용 GMM 번들
├── static/                        # 실제 화면에서 사용하는 CSS·JS
│   ├── forecast.js               # 그래프 조회·계산·SVG 렌더링
│   └── forecast.css              # 그래프·팝업 스타일
├── static copy/                   # 원래 디자인 참고용 CSS
├── images/                        # 항목 이미지 및 로딩 GIF
├── processed_data/
│   └── subpsf_level_mapping.xlsx  # 수준 매핑 참고 자료
├── output/                        # 그래프 예시 및 생성 코드
├── tests/forecast.test.cjs        # 그래프·HEP 계산 테스트
└── env_preprocess_v1.ipynb         # 환경 전처리 노트북
```

`HEP 예측 그래프 API 연동 가이드.pdf`는 요청·응답 형식 참고 자료입니다. 현재 직종별 계수와 보정식은 HTML 코드의 값을 사용하므로 PDF 계산 예시와 구분해서 확인합니다. `saved_models/`는 기존 CNN 모델용 디렉터리이며 실행 시 생성됩니다.

## 모델 및 데이터 관리

- CatBoost 모델 8개와 GMM 번들 4개가 저장소에 포함됩니다. F.4 CatBoost 모델·GMM은 보조 CatBoost 경로용으로 남아 있으며 기본 F.4 공식 경로에서는 사용하지 않습니다.
- 개인별 설문·직무·건강 원본 CSV, PSF 데이터 테이블, 데이터 이력은 `.gitignore`로 제외합니다. 매핑 참고용 XLSX는 포함합니다.
- Python 캐시, 가상환경, `.env` 파일, `tmp/`는 버전 관리에서 제외합니다.
- 기존 CNN 학습에는 `config/paths.py`의 `4.20 설문조사 결과.txt`가 필요하며 이 원본은 저장소에 포함되지 않습니다.
- `/api/retrain`은 기존 `saved_models/psfN_model.h5`를 삭제한 뒤 CNN 학습을 수행합니다. 학습 원본이 준비된 환경에서만 사용합니다.
- 모델 교체 시 특성 이름·순서·클래스 이름과 GMM 번들 구성을 맞춰야 합니다. 캐시된 모델을 다시 로드하려면 서버를 재시작합니다.

## 검증 방법

Node.js가 설치되어 있다면 프로젝트 루트에서 기존 테스트를 실행합니다. 별도 npm 패키지 설치는 필요하지 않습니다.

```powershell
node --test tests/forecast.test.cjs
```

이 테스트는 월별 601개 값과 HEP 계산, 시간별 구간 처리, 다른 PSF 유지, 팝업 열기·닫기, 입력 누락·API 오류·재조회 등을 확인합니다. 실제 CatBoost 모델 정확도나 브라우저 화면 전체를 검증하는 테스트는 아닙니다.

실행 확인 순서:

1. 서버를 시작하고 `/`에서 CSS와 이미지가 로드되는지 확인합니다.
2. 전체 설문을 제출해 결과 8개와 HEP 요약을 확인합니다.
3. 상세 결과에서 교육·훈련 배수와 다른 PSF 확률을 확인합니다.
4. 시간별·월별 팝업을 각각 열어 입력, 조회, 슬라이더, 닫기를 확인합니다.

## 문제 해결과 현재 제약

| 증상 | 확인할 내용 |
| --- | --- |
| CSS·이미지가 보이지 않음 | HTML 직접 실행 대신 Flask 주소로 접속했는지 확인 |
| 결과 화면에 데이터가 없음 | 같은 탭에서 설문을 제출했는지 확인; 결과는 `sessionStorage` 사용 |
| F.4 필수값 오류 | `F.4.training_elapsed_months`를 숫자로 전달했는지 확인 |
| 시간별 조회 오류 | 시작 일시 형식·미래 시각 여부와 `base_psf1` 정수 0/1/2 확인 |
| 모델 누락·특성 불일치 | 모델 파일명과 설정의 특성 이름·순서 확인 |
| TensorFlow 설치 실패 | 선택한 Python·플랫폼과 설치하려는 TensorFlow 버전의 호환성 확인 |
| GMM 버전 경고 | 저장 당시와 실행 환경의 scikit-learn 버전을 비교하고 재검증 |
| CNN 재학습 실패 | 별도 설문 원본과 학습 환경 준비 여부 확인 |
| 8000 포트 사용 중 | 기존 서버 종료 또는 `app.py` 포트 설정 변경 |

기존 실행 기록에는 GMM 저장 당시 scikit-learn 1.9.0과 실행 환경 1.7.2의 버전 차이 경고가 보고되어 있습니다. 이는 현재 환경 버전을 보장하는 정보가 아니며, 재현성 확인 시 학습 환경과 실행 환경을 맞춰 검증해야 합니다.

현재 저장소는 의존성 버전 고정, 운영 배포 설정, 서버 측 결과 저장과 사용자 인증을 제공하지 않습니다. 그래프는 다른 PSF를 고정한 조건부 계산 결과입니다. 모델 정확도와 실제 활용 기준은 별도의 검증이 필요합니다.
