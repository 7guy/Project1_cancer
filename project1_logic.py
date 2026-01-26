import json
import pandas as pd
import joblib
from openai import OpenAI

def gpt_extraction(history, user_input, client):
    system_prompt = """
    너는 암 예측 모델을 위한 데이터 추출기야. 사용자의 입력에서 정보를 추출해서 JSON으로 반환해.
    필드: Age(정수), Gender(0:남, 1:여), BMI(실수), Smoking(0:No, 1:Yes), Alcohol(0~5), Family_History(정수), PhysicalActivity(0~10)
    키와 몸무게를 말하면 BMI를 계산해. (몸무게kg / 키m^2)
    반드시 JSON 형식 {"Age": 50, ...}만 출력해. 추출할 수 없으면 null로 채워.
    """
    messages = [{"role": "system", "content": system_prompt}]
    # 최근 대화 문맥 5개까지만 전달하여 효율성 높임
    messages.extend(history[-5:])
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 속도를 위해 mini 권장
            messages=messages,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return None

def generate_ask_question(collected_data, missing_keys):
    name_map = {
        "Age": "나이", "Gender": "성별", "BMI": "BMI(또는 키와 몸무게)",
        "Smoking": "흡연 여부", "Alcohol": "음주 빈도",
        "Family_History": "가족력", "PhysicalActivity": "운동량"
    }
    questions = []
    for key in missing_keys:
        if key == "BMI":
            questions.append("BMI 또는 키(cm)와 몸무게(kg)를 알려주세요.")
        else:
            questions.append(f"{name_map[key]}를 알려주세요.")
            
    return "암 위험도 분석을 위해 아래 정보가 더 필요해요. 😊\n\n" + "\n".join(
        f"- {q}" for q in questions
    )

def interpret_result_with_gpt(prob, client, cancer_type="암"):
    percent = round(prob * 100, 1)
    if percent < 20:
        risk_level, tone = "낮은", "안심시키는 톤으로 설명"
    elif percent < 35:
        risk_level, tone = "중간", "주의를 주되, 과도한 불안은 주지 말 것"
    else:
        risk_level, tone = "높은", "조심스럽게 위험 신호를 전달하되 공포 조성 금지"

    system_prompt = f"""너는 의료 AI 상담 보조야. {cancer_type} 위험도 결과를 설명해줘.
    위험도 수준: {risk_level}, 말투: {tone}. 확률 기반 예측이며 진단이 아님을 명시할 것."""
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": f"예측된 {cancer_type} 확률은 {percent}%입니다."}]
    )
    return response.choices[0].message.content

# 모델 호출 및 확률 계산
import streamlit as st

# 1. 암종별 상세 설정
CANCER_CONFIG = {
    "TOTAL": {
        "model_path": "sub1_cancer_model.pkl",
        "features": ["Age", "Gender", "BMI", "Smoking", "Alcohol", "Family_History", "PhysicalActivity"],
        "type": "BINARY" # 0 또는 1 (확률)
    },
    "LUNG": {
        "model_path": "lung_xgb_model.pkl",
        # Gender -> 남성 : 0, 여성 : 1
        "features": ["Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy", "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease", "Balanced Diet", "Obesity", "Smoking", "Passive Smoker", "Chest Pain", "Weight Loss", "shortness of Breath", "Dry Cough"],
        "type": "MULTI" # 폐암은 라벨이 0, 1, 2 단계별
    },
    "LIVER": {
        "model_path": "liver_cancer_xgb_model.pkl",
        "scaler_path": "liver_cancer_scaler", # 스케일러 파일 추가
        "features": ["age", "gender", "bmi", "alcohol_consumption", "smoking_status", "hepatitis_b", "hepatitis_c", "cirrhosis_history", "family_history_cancer", "physical_activity_level", "diabetes"],
        "type": "BINARY"
    }
}

# 2. 모델 로드 (캐싱 적용)
@st.cache_resource
def get_model(cancer_type):
    return joblib.load(CANCER_CONFIG[cancer_type]["model_path"])

# 2. 스케일러 로드 함수
@st.cache_resource
def get_scaler(cancer_type):
    path = CANCER_CONFIG[cancer_type].get("scaler_path")
    return joblib.load(path) if path else None

# 3. [핵심] 통합 모델 호출 함수
def predict_cancer_risk(cancer_type, collected_data):
    config = CANCER_CONFIG.get(cancer_type)
    model = get_model(cancer_type)
    
    # 1. 데이터 추출 및 데이터프레임 생성
    input_data = [collected_data.get(f) for f in config["features"]]
    df = pd.DataFrame([input_data], columns=config["features"])
    
    # 2. 암종별 타입에 따른 결과 계산
    if config["type"] == "BINARY":
        # 간암 등: 단순 확률 (0~100)
        prob = model.predict_proba(df)[0][1]
        score = round(prob * 100, 1)

    elif cancer_type == "LIVER":
        scaler = get_scaler("LIVER")
        if scaler:
            df = scaler.transform(df) # 데이터를 모델에 맞는 스케일로 변환
        
    elif config["type"] == "MULTI":
        # 폐암: 팀원분의 가중치 계산식 적용
        # probabilities = [Low_prob, Medium_prob, High_prob]
        probabilities = model.predict_proba(df)[0]
        
        # 팀원분의 식: (Low*0.2 + Med*0.5 + High*0.9) * 100
        score = (
            probabilities[0] * 0.2 +
            probabilities[1] * 0.5 +
            probabilities[2] * 0.9
        ) * 100
        score = round(score, 1)

    # 3. 결과 반환
    return score