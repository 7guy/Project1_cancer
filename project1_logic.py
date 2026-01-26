import json
import pandas as pd
import joblib
import os
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 설정 및 기본값
# ==========================================

CANCER_CONFIG = {
    "TOTAL": {
        "model_path": "sub1_cancer_model.pkl",
        "features": ["Age", "Gender", "BMI", "Smoking", "Alcohol", "Family_History", "PhysicalActivity"],
        "type": "BINARY" 
    },
    "LUNG": {
        "model_path": "lung_xgb_model.pkl",
        "features": ["Age", "Gender", "Air Pollution", "Alcohol use", "Dust Allergy", "OccuPational Hazards", "Genetic Risk", "chronic Lung Disease", "Balanced Diet", "Obesity", "Smoking", "Passive Smoker", "Chest Pain", "Weight Loss", "shortness of Breath", "Dry Cough"],
        "type": "MULTI" 
    },
    "LIVER": {
        "model_path": "liver_cancer_xgb_model.pkl",
        "scaler_path": "liver_cancer_scaler.pkl", 
        "features": ["age", "gender", "bmi", "alcohol_consumption", "smoking_status", "hepatitis_b", "hepatitis_c", "cirrhosis_history", "family_history_cancer", "physical_activity_level", "diabetes"],
        "type": "BINARY"
    }
}

CONTEXT_MAP = {"암": "TOTAL", "폐": "LUNG", "간": "LIVER"}

DEFAULTS = {
    "Age": 40, "Gender": 0, "BMI": 24.0, "Smoking": 0, "Alcohol": 0,
    "Family_History": 0, "PhysicalActivity": 5,
    "Air Pollution": 3, "Alcohol use": 3, "Dust Allergy": 3, 
    "OccuPational Hazards": 3, "Genetic Risk": 3, "chronic Lung Disease": 3, 
    "Balanced Diet": 3, "Obesity": 3, "Passive Smoker": 3, "Chest Pain": 3, 
    "Weight Loss": 3, "shortness of Breath": 3, "Dry Cough": 3
}

# ==========================================
# 2. 똑똑한 AI 판단 함수들 (Router & Extractor)
# ==========================================

def determine_automatic_context(user_input, current_context):
    """
    [신규] 사용자의 말을 듣고 '일반/폐/간' 모드를 자동으로 결정하는 라우터
    예: "폐암은 어때?" -> "폐" 반환
    예: "내 나이가 24살이야" -> 모드 변경 없음 (current_context 반환)
    """
    if any(word in user_input for word in ["폐암", "기침", "가슴", "폐가"]):
        return "폐"
    elif any(word in user_input for word in ["간암", "술", "간염", "황달", "피로"]):
        return "간"
    elif any(word in user_input for word in ["일반", "종합", "전체"]):
        return "일반"
    else:
        return current_context # 변경 사항 없으면 유지

def gpt_extraction(history, user_input, client):
    """ 데이터 추출 (기존 유지) """
    system_prompt = """
    너는 암 예측 모델을 위한 데이터 추출기야. JSON만 반환해.
    필드: Age, Gender(0:남, 1:여), BMI, Smoking(0:No, 1:Yes), Alcohol(0~5), Family_History(0,1), PhysicalActivity(0~10)
    폐암: Air Pollution, Dust Allergy, Occupational Hazards, chronic Lung Disease 등 (유무 1/0)
    간암: Hepatitis B, Hepatitis C, Cirrhosis, Diabetes
    키/몸무게 입력시 BMI 계산.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-5:]) # 과거 대화 기억!
    messages.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return None

def detect_simulation_intent(user_input, client):
    """ 시뮬레이션 의도 파악 (Exercise -> PhysicalActivity 자동 변환 추가) """
    system_prompt = """
    사용자의 질문이 "가정(if)", "미래", "조건 변경"이면 JSON 반환.
    
    [규칙] 표준 키 사용:
    - 운동, Exercise -> PhysicalActivity
    - 나이 -> Age
    - 담배 -> Smoking
    - 술 -> Alcohol
    
    예: "운동 안 하면?" -> {"type": "simulation", "changes": {"PhysicalActivity": 0}}
    아니면 -> {"type": "none"}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_input}],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"type": "none"}

def normalize_keys(extracted: dict) -> dict:
    """ 키 매핑 (Exercise -> PhysicalActivity 등) """
    key_map = {
        "age": "Age", "나이": "Age", "gender": "Gender", "성별": "Gender",
        "bmi": "BMI", "비만도": "BMI", "smoking": "Smoking", "흡연": "Smoking",
        "alcohol": "Alcohol", "음주": "Alcohol", "family_history": "Family_History", "가족력": "Family_History",
        "physicalactivity": "PhysicalActivity", "운동": "PhysicalActivity", "exercise": "PhysicalActivity",
        "physical_activity": "PhysicalActivity",
        "air pollution": "Air Pollution", "alcohol use": "Alcohol use",
        "dust allergy": "Dust Allergy", "occupational hazards": "OccuPational Hazards",
        "genetic risk": "Genetic Risk", "chronic lung disease": "chronic Lung Disease",
        "balanced diet": "Balanced Diet", "obesity": "Obesity", "passive smoker": "Passive Smoker", 
        "chest pain": "Chest Pain", "weight loss": "Weight Loss", 
        "shortness of breath": "shortness of Breath", "dry cough": "Dry Cough",
        "hepatitis_b": "hepatitis_b", "hepatitis_c": "hepatitis_c",
        "cirrhosis": "cirrhosis_history", "diabetes": "diabetes",
        "alcohol_consumption": "alcohol_consumption", "smoking_status": "smoking_status",
        "physical_activity_level": "physical_activity_level", "family_history_cancer": "family_history_cancer"
    }
    return {key_map.get(k.lower(), k): v for k, v in extracted.items()}

def interpret_result_with_gpt(prob, client, cancer_type="암"):
    percent = round(prob, 1)
    if percent < 20: risk_level, tone = "낮은", "안심시키는 톤"
    elif percent < 35: risk_level, tone = "중간", "주의를 주는 톤"
    else: risk_level, tone = "높은", "경고하는 톤"

    system_prompt = f"의료 AI로서 {cancer_type} 위험도({percent}%)를 설명해줘. 위험도: {risk_level}, 말투: {tone}."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"{percent}%"}],
    )
    return response.choices[0].message.content, percent

def generate_ask_question(collected_data, missing_keys):
    name_map = {"Age": "나이", "Gender": "성별", "BMI": "BMI", "Smoking": "흡연", "Alcohol": "음주"}
    questions = [name_map.get(k, k) + "를 알려주세요." for k in missing_keys if k in name_map]
    return "다음 정보가 필요해요:\n" + "\n".join(f"- {q}" for q in questions)

# ==========================================
# 3. 예측 및 시뮬레이션 (평균값 채우기 적용)
# ==========================================

@st.cache_resource
def get_model(config_key):
    try: return joblib.load(CANCER_CONFIG[config_key]["model_path"])
    except: return None

@st.cache_resource
def get_scaler(config_key):
    try:
        path = CANCER_CONFIG[config_key].get("scaler_path")
        return joblib.load(path) if path else None
    except: return None

def predict_cancer_risk(context_korean, collected_data):
    config_key = CONTEXT_MAP.get(context_korean, "TOTAL")
    config = CANCER_CONFIG[config_key]
    model = get_model(config_key)
    if model is None: return 0.0
    
    mapped_data = collected_data.copy()
    
    # 모델별 특수 매핑
    if config_key == "LUNG": mapped_data["Alcohol use"] = collected_data.get("Alcohol")
    elif config_key == "LIVER":
        mapped_data["age"] = collected_data.get("Age")
        mapped_data["gender"] = collected_data.get("Gender")
        mapped_data["bmi"] = collected_data.get("BMI")
        mapped_data["alcohol_consumption"] = collected_data.get("Alcohol")
        mapped_data["smoking_status"] = collected_data.get("Smoking")
        mapped_data["physical_activity_level"] = collected_data.get("PhysicalActivity")
        mapped_data["family_history_cancer"] = collected_data.get("Family_History")
        mapped_data["diabetes"] = collected_data.get("diabetes")

    input_values = []
    for f in config["features"]:
        val = mapped_data.get(f)
        if val is None: val = DEFAULTS.get(f, 0) # 값이 없으면 평균값 사용
        input_values.append(val)

    df = pd.DataFrame([input_values], columns=config["features"])

    if config["type"] == "BINARY":
        if config_key == "LIVER":
            scaler = get_scaler("LIVER")
            if scaler: df = pd.DataFrame(scaler.transform(df), columns=config["features"])
        prob = model.predict_proba(df)[0][1]
        score = round(prob * 100, 1)
    elif config["type"] == "MULTI":
        probabilities = model.predict_proba(df)[0]
        score = (probabilities[0] * 0.2 + probabilities[1] * 0.5 + probabilities[2] * 0.9) * 100
        score = round(score, 1)

    return score

def simulate_lifestyle_change(context_korean, current_data, changes_dict, current_score):
    normalized_changes = normalize_keys(changes_dict)
    sim_data = current_data.copy()
    for key, value in normalized_changes.items():
        sim_data[key] = value
    new_score = predict_cancer_risk(context_korean, sim_data)
    diff = new_score - current_score
    return current_score, new_score, diff