import json
import pandas as pd
import joblib
import os
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 설정 (Configuration)
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

CONTEXT_MAP = {"일반": "TOTAL", "폐": "LUNG", "간": "LIVER"}

# 데이터가 비었을 때 채워넣을 기본값 (시뮬레이션용)
DEFAULTS = {
    "Age": 40, "Gender": 0, "BMI": 24.0, "Smoking": 0, "Alcohol": 0,
    "Family_History": 0, "PhysicalActivity": 5,
    "Air Pollution": 3, "Alcohol use": 3, "Dust Allergy": 3, 
    "OccuPational Hazards": 3, "Genetic Risk": 3, "chronic Lung Disease": 3, 
    "Balanced Diet": 3, "Obesity": 3, "Passive Smoker": 3, "Chest Pain": 3, 
    "Weight Loss": 3, "shortness of Breath": 3, "Dry Cough": 3,
    "hepatitis_b": 0, "hepatitis_c": 0, "cirrhosis_history": 0, "diabetes": 0
}

# ==========================================
# 2. 질문 생성 및 데이터 추출 (요청하신 부분 반영)
# ==========================================

def generate_ask_question(collected_data, missing_keys):
    """ [요청하신 코드 원본 그대로 적용] """
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

def gpt_extraction(history, user_input, client):
    """ 사용자 입력에서 7대 필수 정보 및 암종별 추가 정보 추출 """
    system_prompt = """
    너는 의료 데이터 추출기야. JSON만 반환해.
    
    [필수 항목 7가지]
    Age(정수), Gender(0:남, 1:여), BMI(실수, 키/몸무게 입력시 계산), 
    Smoking(0:비흡연, 1:흡연), Alcohol(0:안마심 ~ 5:매일), 
    Family_History(0:없음, 1:있음), PhysicalActivity(0:안함 ~ 10:매일)
    
    [추가 항목]
    폐암: Cough, Chest Pain 등
    간암: Hepatitis_B, Hepatitis_C, Cirrhosis (유무 1/0)
    
    추출할 수 없으면 null로 채워.
    """
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-5:])
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

def normalize_keys(extracted: dict) -> dict:
    """ GPT 추출 키 -> 시스템 표준 키 매핑 """
    key_map = {
        "age": "Age", "나이": "Age", "gender": "Gender", "성별": "Gender",
        "bmi": "BMI", "비만도": "BMI", "smoking": "Smoking", "흡연": "Smoking",
        "alcohol": "Alcohol", "음주": "Alcohol", "family_history": "Family_History", "가족력": "Family_History",
        "physicalactivity": "PhysicalActivity", "운동": "PhysicalActivity", "exercise": "PhysicalActivity",
        # (기타 폐/간암 키 매핑은 생략했으나 필요시 이전 코드처럼 추가 가능)
    }
    # 키를 소문자로 바꿔서 매핑 확인 후, 원래 값 반환
    return {key_map.get(k.lower(), k): v for k, v in extracted.items()}

def detect_simulation_intent(user_input, client):
    """ [강력해진 버전] 사용자의 가정/예측 질문을 귀신같이 잡아냄 """
    system_prompt = """
    너는 헬스케어 챗봇의 의도 분석기야. 
    사용자의 질문이 현재 상태 입력인지, 아니면 "미래/가정(Simulation)"인지 판단해 JSON을 반환해.
    
    [Simulation 판단 기준]
    1. "~하면 어때?", "~하면 어떻게 돼?", "확률은?" 같이 결과를 묻는 경우
    2. "만약", "가정", "미래", "나중에" 같은 단어
    3. 이미 정보를 입력했는데 나이/습관을 바꿔서 물어보는 경우
    
    [규칙]
    - 키는 표준 영문 키 사용 (Age, Smoking, Alcohol, PhysicalActivity)
    - type은 "simulation" 또는 "none"
    
    [예시]
    "담배 끊으면 어떻게 될까?" -> {"type": "simulation", "changes": {"Smoking": 0}}
    "50살에 술 마시면?" -> {"type": "simulation", "changes": {"Age": 50, "Alcohol": 1}}
    "지금 24살이야" -> {"type": "none"} (단순 정보 제공)
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

def interpret_result_with_gpt(prob, client, cancer_type="암"):
    # [수정] 0.01%라도 나오면 살려내기
    # prob가 0~100 사이 값으로 들어온다고 가정
    percent = prob 
    
    # 1. 수치가 너무 작으면 최소값 보정 (0.0% 방지)
    if 0 < percent < 0.1:
        display_percent = "< 0.1"
    else:
        display_percent = f"{round(percent, 1)}"
    
    # 2. 위험도 레벨 분류
    if percent < 20: 
        risk_level, tone = "낮음 (안심)", "차분하고 친절한 톤"
    elif percent < 35: 
        risk_level, tone = "중간 (주의)", "진지하고 조언하는 톤"
    else: 
        risk_level, tone = "높음 (경고)", "강력하게 경고하는 톤"

    # 3. GPT에게 상황 설명 (프롬프트 강화)
    # 단순히 숫자만 주는 게 아니라, '술 담배를 하는데도 낮게 나왔다'는 맥락을 줌
    system_prompt = f"""
    너는 헬스케어 전문가야. 
    사용자의 {cancer_type} 위험도는 {display_percent}%야. (수준: {risk_level})
    
    [중요 가이드]
    1. 수치가 낮더라도, 사용자가 '술/담배' 등 나쁜 습관이 있다면 "지금은 젊어서 괜찮지만 나중에 위험해질 수 있다"고 따끔하게 조언해줘.
    2. 수치가 0%에 가까우면 "현재로선 매우 건강하지만 방심하지 마세요"라고 해줘.
    3. 절대 "0%니까 막 살아도 된다"는 식의 뉘앙스는 풍기지 마.
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}, 
                  {"role": "user", "content": f"내 위험도는 {display_percent}%입니다."}],
    )
    return response.choices[0].message.content, percent

# ==========================================
# 3. 모델 예측 및 시뮬레이션
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
    
    # 모델별 특수 매핑 (간암 등)
    if config_key == "LUNG": mapped_data["Alcohol use"] = collected_data.get("Alcohol")
    elif config_key == "LIVER":
        mapped_data["age"] = collected_data.get("Age")
        mapped_data["gender"] = collected_data.get("Gender")
        mapped_data["bmi"] = collected_data.get("BMI")
        mapped_data["alcohol_consumption"] = collected_data.get("Alcohol")
        mapped_data["smoking_status"] = collected_data.get("Smoking")
        mapped_data["physical_activity_level"] = collected_data.get("PhysicalActivity")
        mapped_data["family_history_cancer"] = collected_data.get("Family_History")

    input_values = []
    for f in config["features"]:
        val = mapped_data.get(f)
        if val is None: val = DEFAULTS.get(f, 0) # 기본값 채우기
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