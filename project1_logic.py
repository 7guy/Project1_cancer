import json
import pandas as pd
import joblib
from openai import OpenAI

# 모델 로드 (경로 유지)
MODEL_PATH = r"sub1_cancer_model.pkl"
MODEL_PATH_LIVER = r"liver_cancer_xgb_model.pkl"
MODEL_PATH_LUNG = r"lung_xgb_model.pkl"

MODELS = {
    "간암": joblib.load(MODEL_PATH_LIVER),
    "폐암": joblib.load(MODEL_PATH_LUNG)
}

model = joblib.load(MODEL_PATH)
liver_model = joblib.load(MODEL_PATH_LIVER)
lung_model = joblib.load(MODEL_PATH_LUNG)

## 모델별 피쳐
FEATURE_CONFIG = {
    "간암" : ['age', 'gender', 'bmi', 'alcohol_consumption', 'smoking_status',
        'hepatitis_b', 'hepatitis_c', 'cirrhosis_history',
        'family_history_cancer', 'physical_activity_level', 'diabetes'],

    "폐암": ["Age", "Gender", "Alcohol use", "Dust Allergy", "OccuPational Hazards",
            "Genetic Risk", "chronic Lung Disease", "Balanced Diet", "Obesity",
            "Smoking", "Passive Smoker", "Dry Cough"]
}

# 추가된 시나리오 분석 함수
def handle_scenario_analysis(user_input, current_data, cancer_type, client):
    """기존 데이터를 복사하여 가상의 시나리오 수치로 비교 예측 수행"""
    # 1. 시뮬레이션 데이터 준비 (원본 보존)
    scenario_data = current_data.copy()
    features = FEATURE_CONFIG[cancer_type]
    
    # 2. GPT를 통해 어떤 수치를 변경할지 추출
    sim_prompt = f"""
    사용자의 가상 질문에서 변경하고 싶어하는 수치를 추출해 JSON으로 반환해.
    가능한 변수명: {features}
    현재 데이터: {current_data}
    반드시 JSON {{"변수명": 값}} 형식만 출력해.
    예: "담배 끊으면?" -> {{"Smoking": 0}} 또는 {{"smoking_status": 0}}
    """
    sim_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": sim_prompt}],
        response_format={"type": "json_object"}
    )
    changes = json.loads(sim_response.choices[0].message.content)
    scenario_data.update(changes)

    # 3. 비교 예측 수행
    orig_df = pd.DataFrame([current_data])[features]
    scen_df = pd.DataFrame([scenario_data])[features]
    
    orig_prob = MODELS[cancer_type].predict_proba(orig_df)[0][1]
    scen_prob = MODELS[cancer_type].predict_proba(scen_df)[0][1]
    
    diff = (orig_prob - scen_prob) * 100
    
    # 4. 결과 메시지 생성
    if diff > 0:
        msg = f"현재 위험도는 {orig_prob*100:.1f}%입니다. 만약 말씀하신 대로 습관을 개선하시면 위험도가 **{scen_prob*100:.1f}%**로 약 **{diff:.1f}%p 낮아질 것으로 예측**됩니다! ✨"
    elif diff < 0:
        msg = f"현재 위험도는 {orig_prob*100:.1f}%입니다. 하지만 해당 조건이 적용되면 위험도는 **{scen_prob*100:.1f}%**로 높아질 위험이 있습니다. ⚠️"
    else:
        msg = f"해당 변화는 예측 수치에 큰 영향을 주지 않지만, 꾸준한 관리가 중요합니다. (예상 확률: {scen_prob*100:.1f}%)"
    
    return msg

## 암종 분류 함수
def classify_cancer_type(user_input, client):
    prompt = f"""
    사용자의 입력 문장을 보고 분석하고자 하는 암의 종류를 분류하세요.
    반드시 [간암, 폐암, unknown] 중 하나만 출력하세요.
    사용자 문장: "{user_input}"
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 에이전트 컨트롤러
def run_cancer_agent(user_input, session_data, client):
    # 암종 분류
    if not session_data.get("cancer_type"):
        cancer_type = classify_cancer_type(user_input, client)
        if cancer_type == "unknown":
            return "안녕하세요! 간암과 폐암 중 어떤 암을 분석해 드릴까요?", session_data
        
        session_data["cancer_type"] = cancer_type
        session_data["collected_data"] = {k: None for k in FEATURE_CONFIG[cancer_type]}
        for d_key in ["hepatitis_b", "hepatitis_c", "cirrhosis_history"]:
            if d_key in session_data["collected_data"]:
                session_data["collected_data"][d_key] = 0
        
        return f"**{cancer_type}** 분석을 위해 정보를 수집할게요. {user_input}에 대해 더 자세히 말씀해 주시겠어요?", session_data

    # 정보 추출
    c_type = session_data["cancer_type"]
    extracted = gpt_extraction(c_type, session_data["history"], user_input, client)
    if extracted:
        for k, v in extracted.items():
            if v is not None: session_data["collected_data"][k] = v

    # 예측 또는 추가 질문
    missing_keys = [k for k, v in session_data["collected_data"].items() if v is None]
    
    if not missing_keys:
        # --- [추가] 시나리오 분석 트리거 판단 ---
        scenario_keywords = ["만약", "한다면", "끊으면", "줄이면", "빼면", "하면", "경우"]
        if any(word in user_input for word in scenario_keywords):
            comparison_result = handle_scenario_analysis(user_input, session_data["collected_data"], c_type, client)
            return comparison_result, session_data
        
        # 일반 결과 도출
        input_df = pd.DataFrame([session_data["collected_data"]])[FEATURE_CONFIG[c_type]]
        prob = MODELS[c_type].predict_proba(input_df)[0][1]
        return interpret_result_with_gpt(prob, client, c_type), session_data
    else:
        return generate_ask_question(session_data["collected_data"], missing_keys), session_data
#---------------------------------------------------------------------------------

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