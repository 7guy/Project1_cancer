import json
import pandas as pd
import joblib
from openai import OpenAI

MODEL_PATH = r"sub1_cancer_model.pkl"
model = joblib.load(MODEL_PATH)

def gpt_extraction(history, user_input, client):
    system_prompt = """
    너는 암 예측 모델을 위한 데이터 추출기야. 사용자의 입력에서 정보를 추출해서 JSON으로 반환해.
    필드: Age(정수), Gender(0:남, 1:여), BMI(실수), Smoking(0:No, 1:Yes), Alcohol(0~5), Family_History(정수), PhysicalActivity(0~10)
    키와 몸무게를 말하면 BMI를 계산해. (몸무게kg / 키m^2)
    반드시 JSON 형식 {"Age": 50, ...}만 출력해. 추출할 수 없으면 null로 채워.
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
    except Exception:
        return None

def validate_extracted(extracted):
    if not extracted:
        return extracted
    if extracted.get("Age") is not None and not (0 < extracted["Age"] < 120):
        extracted["Age"] = None
    if extracted.get("BMI") is not None and not (10 < extracted["BMI"] < 60):
        extracted["BMI"] = None
    if extracted.get("PhysicalActivity") is not None and not (0 <= extracted["PhysicalActivity"] <= 10):
        extracted["PhysicalActivity"] = None
    return extracted

def get_extraction_fail_reason(extracted, input_type):
    if extracted is None:
        return "GPT_ERROR"
    if all(v is None for v in extracted.values()):
        if input_type == "HEALTH":
            return "SYMPTOM_ONLY"
        return "NO_RELEVANT_INFO"
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
    return "암 발병 위험도 분석을 위해 아래 정보가 더 필요해요. 😊\n\n" + "\n".join(
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

def contains_abuse(text: str) -> bool:
    abuse_words = [
        "시발", "씨발", "ㅅㅂ", "병신", "미친", "좆", "엿같",
        "개같", "꺼져", "존나", "ㅈㄴ", "fuck", "shit", "tlqkf"
    ]
    text = text.lower()
    return any(word in text for word in abuse_words)

def classify_user_input(user_input: str):
    text = user_input.lower()

    if contains_abuse(text):
        return "ABUSE"

    greetings = ["안녕", "안녕하세요", "반가워", "하이", "hello", "ㅎㅇ"]
    health_keywords = [
        "아파", "피곤", "기침", "숨", "통증",
        "몸이", "컨디션", "불편", "증상",
        "잠", "불면", "수면", "못자", "어지러워", "근육통", "지쳐", "힘"
    ]

    if any(g in text for g in greetings):
        return "GREETING"

    if any(h in text for h in health_keywords):
        return "HEALTH"

    return "OTHER"

def is_extraction_failed(extracted: dict):
    if not extracted:
        return True
    return all(v is None for v in extracted.values())
