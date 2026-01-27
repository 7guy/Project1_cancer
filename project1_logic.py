import pandas as pd
import numpy as np
import joblib
import os
import traceback

# --- 1. 자원 로드 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_resources():
    resources = {
        "TOTAL": {"model": None},
        "LIVER": {"model": None, "scaler": None},
        "LUNG": {"model": None}
    }
    try:
        # 1. 통합암 (Binary)
        resources["TOTAL"]["model"] = joblib.load(os.path.join(BASE_DIR, "sub1_cancer_model.pkl"))
        # 2. 간암 (XGB + Scaler + 문자열 범주)
        resources["LIVER"]["model"] = joblib.load(os.path.join(BASE_DIR, "liver_cancer_xgb_model.pkl"))
        resources["LIVER"]["scaler"] = joblib.load(os.path.join(BASE_DIR, "liver_scaler.pkl"))
        # 3. 폐암 (Logistic + 1-8 Scale)
        resources["LUNG"]["model"] = joblib.load(os.path.join(BASE_DIR, "lung_xgb_model.pkl"))
        print("✅ 모든 모델(TOTAL, LIVER, LUNG) 및 스케일러 로드 완료")
    except Exception as e:
        print(f"❌ 모델 로드 중 오류: {e}")
    return resources

res = load_resources()

def scale_it(val, max_val=8): 
    # val(사용자 입력 0~5)을 1~8 사이의 정수로 변환하는 공식, 폐암 데이터 입력에 씀
    return int((float(val) / 5 * (max_val - 1)) + 1)

# logic.py 상단에 추가
def safe_cast(val, to_type, default):
    """None이거나 변환 실패 시 기본값을 반환하는 도우미 함수"""
    if val is None or val == "":
        return default
    try:
        return to_type(val)
    except (ValueError, TypeError):
        return default

    # ... (나머지 변수들도 동일하게 safe_cast 사용)

# --- 2. 모델별 데이터 정규화 엔진 (가장 중요한 부분) ---
def normalize_input(raw_data, cancer_type):
    """
    사용자가 입력한 표준 데이터를 각 모델의 '입구' 규격에 맞춰 변환
    """
    # 기본 공통 변수 (안전장치)
    age = safe_cast(raw_data.get('Age'), int, 50)
    gender = safe_cast(raw_data.get('Gender'), int, 0) # 0:남, 1:여
    bmi = safe_cast(raw_data.get('BMI'), float, 25.0)
    smoke_binary = safe_cast(raw_data.get('Smoking'), int, 0)
    alcohol_score = safe_cast(raw_data.get('Alcohol'), float, 2.5) # 챗봇 표준 0~5

    # --- A. 통합암 모델 (TOTAL) ---
    if cancer_type == "TOTAL":
        # 특징: 표준 0~1, 0~5 척도 그대로 사용
        features = [
            age, gender, bmi, smoke_binary, alcohol_score,
            safe_cast(raw_data.get('Family_History'), float, 0),
            safe_cast(raw_data.get('PhysicalActivity'), float, 5)
        ]
        return np.array([features])

    # --- B. 간암 모델 (LIVER) ---
    elif cancer_type == "LIVER":
        # 특징: 문자열 범주 매핑 필요
        alc_map = {'Never': 0, 'Occasional': 1, 'Regular': 2}
        smk_map = {'Never': 0, 'Former': 1, 'Current': 2}
        act_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        
        # 스케일러 학습 순서 준수
        features = [
            age, gender, bmi,
            alc_map.get(raw_data.get('Alcohol_Status', 'Never'), 0),
            smk_map.get(raw_data.get('Smoking_Status', 'Never'), 0),
            float(raw_data.get('Hepatitis_B', 0)),
            float(raw_data.get('Hepatitis_C', 0)),
            float(raw_data.get('Cirrhosis', 0)),
            float(raw_data.get('Family_History', 0)),
            act_map.get(raw_data.get('PhysicalActivity_Level', 'Moderate'), 1),
            float(raw_data.get('Diabetes', 0))
        ]
        return np.array([features])

    # --- C. 폐암 모델 (LUNG) ---
    elif cancer_type == "LUNG":
        # 특징: 1-2(Gender), 1-8(Alcohol/Smoking) 척도 및 컬럼명 준수
        input_df = pd.DataFrame([{
            'Age': age,
            'Gender': gender + 1,
            'Air Pollution': int(raw_data.get('Air_Pollution', 5)), # 누락된 값들
            'Alcohol use': scale_it(raw_data.get('Alcohol', 2.5)),
            'Dust Allergy': int(raw_data.get('Dust_Allergy', 5)),
            'OccuPational Hazards': int(raw_data.get('Occupational_Hazards', 4)),
            'Genetic Risk': int(raw_data.get('Genetic_Risk', 4)),
            'chronic Lung Disease': int(raw_data.get('Chronic_Disease', 3)),
            'Balanced Diet': int(raw_data.get('Balanced_Diet', 5)),
            'Obesity': int(raw_data.get('Obesity_Score', 4)),
            'Smoking': 8 if smoke_binary == 1 else 1,
            'Passive Smoker': scale_it(raw_data.get('Passive_Smoker', 0)),
            'Chest Pain': int(raw_data.get('Chest_Pain', 4)),
            'Weight Loss': int(raw_data.get('Weight_Loss', 3)),        # 수정: .get 적용
            'Shortness of Breath': int(raw_data.get('Shortness_Breath', 4)), # 수정: .get 적용
            'Dry Cough': int(raw_data.get('Dry_Cough', 3)),            # 수정: .get 적용
        }])
    
    # [주의] 팀원 모델의 피쳐 순서가 다를 수 있으므로 
    # model.feature_names_in_이 있다면 그 순서대로 재배열해야 합니다.
    cols = res["LUNG"]["model"].feature_names_in_
    input_df = input_df[cols] 
    
    return input_df

# --- 3. 통합 예측 인터페이스 ---
def predict_cancer_risk(raw_data, cancer_type="TOTAL"):
    try:
        # 1. 데이터를 정규화 (여기서 딕셔너리 혹은 리스트가 나옴)
        processed = normalize_input(raw_data, cancer_type)
        
        # 2. [핵심] 모델이 인식할 수 있도록 DataFrame으로 변환
        input_df = pd.DataFrame(processed) 
        
        if cancer_type == "LIVER":
            model = res["LIVER"]["model"]
            scaler = res["LIVER"]["scaler"]
            # 스케일러가 있다면 스케일링 적용
            final_input = scaler.transform(input_df) if scaler else input_df
            prob = model.predict_proba(final_input)[0][1]
            
        elif cancer_type == "LUNG":
            model = res["LUNG"]["model"]
            # input_df를 전달하여 Warning 해결 및 0% 탈출
            prob = model.predict_proba(input_df)[0][1]
            
        else: # TOTAL
            model = res["TOTAL"]["model"]
            # input_df를 전달
            prob = model.predict_proba(input_df)[0][1]
            
        return round(float(prob) * 100, 1)
        
    except Exception as e:
        print(f"❌ {cancer_type} 예측 중 오류 발생")
        traceback.print_exc()
        return 0.0
        
    except Exception as e:
        print(f"❌ {cancer_type} 예측 중 오류 발생")
        traceback.print_exc()
        return 0.0

# --- 4. 질문 생성 로직 (Missing Keys) ---
def get_missing_info_question(collected_data, cancer_type):
    """
    어떤 정보가 더 필요한지 알려주는 함수
    """
    # 모델별 필요 필드 정의
    required_fields = {
        "TOTAL": ["Age", "Gender", "BMI", "Smoking", "Alcohol"],
        "LIVER": ["Age", "Gender", "BMI", "Alcohol_Status", "Smoking_Status", "Hepatitis_B"],
        "LUNG": ["Age", "Gender", "Smoking", "Occupational_Hazards", "Obesity_Score"]
    }
    
    missing = [f for f in required_fields[cancer_type] if f not in collected_data]
    
    if not missing:
        return None # 모든 정보 수집 완료
    
    # 한국어 매핑 (예시)
    label_map = {"Age": "나이", "Gender": "성별", "BMI": "체질량지수(BMI)", 
                 "Alcohol_Status": "음주 습관", "Smoking_Status": "흡연 상태",
                 "Occupational_Hazards": "직업적 위험 요소"}
    
    return f"정확한 진단을 위해 {label_map.get(missing[0], missing[0])} 정보를 알려주세요."

# --- [추가] 5. 사용자의 후속 질문 의도 판별 ---
def classify_intent(user_input):
    """
    사용자의 추가 질문이 '가정(What-if)'인지 '타 암종 상세분석'인지 판별
    (이 기능은 GPT 프롬프트에 포함시켜 JSON으로 받는 것이 가장 정확하지만, 
    로직상 분류 기준을 세워둡니다.)
    """
    # 실제 구현은 app.py 내의 gpt_extraction 시 시스템 프롬프트에 
    # 'intent': 'what_if' | 'detail_request' | 'general' 을 추가하도록 설정합니다.
    pass

# --- [추가] 6. '만약에' 시나리오 재계산 로직 ---
def predict_scenario(current_data, change_key, change_value, cancer_type):
    """
    "담배를 끊는다면?" 처럼 특정 조건을 바꿨을 때의 위험도를 시뮬레이션
    """
    # 1. 기존 데이터 복사
    scenario_data = current_data.copy()
    
    # 2. 특정 필드 값 변경
    # 예: "담배를 끊는다면?" -> Smoking=0, Smoking_Status='Never'
    if change_key == "Smoking":
        scenario_data["Smoking"] = 0
        scenario_data["Smoking_Status"] = "Never"
    elif change_key == "Alcohol":
        scenario_data["Alcohol"] = 0
        scenario_data["Alcohol_Status"] = "Never"
        scenario_data["Alcohol_Score"] = 0
        
    # 3. 변경된 데이터로 다시 예측
    new_risk = predict_cancer_risk(scenario_data, cancer_type)
    return new_risk

# --- [기존] 3. 통합 예측 인터페이스 (수정 없음) ---
# ... 기존 predict_cancer_risk 함수 그대로 유지 ...

import json

# --- 7. GPT 정보 추출 함수 (사용자 입력 -> JSON) ---
def gpt_extraction(messages, user_input, client):
    """
    사용자의 자연어 입력에서 모델에 필요한 Key값들을 추출
    """
    system_prompt = """
    사용자의 대화에서 다음 정보를 JSON으로 추출해줘. 
    알 수 없는 정보는 null로 표시해.
    추출 대상: Age, Gender(남:0, 여:1), BMI, Smoking(안함:0, 함:1), Alcohol(0~5), Family_History(0,1), PhysicalActivity(0~10)
    추출 대상(간암): Alcohol_Status(Never, Occasional, Regular), Smoking_Status(Never, Former, Current), Hepatitis_B, Hepatitis_C, Cirrhosis, Diabetes, PhysicalActivity_Level(Low, Moderate, High)
    추출 대상(폐암): Air_Pollution, Dust_Allergy, Occupational_Hazards, Genetic_Risk, Chronic_Disease, Balanced_Diet, Obesity_Score, Passive_Smoker, Chest_Pain, Cough_Blood
    """
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

# --- 8. GPT 결과 해석 및 조언 함수 ---
def interpret_result_with_gpt(prob, client, cancer_type):
    """
    계산된 확률을 바탕으로 사용자에게 건강 조언 제공
    """
    prompt = f"사용자의 {cancer_type} 발병 위험도가 {prob}%로 계산되었습니다. 이 결과에 대한 짧은 분석과 건강 조언을 3줄 내외로 친절하게 해주세요."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content