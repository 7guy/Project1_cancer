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
        # 🔥 이 부분을 추가해서 터미널(콘솔) 로그를 확인하세요!
        print(f"DEBUG [{cancer_type}] Input Features: {features}")
        return [features]

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
        return [features]

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
        # 1. 암종별 정확한 피처 순서 정의 (학습 시와 반드시 일치해야 함)
        feature_configs = {
            "TOTAL": [
                "Age", "Gender", "BMI", "Smoking", "Alcohol", 
                "Family_History", "PhysicalActivity"
            ],
            "LIVER": [
                "age", "gender", "bmi", "alcohol_consumption", "smoking_status", 
                "hepatitis_b", "hepatitis_c", "cirrhosis_history", "family_history_cancer", 
                "physical_activity_level", "diabetes"
            ]
        }

        # 2. 현재 암종에 맞는 컬럼명 가져오기
        cols = feature_configs.get(cancer_type)
        processed = normalize_input(raw_data, cancer_type)
        
        # 1. 반환값이 DataFrame이 아니면(리스트면) 변환
        if not isinstance(processed, pd.DataFrame):
            input_df = pd.DataFrame(processed, columns=cols)
        else:
            input_df = processed
        
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

# --- 4. 질문 생성 로직 (Missing Keys) ---
def get_missing_info_question(collected_data, cancer_type):
    # 1. [핵심 수정] 검문해야 할 리스트를 프롬프트 내용과 똑같이 맞춰줍니다.
    required_fields = {
        # 일반암 (기초 7종)
        "TOTAL": [
            "Age", "Gender", "BMI", "Smoking", "Alcohol", 
            "Family_History", "PhysicalActivity"
        ],
        
        # 간암 (프롬프트에 적힌 상세 항목 추가)
        "LIVER": [
            "Age", "Gender", "BMI", "Alcohol_Status", "Smoking_Status", 
            "Hepatitis_B", "Hepatitis_C", "Cirrhosis", "Diabetes", 
            "PhysicalActivity_Level"
        ],
        
        # 폐암 (프롬프트에 적힌 상세 항목 추가)
        "LUNG": [
            "Age", "Gender", "Smoking", "Passive_Smoker",
            "Air_Pollution", "Dust_Allergy", "Occupational_Hazards", 
            "Genetic_Risk", "Chronic_Disease", "Balanced_Diet", 
            "Obesity_Score", "Chest_Pain"
        ]
    }
    
    # 2. 누락된 항목 찾기
    missing = []
    # 해당 암종(cancer_type)에 필요한 필드 리스트를 가져와서 검사
    target_fields = required_fields.get(cancer_type, required_fields["TOTAL"])
    
    for f in target_fields:
        val = collected_data.get(f)
        if val is None or val == "": 
            missing.append(f)
    
    # 누락된 게 없으면 통과(None 반환)
    if not missing:
        return None 
    
    # 3. [핵심 수정] 사용자에게 보여줄 친절한 한국어 이름표
    label_map = {
        # 공통
        "Age": "나이", "Gender": "성별", "BMI": "키와 몸무게(또는 BMI)", 
        "Smoking": "흡연 여부", "Alcohol": "음주 빈도",
        "Family_History": "가족력", "PhysicalActivity": "운동량",
        
        # 간암 상세
        "Alcohol_Status": "음주 습관(가끔/자주/안함)", 
        "Smoking_Status": "흡연 이력(과거/현재/안함)", 
        "Hepatitis_B": "B형 간염 여부", 
        "Hepatitis_C": "C형 간염 여부", 
        "Cirrhosis": "간경변증 여부", 
        "Diabetes": "당뇨병 여부", 
        "PhysicalActivity_Level": "활동 강도(상/중/하)",

        # 폐암 상세
        "Passive_Smoker": "간접 흡연 노출 여부",
        "Air_Pollution": "공기 오염 노출 정도", 
        "Dust_Allergy": "먼지 알레르기 여부", 
        "Occupational_Hazards": "직업적 위험 요소 노출", 
        "Genetic_Risk": "폐암 유전적 위험도", 
        "Chronic_Disease": "만성 폐질환 여부", 
        "Balanced_Diet": "균형 잡힌 식단 여부", 
        "Obesity_Score": "비만도", 
        "Chest_Pain": "흉통(가슴 통증) 유무"
    }
    
    # 질문 만들기
    missing_labels = [label_map.get(m, m) for m in missing]
    
    # 너무 많이 물어보면 당황하니까 개수에 따라 말투 다르게
    if len(missing_labels) == 1:
        return f"정확한 {cancer_type} 분석을 위해 **{missing_labels[0]}** 정보를 알려주세요."
    else:
        # 3개 이상이면 줄바꿈으로 깔끔하게
        if len(missing_labels) > 3:
            list_str = "\n".join([f"- {item}" for item in missing_labels])
            return f"정밀한 분석을 위해 다음 정보들이 더 필요해요! 🧐\n\n{list_str}"
        else:
            return f"정확한 분석을 위해 **{', '.join(missing_labels)}** 정보를 알려주시겠어요?"

# --- [추가] 6. '만약에' 시나리오 재계산 로직 ---
def predict_scenario(current_data, changes, cancer_type):
    """
    detect_simulation_intent에서 추출한 changes 딕셔너리를 
    기존 데이터에 통째로 업데이트하여 시뮬레이션 결과를 반환합니다.
    """
    # 1. 원본 데이터 복사 (원본 유지, 시뮬레이션용 복제본 생성)
    scenario_data = current_data.copy()
    
    # 2. GPT가 추출한 변경사항들을 한 번에 반영
    if changes:
        scenario_data.update(changes)
        
        # [중요] 연관 변수 보정 (예: Smoking이 바뀌면 Smoking_Status도 변경)
        if "Smoking" in changes:
            scenario_data["Smoking_Status"] = "Never" if changes["Smoking"] == 0 else "Current"
        if "Alcohol" in changes:
            # 음주 빈도(0~5)에 따른 상태 매핑
            val = changes["Alcohol"]
            scenario_data["Alcohol_Status"] = "Never" if val == 0 else ("Occasional" if val <= 2 else "Regular")
            
    # 3. 이미 잘 만들어진 기존 예측 함수 재사용
    new_risk = predict_cancer_risk(scenario_data, cancer_type)
    
    return new_risk

# --- [기존] 3. 통합 예측 인터페이스 (수정 없음) ---
# ... 기존 predict_cancer_risk 함수 그대로 유지 ...

import json

def detect_simulation_intent(user_input, client):
    """
    사용자의 질문이 '현재 상태 입력'인지 '미래/가정(Simulation)'인지 판단하는 함수
    """
    system_prompt = """
    너는 헬스케어 AI의 '의도 분석기'야. 사용자의 입력이 "가정(Simulation)"인지 "단순 정보 제공"인지 판단해서 JSON으로 반환해.

    [판단 기준]
    1. Simulation (가정): "만약 ~라면?", "~하면 어때?", "나중에 ~하면?", "50살이 되면?", "담배 끊으면?" 같이 조건 변경이나 미래 예측을 묻는 경우.
    2. None (정보 제공): "나 30살이야", "담배 안 펴", "키 170이야" 같이 자신의 현재 정보를 설명하는 경우.

    [JSON 출력 형식]
    - Simulation일 때: {"type": "simulation", "changes": {"필드명": 값}}
    - 아닐 때: {"type": "none"}

    [필드명 매핑 규칙] (반드시 이 영문 키를 사용해!)
    - 나이 -> Age (정수)
    - 성별 -> Gender (0:남, 1:여)
    - 담배/흡연 -> Smoking (0:안함, 1:함)
    - 술/음주 -> Alcohol (0:안함 ~ 5:매일)
    - 운동 -> PhysicalActivity (0:안함 ~ 10:매일)  <-- 중요: Exercise라고 쓰지 마!
    - 가족력 -> Family_History (0:없음, 1:있음)
    - BMI -> BMI

    [예시]
    User: "50살에 담배 피면 암 걸릴까?"
    Output: {"type": "simulation", "changes": {"Age": 50, "Smoking": 1}}

    User: "운동 안 하고 술 매일 마시면?"
    Output: {"type": "simulation", "changes": {"PhysicalActivity": 0, "Alcohol": 5}}

    User: "나 지금 24살이고 여자야."
    Output: {"type": "none"}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            response_format={"type": "json_object"},
            temperature=0  # 의도 파악은 정확해야 하므로 창의성(temperature)을 0으로 설정
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ 의도 파악 중 에러: {e}")
        return {"type": "none"}
    
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
    추출 대상(폐암): Air_Pollution(1~8), Dust_Allergy(1~8), Occupational_Hazards(1~8), Genetic_Risk(1~8), Chronic_Disease(1~8), Balanced_Diet(1~8), Obesity_Score(1~8), Passive_Smoker(1~8), Chest_Pain(1~8), Weight_Loss(1~8), Shortness_Breath(1~8), Dry_Cough(1~8)
    """
    
    response = client.chat.completions.create(
        model="gpt-4o",
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