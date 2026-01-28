import pandas as pd
import joblib
import openai
import json
import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# API 키 설정 
load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

# 모델 로드
model = joblib.load("C:\py_project\module_project\sub1_cancer_model.pkl")
print(f" 모델 로드 완료: {'C:\py_project\module_project\sub1_cancer_model.pkl'}")


system_prompt = """
    너는 암 예측 모델을 위한 데이터 추출기야. 사용자의 입력에서 다음 정보를 추출해서 JSON으로 반환해
    모르는 정보가 있으면 null로 남겨

    1. 사용자가 '키'와 '몸무게'를 말하면, 네가 직접 BMI를 계산해서 'BMI' 필드에 넣어.
    (공식: 몸무게(kg) / (키(m) * 키(m)))
    2. 사용자가 이미 BMI 수치를 말했으면 그대로 넣어.

    [필수 기준]
    1. Age: 정수 (나이)
    2. Gender: 0 (Male), 1 (Female)
    3. BMI: 실수 (체질량지수, 모르면 키/몸무게로 계산하거나 평균값 25 가정)
    4. Smoking: 0 (No), 1 (Yes)
    5. Alcohol: 0~5 사이 실수 (0 : 안마심, 보통: 2.5, 5 : 매일 폭음)
    6. Family_History: 가족 중 암 환자 수 (정수)
    7. PhysicalActivity: 0~10 사이 실수 (0: 운동 안함, 10: 매일 고강도 운동)
    
    출력 형식: {"Age": 50, "Gender": 1, ...} 만 출력해.
    """

# gpt질문 -> json 데이터 추출
def gpt_extraction(history, user_input):
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history) # 이전 대화 기억 추가
    messages.append({"role": "user", "content": user_input})

    try:
        #답변 생성
        response = client.chat.completions.create( 
            model="gpt-5.2", 
            messages=messages,
            response_format={"type": "json_object"} # 기계가 읽을 수 있는 JSON으로 포맷
        )
        return json.loads(response.choices[0].message.content) # 답변 추출
    except Exception as e: # 에러
        print(f" GPT 오류: {e}")
        return None
    
def get_health_advice(user_data, prob, risk_level):
    prompt = f"""
    [사용자 건강 정보]
    {user_data}
    
    [AI 예측 결과]
    - 암 발병 위험도: {risk_level}
    - 예측 확률: {prob:.2%}
    
    [요청 사항]
    위 정보를 바탕으로 사용자에게 도움이 될 구체적인 건강 조언을 3줄로 요약해줘.
    - 특히 흡연, 음주, BMI, 운동량 수치를 보고 개선이 필요한 부분을 콕 집어서 말해줘.
    - 말투는 따뜻하고 전문적인 상담사처럼 해줘.
    - 가장 첫 줄에는 반드시 "이 결과는 단순 예측이며 의학적 진단이 아닙니다."라고 명시해줘.
    - 끝줄에는 반드시 '추가적인 상담이 필요하시다면 연결해드리도록 하겠습니다.(간암,위암)'라고 명시해줘.
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # gpt-4o 또는 gpt-3.5-turbo 사용 권장
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"조언 생성 중 오류: {e}")
        return "건강 조언을 불러오는 데 실패했습니다."

# 메인실행 코드    
def main():
    print("\n 사용자 입력 기반 암 위험도 측정 AI 입니다. (종료 : q)")
    print("--------------------------------------------------")

    #사전 매핑
    name_map = {
        "Age": "나이",
        "Gender": "성별",
        "BMI": "BMI(체질량지수)",
        "Smoking": "흡연 여부",
        "Alcohol": "주간 음주량",
        "Family_History": "가족력",
        "PhysicalActivity": "주간 운동량"
    }
    
    # 대화 내용 저장 리스트
    history = []
    
    # 수집된 데이터 저장 딕셔너리, 초기화 진행
    collected_data = {
        "Age": None, "Gender": None, "BMI": None, "Smoking": None,
        "Alcohol": None, "Family_History": None, "PhysicalActivity": None
    }

    while True:
        user_input = input("\n 사용자의 정보를 입력해주세요(이름,나이,성별,BMI,...): ")
        if user_input.lower() in ['q', 'exit', 'quit']: #사용자가 대문자로 사용하든소문자로 사용하든 상관X
            print("상담을 종료합니다.")
            break

        # gpt 데이터 추출 요청
        data = gpt_extraction(history, user_input)

        if data:
            # 대화 기록에 추가 (문맥 유지)
            history.append(
                {
                "role": "user", 
                "content": user_input
                }
                )
            history.append(
                {
                "role": "assistant",
                "content": json.dumps(data)
                }
                )
        #데이터 {data} 하나씩 확인, v : 항목이름, k : 항목의값    
        for k, v in data.items():
                if v is not None: 
                    collected_data[k] = v
        # v : none ->  k=missing
        missing = [k for k, v in collected_data.items() if v is None]
        
        if missing:
            # missing 항목 한국어로 변환
            missing_ko = [name_map[k] for k in missing]
            # 수집 데이터 한국어로 변환 
            current_ko = {name_map[k]: v for k, v in collected_data.items()}
            print(f"다음 정보가 더 필요합니다 -> {', '.join(missing_ko)}")
            print(f"(현재 수집됨: {current_ko})")
        else:
            print("\n 분석 중입니다")

            # 전처리 (가족력 1명 이상이면 1로 변환)
            fam_hist = 1 if collected_data['Family_History'] > 0 else 0
                
            input_df = pd.DataFrame([{
                'Age': collected_data['Age'],
                'Gender': collected_data['Gender'],
                'BMI': collected_data['BMI'],
                'Smoking': collected_data['Smoking'],
                'Alcohol': collected_data['Alcohol'],
                'Family_History': fam_hist,
                'PhysicalActivity': collected_data['PhysicalActivity']
            }])

            # 모델 암 확률 예측 [0] : 정상확률, [1] : 암 확률
            prob = model.predict_proba(input_df)[0][1]
            # predict 사용시 암입니다. 아닙니다로 확정지어 신뢰도를 위해 if로 대신함
            risk_level = "높음" if prob >= 0.5 else "낮음" 

            advice = get_health_advice(collected_data, prob, risk_level)

            print("\n" + "="*50)
            print(f" 분석 결과")
            print(f" 암 발병 위험도: {prob:.2%} ({risk_level})")
            print(f" AI 건강 조언\n{advice}") # 조언 출력
            print("-" * 50)
            
            #상담 종료 및 초기화
            retry = input("\n다시 상담하시겠습니까? (y/n): ")
            if retry.lower() == 'y':
                collected_data = {k: None for k in collected_data}
                history = [] # 초기화
                print("\n 초기화되었습니다.")
            else:
                break

if __name__ == "__main__":
    main()