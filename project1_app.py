import streamlit as st
from openai import OpenAI
from project1_logic import gpt_extraction, generate_ask_question, interpret_result_with_gpt, model
import pandas as pd
import os
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI 암 진단 챗봇", layout="wide")
st.title("🩺 AI 암 발병 위험도 진단 서비스")

# --- 1. 세션 상태 초기화 ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = {k: None for k in ["Age", "Gender", "BMI", "Smoking", "Alcohol", "Family_History", "PhysicalActivity"]}
if 'step' not in st.session_state:
    st.session_state.step = "COLLECTING"

# --- 2. 사이드바 (실시간 대시보드) ---
with st.sidebar:
    st.subheader("📋 입력 정보 확인")

    checklist_area = st.empty()

    def render_checklist():
        check_items = {
            "Age": "나이",
            "Gender": "성별",
            "BMI": "BMI(또는 키와 몸무게)",
            "Smoking": "흡연 여부",
            "Alcohol": "음주 빈도",
            "Family History": "가족력",
            "Physical Activity": "운동량"
        }

        with checklist_area.container():
            for label, key in check_items.items():
                value = st.session_state.collected_data.get(key)
                status = "⬜" if value is None else "✅"
                st.write(f"{status} {label}")

    render_checklist()

    if st.button("🔄 상담 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 3. 대화 내용 출력 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 4. 사용자 입력 및 총괄 로직 (Orchestrator) ---
if prompt := st.chat_input("증상이나 건강 정보를 입력하세요..."):
    # 유저 메시지 저장 및 출력
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # A. 정보 추출
        extracted = gpt_extraction(st.session_state.messages, prompt, client)
        if extracted:
            for k, v in extracted.items():
                if v is not None: st.session_state.collected_data[k] = v
            
            # rerun 없이 체크 즉시 갱신
            render_checklist()
        
        # B. 상태별 분기 로직 (Orchestrator)
        missing = [k for k, v in st.session_state.collected_data.items() if v is None]
        
        if st.session_state.step == "COLLECTING":
            if missing:
                # 정보가 부족하면 자연스러운 질문 생성
                response = generate_ask_question(st.session_state.collected_data, missing)
            else:
                # 모든 정보 수집 완료 -> 모델 호출
                st.write("🔄 모든 정보가 수집되었습니다. 분석 중입니다...")
                data = st.session_state.collected_data
                input_df = pd.DataFrame([{
                    'Age': data['Age'], 'Gender': data['Gender'], 'BMI': data['BMI'],
                    'Smoking': data['Smoking'], 'Alcohol': data['Alcohol'],
                    'Family_History': 1 if data['Family_History'] > 0 else 0,
                    'PhysicalActivity': data['PhysicalActivity']
                }])
                prob = model.predict_proba(input_df)[0][1]
                
                # 결과 해석 (GPT)
                response = interpret_result_with_gpt(prob, client)
                response += "\n\n**추가적인 상담이 필요하시다면 간암이나 위암 정밀 진단도 가능합니다. 계속할까요?**"
                st.session_state.step = "ASK_ADDITIONAL"
        
        elif st.session_state.step == "ASK_ADDITIONAL":
            # 의도 파악 로직 (간단히 구현)
            if any(word in prompt for word in ["응", "네", "해줘", "좋아", "간암", "위암"]):
                response = "알겠습니다. 정밀 진단을 위해 추가 정보를 확인하겠습니다. (로직 확장 가능)"
                # 여기서 step을 LIVER 등으로 전환 가능
            else:
                response = "상담을 종료합니다. 건강한 하루 되세요!"
                st.session_state.step = "FINISHED"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})