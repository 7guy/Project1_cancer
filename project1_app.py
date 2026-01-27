import streamlit as st
from openai import OpenAI
from project1_logic import gpt_extraction, generate_ask_question, interpret_result_with_gpt, run_cancer_agent, model, MODELS, FEATURE_CONFIG
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
if 'agent_session' not in st.session_state: # 로직 엔진과 상태를 공유하기 위한 통합 세션
    st.session_state.agent_session = {"cancer_type": None, "collected_data": {}, "history": []}
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = {k: None for k in ["Age", "Gender", "BMI", "Smoking", "Alcohol", "Family_History", "PhysicalActivity"]}
if 'step' not in st.session_state:
    st.session_state.step = "COLLECTING"

# --- 2. 사이드바 (실시간 대시보드) ---
with st.sidebar:
    st.subheader("📋 입력 정보 확인")

    checklist_area = st.empty()

    def render_checklist():
        # 암종이 정해지면 해당 암종의 피처 리스트를 가져옴
        current_type = st.session_state.agent_session.get("cancer_type")
        if current_type:
            check_items = {k: k for k in FEATURE_CONFIG.get(current_type, [])}
        else:
            check_items = {
                "Age": "나이", "Gender": "성별", "BMI": "BMI(또는 키와 몸무게)",
                "Smoking": "흡연 여부", "Alcohol": "음주 빈도",
                "Family History": "가족력", "Physical Activity": "운동량"
            }

        with checklist_area.container():
            for label, key in check_items.items():

                value = st.session_state.agent_session["collected_data"].get(key)
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    #암종 분류, 데이터 추출, 시나리오 분석
    response_text, updated_session = run_cancer_agent(
        prompt, 
        st.session_state.agent_session, 
        client
    )
    
    #세션에 반영
    st.session_state.agent_session = updated_session
    st.session_state.collected_data = updated_session["collected_data"]

    with st.chat_message("assistant"):
        render_checklist()
        
        missing = [k for k, v in st.session_state.collected_data.items() if v is None]
        
        if st.session_state.step == "COLLECTING":
            if missing:
                response = response_text
            else:
                #시나리오 분석
                scenario_keywords = ["만약", "한다면", "끊으면", "줄이면", "빼면", "하면", "경우"]
                if any(word in prompt for word in scenario_keywords):
                    response = response_text # 에이전트가 생성한 비교 분석 텍스트 사용
                else:
                    st.write("🔄 모든 정보가 수집되었습니다. 분석 중입니다...")
                    c_type = st.session_state.agent_session["cancer_type"]
                    #해당 암종 모델에 필요한 피처만 추출
                    input_df = pd.DataFrame([st.session_state.collected_data])[FEATURE_CONFIG[c_type]]
                    prob = MODELS[c_type].predict_proba(input_df)[0][1]
                    
                    response = interpret_result_with_gpt(prob, client, c_type)
                    response += "\n\n**추가적인 상담이나 '만약 ~한다면?' 시뮬레이션도 가능합니다. 궁금한 점이 있으신가요?**"
                    st.session_state.step = "ASK_ADDITIONAL"
        
        elif st.session_state.step == "ASK_ADDITIONAL":
            #추가 상담에서 시나리오 분석
            scenario_keywords = ["만약", "한다면", "끊으면", "줄이면", "빼면", "하면", "경우"]
            if any(word in prompt for word in scenario_keywords):
                response = response_text
            elif any(word in prompt for word in ["응", "네", "해줘", "좋아", "간암", "위암"]):
                response = "알겠습니다. 정밀 진단을 위해 추가 정보를 확인하겠습니다."
            else:
                response = "상담을 종료합니다. 건강한 하루 되세요!"
                st.session_state.step = "FINISHED"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.session_state.agent_session["history"].append({"role": "assistant", "content": response})