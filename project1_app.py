import streamlit as st
from openai import OpenAI
from project1_logic import (
    gpt_extraction,
    generate_ask_question,
    interpret_result_with_gpt,
    model,
    classify_user_input,
    is_extraction_failed,
    get_extraction_fail_reason,  
    validate_extracted    
)
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI 암 발병 위험도 판단 챗봇", layout="wide")
st.title("🩺 AI 암 발병 위험도 판단 서비스")

# --- 1. 세션 상태 초기화 ---
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = {
        k: None for k in [
            "Age", "Gender", "BMI",
            "Smoking", "Alcohol",
            "Family_History", "PhysicalActivity"
        ]
    }
if 'step' not in st.session_state:
    st.session_state.step = "COLLECTING"
if 'health_count' not in st.session_state:
    st.session_state.health_count = 0
if 'invalid_count' not in st.session_state:
    st.session_state.invalid_count = 0

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
            "Family_History": "가족력",
            "PhysicalActivity": "운동량"
        }

        with checklist_area.container():
            for key, label in check_items.items():
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = None

        input_type = classify_user_input(prompt)

        if input_type == "ABUSE":
            response = (
                "많이 답답하신 것 같아요 😔\n\n"
                "그래도 제가 도와드릴 수 있어요.\n"
                "암 위험도 판단을 위해 아래 예시처럼 정보를 알려주세요.\n\n"
                "👉 예: 키는 160이고 몸무게 58이야. 담배를 자주 피고 운동은 거의 안 해"
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        if input_type == "GREETING":
            response = (
                "안녕하세요 😊\n"
                "암 발병 위험도를 참고용으로 안내해드리는 서비스예요.\n\n"
                "나이, 성별, 흡연 여부처럼 편한 정보부터 알려주시면 시작할게요!"
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        extracted = gpt_extraction(st.session_state.messages, prompt, client)
        extracted = validate_extracted(extracted)

        if is_extraction_failed(extracted):
            st.session_state.invalid_count += 1

            reason = get_extraction_fail_reason(extracted, input_type)

            if reason == "NO_RELEVANT_INFO":
                response = "현재 입력만으로는 분석이 어려워요 😅 기본 정보를 알려주세요."
            elif reason == "SYMPTOM_ONLY":
                response = "증상은 참고했어요 🙂 나이처럼 기본 정보도 우선 알려주시면 좋아요."
            elif reason == "GPT_ERROR":
                response = "일시적인 오류가 발생했어요. 다시 한 번 입력해 주세요."
            else:
                if st.session_state.invalid_count == 1:
                    response = "분석을 위해 기본 정보가 필요해요 🙂"
                elif st.session_state.invalid_count == 2:
                    response = "아래 예시처럼 입력해주시면 좋아요 👇\n\n👉 예: 25살 여자, 비흡연"
                else:
                    response = (
                        "입력 예시를 하나 선택해주세요 👇\n"
                        "1️⃣ 키(cm)와 몸무게(kg) 또는 BMI\n"
                        "2️⃣ 흡연 여부 (예: 비흡연 / 흡연)\n"
                        "3️⃣ 음주 빈도 (예: 거의 안 함 / 주 1~2회)\n"
                        "4️⃣ 가족력 (예: 없음 / 있음)\n"
                        "5️⃣ 운동량 (예: 거의 안 함 / 주 3회 이상)\n\n"
                        "👉 예: 키는 160이고 몸무게는 58이야. 담배를 자주 피고 운동을 거의 안 해"
                    )

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        for k, v in extracted.items():
            if v is not None:
                st.session_state.collected_data[k] = v

        render_checklist()

        missing = [
            k for k, v in st.session_state.collected_data.items()
            if v is None
        ]

        if st.session_state.step == "COLLECTING":
            if missing:
                response = generate_ask_question(
                    st.session_state.collected_data, missing
                )
            else:
                st.write("🔄 모든 정보가 수집되었습니다. 분석 중입니다...")

                data = st.session_state.collected_data
                input_df = pd.DataFrame([{
                    "Age": data["Age"],
                    "Gender": data["Gender"],
                    "BMI": data["BMI"],
                    "Smoking": data["Smoking"],
                    "Alcohol": data["Alcohol"],
                    "Family_History": 1 if data["Family_History"] > 0 else 0,
                    "PhysicalActivity": data["PhysicalActivity"]
                }])

                prob = model.predict_proba(input_df)[0][1]
                response = interpret_result_with_gpt(prob, client)
                response += ("\n\n추가적인 상담이 필요하시다면 "
                            "간암이나 폐암 위험도 판단도 가능합니다. 계속 진행할까요?")
                st.session_state.step = "ASK_ADDITIONAL"

        if response is not None:
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
