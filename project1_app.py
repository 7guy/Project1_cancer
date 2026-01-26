import streamlit as st
from openai import OpenAI
from project1_logic import (
    gpt_extraction,
    generate_ask_question,
    interpret_result_with_gpt,
    model,
    classify_user_input,
    is_extraction_failed
)
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
    # 유저 메시지 저장 및 출력
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = None  # 이 한 줄이 핵심

        # 🔹 0. 사용자 입력 유형 판별
        input_type = classify_user_input(prompt)

        # 🔹 0. 비속어 처리 (최우선)
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

        # 🔹 1. 인사 처리
        if input_type == "GREETING":
            response = (
                "안녕하세요 😊\n"
                "암 발병 위험도를 참고용으로 안내해드리는 서비스예요.\n\n"
                "나이, 성별, 흡연 여부처럼 편한 정보부터 알려주시면 시작할게요!"
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        # 🔹 2. 정보 추출
        extracted = gpt_extraction(st.session_state.messages, prompt, client)

        # 🔹 3. 추출 실패 처리
        if is_extraction_failed(extracted):
            response = (
                "이 서비스는 암 발병 위험도를 참고용으로 안내해드려요 😊\n"
                "나이, 성별 같은 기본 정보부터 알려주시면 분석을 시작할 수 있어요."
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.stop()

        # 🔹 4. 추출 성공 → 세션에 반영
        for k, v in extracted.items():
            if v is not None:
                st.session_state.collected_data[k] = v

        render_checklist()

        # 🔹 5. 누락 정보 계산
        missing = [
            k for k, v in st.session_state.collected_data.items()
            if v is None
        ]

        # 🔹 6. 상태별 분기
        if st.session_state.step == "COLLECTING":
            if missing:
                if input_type == "HEALTH":
                    st.session_state.health_count += 1

                    if st.session_state.health_count == 1:
                        response = (
                            "말씀해주신 증상은 참고할게요 🙂\n\n"
                            "암 위험도 분석을 위해 기본 정보도 함께 알려주시면 좋아요."
                        )

                    elif st.session_state.health_count == 2:
                        question_text = generate_ask_question(
                            st.session_state.collected_data, missing
                        )
                        response = (
                            "말씀해주신 증상은 참고할게요 🙂\n\n"
                            "먼저 아래 기본 정보부터 마저 알려주세요.\n\n"
                            + question_text
                        )

                    else:
                        response = (
                            "말씀해주신 증상은 충분히 이해했어요 🙂\n\n"
                            "지금은 암 위험도 판단을 위해 아래 항목 중 선택해서 알려주세요 👇\n\n"
                            "1️⃣ 키(cm)와 몸무게(kg) 또는 BMI\n"
                            "2️⃣ 흡연 여부 (예: 비흡연 / 흡연)\n"
                            "3️⃣ 음주 빈도 (예: 거의 안 함 / 주 1~2회)\n"
                            "4️⃣ 가족력 (예: 없음 / 있음)\n"
                            "5️⃣ 운동량 (예: 거의 안 함 / 주 3회 이상)\n\n"
                            "👉 예: 키는 160이고 몸무게는 58이야. 담배를 자주 피고 운동을 거의 안 해"
                        )
                else:
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
                response += (
                    "\n\n**추가적인 상담이 필요하시다면 "
                    "간암이나 위암 정밀 진단도 가능합니다. 계속할까요?**"
                )
                st.session_state.step = "ASK_ADDITIONAL"

        elif st.session_state.step == "ASK_ADDITIONAL":
            if any(word in prompt for word in ["응", "네", "해줘", "좋아", "간암", "위암"]):
                response = "알겠습니다. 정밀 진단을 위해 추가 정보를 확인하겠습니다."
            else:
                response = "상담을 종료합니다. 건강한 하루 되세요!"
                st.session_state.step = "FINISHED"

        if response is not None:
            st.markdown(response)
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
