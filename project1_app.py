import streamlit as st
from openai import OpenAI
from project1_logic import gpt_extraction, generate_ask_question, interpret_result_with_gpt, model
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI 암 진단 챗봇", layout="wide")
st.title("🩺 AI 암 발병 위험도 판단 서비스")

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
        # A. 정보 추출
        extracted = gpt_extraction(st.session_state.messages, prompt, client)
        if extracted:
            for k, v in extracted.items():
                if k in st.session_state.collected_data and v is not None:
                    st.session_state.collected_data[k] = v 
            # rerun 없이 체크 즉시 갱신
            render_checklist()
        
        # B. 상태별 분기 로직 (Orchestrator)
        missing = [k for k, v in st.session_state.collected_data.items() if v is None]
        
        # step 1 : 필수 건강 정보 수집 단계
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
                response += "\n\n**추가적인 상담이 필요하시다면 간암이나 폐암 정밀 상담도 가능합니다. 계속할까요?**"
                st.session_state.step = "ASK_ADDITIONAL"
        
        # step 2 : 결과 이후 추가 상담 의사 확인
        elif st.session_state.step == "ASK_ADDITIONAL":
            positive = ["응", "네", "해줘", "좋아", "할게", "계속"]
            negative = ["아니", "괜찮아", "종료", "그만"]
            if any(word in prompt for word in positive):
                # 사용자가 추가 상담 의사 표시 → FOLLOW_UP 단계로 이동
                response = (
                    "좋아요 😊\n"
                    "어떤 상담을 원하시나요?\n"
                    "- **간암 정밀 상담**\n"
                    "- **폐암 정밀 상담**\n\n"
                    "또는 생활습관을 바꿨을 때의 영향도 질문할 수 있어요."
                )
                st.session_state.step = "FOLLOW_UP"

            elif any(word in prompt for word in negative):
                response = (
                    "상담을 종료합니다. 언제든 다시 이용해 주세요 🙂\n\n"
                    "👉 새 상담을 원하시면 왼쪽의 **상담 초기화** 버튼을 눌러주세요."
                )
                st.session_state.step = "FINISHED"
                
            else:
                # 의도가 불명확하면 종료하지 않고 다시 안내
                response = (
                    "추가 상담을 도와드릴 수 있어요.\n"
                    "**간암**, **폐암** 중 원하시는 상담을 말씀해 주세요.\n"
                    "또는 '종료'라고 입력하셔도 됩니다."
                )

        # step 3 : 추가 질문(FOLLOW_UP) 대기 및 분기 단계
        elif st.session_state.step == "FOLLOW_UP":
            # 언제든 종료 가능
            if any(word in prompt for word in ["종료", "그만", "괜찮아"]):
                response = (
                    "상담을 종료합니다. 언제든 다시 이용해 주세요 🙂\n\n"
                    "👉 새 상담을 원하시면 왼쪽의 **상담 초기화** 버튼을 눌러주세요."
                )
                st.session_state.step = "FINISHED"

            # 간암 정밀 상담
            elif "간암" in prompt:
                response = (
                    "간암 정밀 상담을 시작할게요.\n\n"
                    "최근 간염, 간경화, 지방간 진단을 받은 적이 있나요?"
                )

            # 폐암 정밀 상담
            elif "폐암" in prompt:
                response = (
                    "폐암 정밀 상담을 시작할게요.\n\n"
                    "현재 흡연 중이신가요? 또는 과거 흡연 이력이 있나요?"
                )

            # 그 외
            else:
                response = (
                    "추가 상담을 도와드릴 수 있어요.\n\n"
                    "**간암**, **폐암** 중 하나를 선택하시거나\n"
                    "'종료'라고 입력하시면 상담을 마칠 수 있어요."
                )


        # step 4 : 상담 종료 단계
        elif st.session_state.step == "FINISHED":
            response = (
                "상담을 종료합니다. 언제든 다시 이용해 주세요 🙂\n\n"
                "👉 새 상담을 원하시면 왼쪽의 **상담 초기화** 버튼을 눌러주세요."
            )


        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})