import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

from project1_logic import (
    predict_cancer_risk, 
    get_missing_info_question,
    gpt_extraction,
    interpret_result_with_gpt,
    predict_scenario,
    detect_simulation_intent 
)

# ===============================
# 모델 표시용 (UI 전용)
# ===============================
model_label_map = {
    "TOTAL": "암 위험도 모델",
    "LUNG": "폐암 위험도 모델",
    "LIVER": "간암 위험도 모델"
}

# ===============================
# 환경 설정
# ===============================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Cancer Care", layout="wide")
st.title("🩺 맞춤형 AI 암 위험도 정밀 분석")

# ===============================
# 세션 상태
# ===============================
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 통합 암 발병 위험도 분석을 위해 건강 정보를 알려주세요."}
    ]

if 'collected_data' not in st.session_state:
    st.session_state.collected_data = {}

if 'current_cancer' not in st.session_state:
    st.session_state.current_cancer = "TOTAL"

if 'data_updated' not in st.session_state:
    st.session_state.data_updated = False

# 🔑 모델 전환 직후 플래그
if 'just_switched_model' not in st.session_state:
    st.session_state.just_switched_model = False


# ===============================
# 사이드바
# ===============================
with st.sidebar:
    st.header("📊 데이터 현황")
    st.write(f"**현재 타겟:** {st.session_state.current_cancer}")
    st.write(f"🧠 **현재 모델:** {model_label_map.get(st.session_state.current_cancer)}")
    st.divider()

    if st.session_state.collected_data:
        for k, v in st.session_state.collected_data.items():
            if v is not None and v != "":
                st.write(f"✅ {k}: {v}")

    if st.button("🔄 상담 초기화"):
        st.session_state.clear()
        st.rerun()


# ===============================
# 채팅 화면
# ===============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ===============================
# 메인 로직
# ===============================
if prompt := st.chat_input("내용을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""

        # 1. 시뮬레이션 의도 파악
        sim_intent = detect_simulation_intent(prompt, client)

        # 2. 암종 전환 (❌ rerun 금지)
        new_cancer_type = None
        if "폐암" in prompt:
            new_cancer_type = "LUNG"
        elif "간암" in prompt:
            new_cancer_type = "LIVER"

        if new_cancer_type and new_cancer_type != st.session_state.current_cancer:
            st.session_state.current_cancer = new_cancer_type
            st.session_state.just_switched_model = True
            st.info(f"🔄 **{new_cancer_type} 정밀 분석 모드**로 전환합니다.")

        # 3. 시뮬레이션
        if sim_intent.get("type") == "simulation":
            current_score = predict_cancer_risk(
                st.session_state.collected_data,
                st.session_state.current_cancer
            )

            future_score = predict_scenario(
                st.session_state.collected_data,
                sim_intent.get("changes"),
                st.session_state.current_cancer
            )

            diff = future_score - current_score
            diff_text = "증가" if diff > 0 else "감소"

            changes_str = ", ".join(
                [f"{k}: {v}" for k, v in sim_intent.get("changes", {}).items()]
            )

            response = f"""
📊 **시뮬레이션 결과 ({st.session_state.current_cancer})**

가정(**{changes_str}**)을 적용하면:

- 현재 위험도: **{current_score}%**
- 예상 위험도: **{future_score}%**

👉 결과적으로 암 발병 확률이 **{abs(diff):.1f}%p {diff_text}**합니다.
"""

        # 4. 일반 정보 수집
        else:
            new_data = gpt_extraction(
                st.session_state.messages,
                prompt,
                client
            )

            if new_data:
                for key, value in new_data.items():
                    if value is not None and value != "":
                        st.session_state.collected_data[key] = value
                        st.session_state.data_updated = True

            missing_q = get_missing_info_question(
                st.session_state.collected_data,
                st.session_state.current_cancer
            )

            if missing_q:
                response = missing_q
            else:
                score = predict_cancer_risk(
                    st.session_state.collected_data,
                    st.session_state.current_cancer
                )

                analysis = interpret_result_with_gpt(
                    prob=score,
                    cancer_type=st.session_state.current_cancer,
                    client=client
                )

                response = f"### 📊 {st.session_state.current_cancer} 분석 결과\n\n"
                response += f"**현재 예측 위험도: {score}%**\n\n"
                response += analysis
                response += (
                    "\n\n💡 **Tip:** 지금 상태에서 *'폐암 모드로 바꿔줘'* 혹은 "
                    "*'간암 결과는 어때?'* 라고 물어보세요."
                )

        st.markdown(response)
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

        # 🔥 질문 출력 후에만 rerun
        if st.session_state.just_switched_model:
            st.session_state.just_switched_model = False
            st.rerun()


# ===============================
# 사이드바 즉시 반영
# ===============================
if st.session_state.data_updated:
    st.session_state.data_updated = False
    st.rerun()
