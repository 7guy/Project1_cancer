import streamlit as st
import os
from openai import OpenAI
from dotenv import load_dotenv

from project_logic import (
    predict_cancer_risk, 
    get_missing_info_question,
    gpt_extraction,
    interpret_result_with_gpt,
    predict_scenario,
    detect_simulation_intent 
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Cancer Care", layout="wide")
st.title("🩺 맞춤형 AI 암 위험도 정밀 분석")

# --- 세션 상태 ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 통합 암 발병 위험도 분석을 위해 건강 정보를 알려주세요."}]
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = {}
if 'current_cancer' not in st.session_state:
    st.session_state.current_cancer = "TOTAL"

# --- 사이드바 ---
with st.sidebar:
    st.header("📊 데이터 현황")
    st.write(f"**현재 타겟:** {st.session_state.current_cancer}")
    st.divider()
    if st.session_state.collected_data:
        for k, v in st.session_state.collected_data.items():
            if v is not None: st.write(f"✅ {k}: {v}")
    
    if st.button("🔄 상담 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 대화 화면 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 메인 로직 ---
if prompt := st.chat_input("내용을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        
        # 1. 시뮬레이션 의도 파악 (최우선 순위)
        sim_intent = detect_simulation_intent(prompt, client)
        
        # 2. 모드 전환 체크 (실제로 암종이 바뀔 때만 안내 출력)
        new_cancer_type = None
        if "폐암" in prompt: new_cancer_type = "LUNG"
        elif "간암" in prompt: new_cancer_type = "LIVER"
        
        # 암종이 명시되었고, 현재와 다를 때만 전환 안내
        if new_cancer_type and new_cancer_type != st.session_state.current_cancer:
            st.session_state.current_cancer = new_cancer_type
            st.info(f"🔄 **{new_cancer_type} 정밀 분석 모드**로 전환합니다.")

        # 3. 시나리오 분기 처리
        if sim_intent.get("type") == "simulation":
            # [시뮬레이션 로직] - 기존 데이터 유지하며 가상 계산
            current_score = predict_cancer_risk(st.session_state.collected_data, st.session_state.current_cancer)
            future_score = predict_scenario(
                st.session_state.collected_data, 
                sim_intent.get("changes"), 
                st.session_state.current_cancer
            )
            
            diff = future_score - current_score
            diff_text = "증가" if diff > 0 else "감소"
            changes_str = ", ".join([f"{k}: {v}" for k, v in sim_intent.get('changes').items()])
            
            response = f"""
📊 **시뮬레이션 결과 ({st.session_state.current_cancer})**

가정(**{changes_str}**)을 적용하면:

- 현재 위험도: **{current_score}%**
- 예상 위험도: **{future_score}%**

👉 결과적으로 암 발병 확률이 **{abs(diff):.1f}% 포인트 {diff_text}**합니다.
"""

        else:
            # [일반 정보 수집 로직]
            new_data = gpt_extraction(st.session_state.messages, prompt, client)
            if new_data:
                for key, value in new_data.items():
                    if value is not None and value != "":
                        st.session_state.collected_data[key] = value
                
                # BMI 계산 생략 (기존 코드와 동일)

            missing_q = get_missing_info_question(st.session_state.collected_data, st.session_state.current_cancer)
            
            if missing_q:
                response = missing_q
            else:
                score = predict_cancer_risk(st.session_state.collected_data, st.session_state.current_cancer)
                analysis = interpret_result_with_gpt(prob=score, cancer_type=st.session_state.current_cancer, client=client)
                
                response = f"### 📊 {st.session_state.current_cancer} 분석 결과\n\n"
                response += f"**현재 예측 위험도: {score}%**\n\n"
                response += analysis
                response += "\n\n--- \n💡 **Tip:** *'담배 끊으면?'* 처럼 가정해서 물어보세요."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})