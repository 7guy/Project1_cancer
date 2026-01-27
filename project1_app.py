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
        
        # 0. 모드 전환 체크 (항상 우선)
        if "폐암" in prompt:
            st.session_state.current_cancer = "LUNG"
            st.info("🫁 **폐암 정밀 분석 모드**로 전환합니다.")
        elif "간암" in prompt:
            st.session_state.current_cancer = "LIVER"
            st.info("🍷 **간암 정밀 분석 모드**로 전환합니다.")

        # =========================================================
        # 🔥 1. 시뮬레이션 의도 파악 (데이터 저장 전에 수행!)
        # =========================================================
        sim_intent = detect_simulation_intent(prompt, client)
        
        if sim_intent.get("type") == "simulation":
            # 1. 현재 상태 점수 계산 (기준점)
            current_score = predict_cancer_risk(st.session_state.collected_data, st.session_state.current_cancer)
            
            # 2. 미래 점수 계산
            future_score = predict_scenario(
                st.session_state.collected_data, 
                sim_intent.get("changes"), 
                st.session_state.current_cancer
            )
            
            # 3. 차이 계산 및 문장 생성 (원하시던 포맷!)
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
            # =====================================================
            # 🔥 2. 시뮬레이션이 아닐 때만 정보를 저장 (오염 방지)
            # =====================================================
            new_data = gpt_extraction(st.session_state.messages, prompt, client)
            if new_data:
                for key, value in new_data.items():
                    if value is not None and value != "":
                        st.session_state.collected_data[key] = value
                
                # BMI 자동 계산
                if 'Height' in st.session_state.collected_data and 'Weight' in st.session_state.collected_data:
                    h = st.session_state.collected_data['Height']
                    w = st.session_state.collected_data['Weight']
                    st.session_state.collected_data['BMI'] = round(w / ((h/100)**2), 1)

            # 3. 필수 정보 체크
            missing_q = get_missing_info_question(st.session_state.collected_data, st.session_state.current_cancer)
            
            if missing_q:
                response = missing_q
            else:
                # 4. 최종 결과 (모든 정보 수집 완료)
                score = predict_cancer_risk(st.session_state.collected_data, st.session_state.current_cancer)
                analysis = interpret_result_with_gpt(prob=score, cancer_type=st.session_state.current_cancer, client=client)
                
                response = f"### 📊 {st.session_state.current_cancer} 분석 결과\n\n"
                response += f"**현재 예측 위험도: {score}%**\n\n"
                response += analysis
                response += "\n\n--- \n💡 **Tip:** *'담배 끊으면?'*, *'50살이 되면?'* 처럼 가정해서 물어보세요."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})