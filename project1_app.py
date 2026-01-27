import streamlit as st
import os
import traceback
from openai import OpenAI
from dotenv import load_dotenv

# 질문자님의 logic.py 함수들 임포트
from project1_logic import (
    predict_cancer_risk, 
    get_missing_info_question,
    gpt_extraction,          # 보완한 GPT 추출 함수
    interpret_result_with_gpt, # gpt-4o로 업그레이드한 해석 함수
    predict_scenario
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI Cancer Care", layout="wide")
st.title("🩺 맞춤형 AI 암 위험도 정밀 분석")

# --- 세션 상태 초기화 ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 통합 암 발병 위험도 분석을 위해 건강 정보를 알려주세요."}]
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = {}
if 'step' not in st.session_state:
    st.session_state.step = "TOTAL_COLLECT"
if 'current_cancer' not in st.session_state:
    st.session_state.current_cancer = "TOTAL"

# --- 사이드바: 대시보드 유지 ---
with st.sidebar:
    st.header("📊 데이터 현황")
    st.write(f"**진행 단계:** {st.session_state.step}")
    st.write(f"**현재 타겟:** {st.session_state.current_cancer}")
    st.divider()
    if st.session_state.collected_data:
        for k, v in st.session_state.collected_data.items():
            if v is not None: st.write(f"✅ {k}: {v}")
    else:
        st.info("수집된 정보 없음")
    
    if st.button("🔄 상담 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 대화 화면 렌더링 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 메인 대화 로직 ---
if prompt := st.chat_input("질문에 답하거나 궁금한 점을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. GPT 정보 추출 (현재 메시지만 확인)
        # 1. 정보 추출 및 저장 (질문자님이 제시한 코드)
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

        # 2. 누락 정보 확인 (질문자님이 제시한 코드)
        missing_q = get_missing_info_question(st.session_state.collected_data, st.session_state.current_cancer)
        
        if missing_q:
            response = missing_q # 아직 정보가 부족하면 질문만 할당
        else:
            # ---------------------------------------------------------
            # 3. 모든 정보가 있을 때만 진입하는 [복잡한 로직 구간]
            # ---------------------------------------------------------
            
            # (A) 시나리오 질문 (금연한다면? 등)
            if any(word in prompt for word in ["한다면", "하면", "끊으면"]):
                score = predict_scenario(st.session_state.collected_data, "Smoking", 0, st.session_state.current_cancer)
                analysis = interpret_result_with_gpt(prob=score, cancer_type=st.session_state.current_cancer, client=client)
                response = f"💡 시뮬레이션 결과입니다:\n{analysis}\n\n**(변화된 위험도: {score}%)**"
            
            # (B) 특정암 정밀 분석 선택
            elif any(word in prompt for word in ["간암", "폐암"]):
                target = "LIVER" if "간" in prompt else "LUNG"
                st.session_state.current_cancer = target
                # 타겟이 바뀌었으니 바로 다시 체크하기 위해 rerun 하거나 질문 던짐
                missing_q_new = get_missing_info_question(st.session_state.collected_data, target)
                if missing_q_new:
                    response = f"### 🧪 {target} 정밀 분석 모드\n{missing_q_new}"
                else:
                    score = predict_cancer_risk(st.session_state.collected_data, target)
                    analysis = interpret_result_with_gpt(prob=score, cancer_type=target, client=client)
                    response = f"### 🏁 {target} 분석 결과\n{analysis}\n\n**(위험도: {score}%)**"

            # (C) 일반 분석 결과 (데이터가 막 다 채워진 시점)
            else:
                score = predict_cancer_risk(st.session_state.collected_data, st.session_state.current_cancer)
                analysis = interpret_result_with_gpt(prob=score, cancer_type=st.session_state.current_cancer, client=client)
                response = f"### 📊 {st.session_state.current_cancer} 분석 결과\n\n{analysis}\n\n**(위험도: {score}%)**"
                response += "\n\n다른 암 분석이나 '금연 시나리오'가 궁금하시면 말씀해주세요!"

        # 최종 응답 출력 및 저장
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})