import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# 로직 파일 연동
from project1_logic import (
    gpt_extraction, 
    generate_ask_question, 
    interpret_result_with_gpt, 
    predict_cancer_risk,       
    simulate_lifestyle_change, 
    detect_simulation_intent,  
    normalize_keys,
    determine_automatic_context # 라우터 함수
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI 통합 암 진단 에이전트", layout="centered") # 중앙 정렬

# --- [상단] 헤더 및 상태 표시 ---
st.title("🩺 AI 헬스케어 에이전트")
st.caption("버튼 없이 대화로만 진단하고 시뮬레이션까지 가능한 AI입니다.")

# 세션 초기화
if 'messages' not in st.session_state: st.session_state.messages = []
if 'collected_data' not in st.session_state: st.session_state.collected_data = {}
if 'context' not in st.session_state: st.session_state.context = "일반"
if 'last_risk_score' not in st.session_state: st.session_state.last_risk_score = None 

# 현재 상태를 작게 보여줌 (사이드바 대신 상단에 배치해도 됨, 혹은 사이드바는 정보창으로만 사용)
with st.sidebar:
    st.header(f"현재 모드: {st.session_state.context}암 분석 중")
    st.info("💡 팁: '폐암 봐줘', '간암은 어때?' 라고 말하면 모드가 바뀝니다.")
    
    st.subheader("📋 현재 파악된 정보")
    if st.session_state.collected_data:
        st.json(st.session_state.collected_data)
    else:
        st.write("아직 정보가 없습니다.")
        
    if st.button("🗑️ 기억 지우기 (초기화)"):
        st.session_state.clear()
        st.rerun()

# --- [메인] 대화창 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- [핵심] 사용자 입력 처리 ---
if prompt := st.chat_input("예: 24살 여자고 운동 매일 해. 폐암 확률은?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""

        # 1. [라우팅] 사용자의 말에서 모드(Context) 자동 결정
        new_context = determine_automatic_context(prompt, st.session_state.context)
        if new_context != st.session_state.context:
            st.session_state.context = new_context
            st.toast(f"🔄 {new_context}암 분석 모드로 전환되었습니다!") # 토스트 메시지로 세련되게 알림

        # 2. [의도 파악] 시뮬레이션인가?
        sim_intent = detect_simulation_intent(prompt, client)
        
        if sim_intent.get("type") == "simulation":
            # 시뮬레이션 로직
            if st.session_state.last_risk_score is None:
                # 점수가 없으면 현재 데이터로 즉석 계산해서 기준점 잡기
                st.session_state.last_risk_score = predict_cancer_risk(st.session_state.context, st.session_state.collected_data)
            
            cur, new, diff = simulate_lifestyle_change(
                st.session_state.context,
                st.session_state.collected_data,
                sim_intent.get("changes"),
                st.session_state.last_risk_score
            )
            
            diff_text = "감소" if diff < 0 else "증가"
            changes_str = ", ".join([f"{k}: {v}" for k, v in sim_intent.get('changes').items()])
            
            response = f"""
            📊 **가상 시뮬레이션 ({st.session_state.context}암)**
            
            조건 변경 (**{changes_str}**) 결과입니다:
            - 현재 위험도: **{cur:.1f}%**
            - 🔮 예상 위험도: **{new:.1f}%**
            
            약 **{abs(diff):.1f}% 포인트 {diff_text}**할 것으로 보입니다.
            """

        else:
            # 3. [정보 수집 및 예측]
            # (1) 정보 추출 및 저장 (기존 정보에 덮어쓰기 -> 누적됨)
            extracted = gpt_extraction(st.session_state.messages, prompt, client)
            if extracted:
                normalized = normalize_keys(extracted)
                st.session_state.collected_data.update(normalized)

            # (2) 필수 정보 체크
            if st.session_state.context == "일반": check = ["Age", "Gender", "BMI", "Smoking", "Alcohol"]
            elif st.session_state.context == "폐": check = ["Age", "Gender", "Smoking"]
            else: check = ["Age", "Gender", "Alcohol"]

            missing = [k for k in check if st.session_state.collected_data.get(k) is None]

            if missing:
                # 정보가 부족하면 질문
                response = generate_ask_question(st.session_state.collected_data, missing)
            else:
                # 정보가 충분하면 예측
                with st.spinner(f"{st.session_state.context}암 분석 중..."):
                    prob = predict_cancer_risk(st.session_state.context, st.session_state.collected_data)
                    gpt_msg, score = interpret_result_with_gpt(prob, client, st.session_state.context + "암")
                    st.session_state.last_risk_score = score
                    response = gpt_msg
                    response += "\n\n--- \n💡 *'담배 끊으면?'*, *'간암은 어때?'* 처럼 자유롭게 말씀해주세요."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})