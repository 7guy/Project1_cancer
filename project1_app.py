import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

# 1. 로직 파일 연동
from project1_logic import (
    gpt_extraction, 
    generate_ask_question, 
    interpret_result_with_gpt, 
    predict_cancer_risk,       
    simulate_lifestyle_change, 
    detect_simulation_intent,  
    normalize_keys
)

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI 암 진단 에이전트", layout="wide")
st.title("🩺 AI 암 발병 위험도 판단 서비스")

# --- 세션 초기화 ---
if 'messages' not in st.session_state: st.session_state.messages = []
if 'context' not in st.session_state: st.session_state.context = "일반"
if 'last_risk_score' not in st.session_state: st.session_state.last_risk_score = None 

# 데이터 저장소
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = {
        "Age": None, "Gender": None, "BMI": None, "Smoking": None, 
        "Alcohol": None, "Family_History": None, "PhysicalActivity": None
    }

# --- 사이드바 ---
with st.sidebar:
    st.header(f"현재 모드: [{st.session_state.context}암]")
    st.divider()
    st.subheader("📋 기초 정보 수집 현황")
    
    REQUIRED_KEYS = ["Age", "Gender", "BMI", "Smoking", "Alcohol", "Family_History", "PhysicalActivity"]
    
    for key in REQUIRED_KEYS:
        val = st.session_state.collected_data.get(key)
        icon = "✅" if val is not None else "⬜"
        st.write(f"{icon} {key}")
        
    if st.button("🔄 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 대화 내용 출력 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 사용자 입력 처리 ---
if prompt := st.chat_input("내용을 입력하세요..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ""
        
        # [Step 0] 모드 자동 전환
        if "폐암" in prompt:
            st.session_state.context = "폐"
            st.info("🫁 **폐암 정밀 분석 모드**로 전환합니다.")
        elif "간암" in prompt:
            st.session_state.context = "간"
            st.info("🍷 **간암 정밀 분석 모드**로 전환합니다.")

        # [Step 1] 시뮬레이션 의도 파악 (최우선 순위)
        sim_intent = detect_simulation_intent(prompt, client)
        
        if sim_intent.get("type") == "simulation":
            # 1. 기준점 계산 (저장하지 않고 계산만 함 -> 오염 방지)
            current_prob = predict_cancer_risk(st.session_state.context, st.session_state.collected_data)
            
            # 2. 미래 예측
            cur, new, diff = simulate_lifestyle_change(
                st.session_state.context,
                st.session_state.collected_data,
                sim_intent.get("changes"),
                current_prob 
            )
            
            diff_text = "증가" if diff > 0 else "감소" # 0보다 크면 증가
            changes_str = ", ".join([f"{k}: {v}" for k, v in sim_intent.get('changes').items()])
            
            # 🚨 [요청하신 대로 문장형 출력 수정]
            response = f"""
            📊 **시뮬레이션 결과 ({st.session_state.context}암)**
            
            가정하신 상황(**{changes_str}**)을 적용하면:
            
            암 발병률은 **{new:.1f}%**가 되고,
            현재보다 암 발병 확률이 **{abs(diff):.1f}% 포인트 {diff_text}**합니다.
            """

        else:
            # [Step 2] 일반 정보 추출 및 저장
            extracted = gpt_extraction(st.session_state.messages, prompt, client)
            if extracted:
                normalized = normalize_keys(extracted)
                for k, v in normalized.items():
                    if v is not None:
                        st.session_state.collected_data[k] = v

            # [Step 3] 필수 정보 체크
            missing = [k for k in REQUIRED_KEYS if st.session_state.collected_data.get(k) is None]

            if missing:
                response = generate_ask_question(st.session_state.collected_data, missing)
            
            else:
                # [Step 4] 일반 예측 결과
                current_prob = predict_cancer_risk(st.session_state.context, st.session_state.collected_data)
                
                # 점수 저장
                st.session_state.last_risk_score = current_prob
                
                gpt_msg, score = interpret_result_with_gpt(current_prob, client, st.session_state.context + "암")
                response = gpt_msg
                response += "\n\n--- \n💡 **Tip:** *'50살에 담배 피면?'* 처럼 물어보시면 미래를 예측해 드려요."

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})