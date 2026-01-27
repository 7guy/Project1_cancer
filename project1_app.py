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

# --- 1. 세션 상태 관리 (흐름 제어 핵심) ---
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 건강 정보를 입력하시면 먼저 통합 암 발병 위험도를 분석해 드립니다."}]
if 'collected_data' not in st.session_state:
    st.session_state.collected_data = {}
if 'step' not in st.session_state:
    st.session_state.step = "TOTAL_COLLECT"  # 진행 단계
if 'cancer_type' not in st.session_state:
    st.session_state.cancer_type = "TOTAL"

# --- 2. 사이드바 (정보 대시보드 - 유지) ---
with st.sidebar:
    st.header("📊 현재 데이터 수집 현황")
    st.caption(f"현재 분석 단계: {st.session_state.step}")
    st.divider()
    
    if not st.session_state.collected_data:
        st.info("입력된 정보가 없습니다.")
    else:
        # 입력된 정보들을 보기 좋게 나열
        for k, v in st.session_state.collected_data.items():
            if v is not None:
                st.write(f"**{k}**: {v}")
    
    st.divider()
    if st.button("🔄 상담 초기화"):
        st.session_state.clear()
        st.rerun()

# --- 3. 대화창 렌더링 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- [중요] 세션 상태에 '현재 선택된 암' 저장 변수 추가 ---
if 'current_cancer' not in st.session_state:
    st.session_state.current_cancer = "TOTAL"
if 'finished_cancers' not in st.session_state:
    st.session_state.finished_cancers = [] # 이미 분석 완료된 암 목록

# --- 메인 대화 로직 ---
if prompt := st.chat_input("원하시는 단계를 입력하세요 (예: 폐암 분석, 금연하면?, 종료)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        # 1. 의도 파악 (질문자님의 classify_intent 또는 키워드 매칭)
        # 1-A. 시나리오 질문 (금연한다면 등)
        if any(word in prompt for word in ["한다면", "하면", "끊으면", "줄이면"]):
            # 시뮬레이션 로직 수행
            # (가정: Smoking을 0으로 시뮬레이션)
            sim_score = predict_scenario(st.session_state.collected_data, "Smoking", 0, st.session_state.current_cancer)
            response = f"💡 만약 금연하신다면, 현재 분석 중인 **{st.session_state.current_cancer}** 위험도가 **{sim_score}%**로 변화할 것으로 예측됩니다."
            response += "\n\n다른 가정을 해보시겠어요, 아니면 다른 암 정밀 분석을 해보시겠어요?"

        # 1-B. 다른 암 분석 선택 (간암/폐암)
        elif any(word in prompt for word in ["간암", "폐암", "정밀"]):
            target = "LIVER" if "간" in prompt else "LUNG"
            st.session_state.current_cancer = target
            
            # 정보 추출 및 추가 질문 확인
            new_data = gpt_extraction(st.session_state.messages, prompt, client)
            st.session_state.collected_data.update(new_data)
            
            missing_q = get_missing_info_question(st.session_state.collected_data, target)
            if missing_q:
                response = f"### 🧪 {target} 정밀 분석을 위해 추가 정보가 필요합니다.\n\n{missing_q}"
            else:
                score = predict_cancer_risk(st.session_state.collected_data, target)
                analysis = interpret_result_with_gpt(prob=score, cancer_type=target, client=client)
                response = f"### 🏁 {target} 정밀 분석 결과\n\n{analysis}\n\n**위험도: {score}%**"
                if target not in st.session_state.finished_cancers:
                    st.session_state.finished_cancers.append(target)
                
                # 다음 선택지 제공
                remaining = [c for c in ["LIVER", "LUNG"] if c not in st.session_state.finished_cancers]
                response += f"\n\n---\n이제 어떤 것을 도와드릴까요?\n"
                if remaining: response += f"- **{remaining[0]}** 정밀 분석\n"
                response += f"- 현재 결과에서 **시뮬레이션** (예: 술을 끊는다면?)\n- 상담 **종료**"

        # 1-C. 종료
        elif "종료" in prompt or "끝" in prompt:
            response = "모든 상담이 완료되었습니다. 건강한 하루 되시길 바랍니다! 분석 결과 요약이 필요하시면 말씀해 주세요."

        # 1-D. 그 외 (데이터 수집 중인 경우)
        else:
            new_data = gpt_extraction(st.session_state.messages, prompt, client)
            st.session_state.collected_data.update(new_data)
            
            # 현재 분석 중인 암 종류에 맞춰 질문
            missing_q = get_missing_info_question(st.session_state.collected_data, st.session_state.current_cancer)
            if missing_q:
                response = missing_q
            else:
                # 데이터가 다 모이면 즉시 분석 결과 출력
                score = predict_cancer_risk(st.session_state.collected_data, st.session_state.current_cancer)
                analysis = interpret_result_with_gpt(prob=score, cancer_type=st.session_state.current_cancer, client=client)
                response = f"### 📊 {st.session_state.current_cancer} 분석 결과\n\n{analysis}\n\n**위험도: {score}%**"
                
                # 분석 완료 후 메뉴 제안
                response += "\n\n---\n**[다음 선택지]**\n1. 다른 암(간암/폐암) 정밀 분석\n2. 건강 수치 시뮬레이션 (예: 담배를 끊는다면?)\n3. 상담 종료"

        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})