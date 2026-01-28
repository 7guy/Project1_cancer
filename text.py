import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# ==========================================
# 0. 한글 폰트 설정 (시각화 깨짐 방지)
# ==========================================
import platform
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif system_name == 'Darwin': # Mac
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# ==========================================
# 파일 로드 및 전처리 (기존 코드 유지)
# ==========================================

# 파일 경로 (본인 환경에 맞게 수정)
filename = r"C:\py_project\module_project\processed_data_check.csv"

# 테스트를 위해 임시로 try-except 유지 (실제 사용 시 경로 확인 필수)
try:
    df = pd.read_csv(filename)
    print(f"✅ {filename} 로드 성공")
except FileNotFoundError:
    print(f"❌ 파일을 찾을 수 없습니다: {filename}")
    # 코드가 멈추지 않게 임시 데이터 생성 (사용자 환경에서는 이 부분 무시됨)
    # 실제 파일이 있다면 아래 4줄은 실행되지 않음
    data = {
        'Age': np.random.randint(20, 80, 1000),
        'Gender': np.random.randint(0, 2, 1000),
        'BMI': np.random.rand(1000) * 30,
        'Smoking': np.random.randint(0, 2, 1000),
        'Alcohol': np.random.randint(0, 2, 1000),
        'Family_History': np.random.randint(0, 2, 1000),
        'PhysicalActivity': np.random.randint(0, 10, 1000),
        'Diagnosis': np.random.randint(0, 2, 1000)
    }
    df = pd.DataFrame(data)

# 1. 사용할 컬럼 선택
inven = ['Age', 'Gender', 'BMI', 'Smoking', 'Alcohol', 'Family_History', 'PhysicalActivity']
target = 'Diagnosis'
df = df[inven + [target]]

# 2. 데이터 전처리
sclaer_cols = ['Family_History']
for col in sclaer_cols:
    df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

# 3. 데이터 분할
X = df[inven]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 학습
print("\n🏋️ 모델 학습 시작 (class_weight='balanced' 적용)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# 5. 성능 평가 및 시각화 (수정된 부분!)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# (1) 데이터 준비
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report_dict = classification_report(y_test, y_pred, output_dict=True)

# 분류 리포트 데이터프레임 변환 (히트맵용)
# 0: 정상, 1: 환자로 매핑
metrics_df = pd.DataFrame([
    [report_dict['0']['precision'], report_dict['0']['recall'], report_dict['0']['f1-score']],
    [report_dict['1']['precision'], report_dict['1']['recall'], report_dict['1']['f1-score']]
], columns=['precision', 'recall', 'f1-score'], index=['정상', '환자'])

# (2) 시각화 그리기
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('간암 위험도 예측 모델 성능평가', fontsize=20, fontweight='bold', color='red', y=0.98)

# --- 왼쪽 그래프: Classification Report (Heatmap 스타일) ---
sns.heatmap(metrics_df, annot=True, fmt='.2f', cmap='Reds', cbar=False, 
            ax=axes[0], annot_kws={"size": 14}, vmin=0, vmax=1)
axes[0].set_title('Classification Report', fontsize=14)
axes[0].tick_params(axis='y', rotation=0, labelsize=12)
axes[0].tick_params(axis='x', labelsize=12)

# 정확도 텍스트 추가 (이미지 왼쪽 하단처럼)
axes[0].text(0.5, -0.2, f"accuracy : {acc:.2f}", 
             transform=axes[0].transAxes, 
             fontsize=24, fontweight='bold', ha='center')

# --- 오른쪽 그래프: Confusion Matrix ---
# 레이블 설정
labels = ['Predicted Healthy (0)', 'Predicted Cancer (1)']
y_labels = ['Actual Healthy (0)', 'Actual Cancer (1)']

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=axes[1],
            annot_kws={"size": 14}, xticklabels=labels, yticklabels=y_labels)

axes[1].set_title('Confusion Matrix: Liver Cancer Prediction', fontsize=14)
axes[1].set_ylabel('Actual Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout(rect=[0, 0.03, 1, 0.90]) # 타이틀 공간 확보
plt.show()

# 텍스트 결과도 콘솔에 출력 (확인용)
print("\n" + "="*40)
print(f"📊 정확도 (Accuracy): {acc:.2%}")
print("="*40)
print(classification_report(y_test, y_pred))

# 6. 모델 저장
joblib.dump(model, "sub1_cancer_model.pkl")
print("\n💾 모델 저장 완료: sub1_cancer_model.pkl")