import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

filename = "C:\py_project\module_project\The_Cancer_data_1500_V2.csv"

df = pd.read_csv(filename)
print(f" {filename} 로드 ")


inven = ['Age', 'Gender', 'BMI', 'Smoking', 'Alcohol', 'Family_History', 'PhysicalActivity']
target = 'Diagnosis'

df = df[inven + [target]]


sclaer_cols = ['Family_History']

print("\n 0보다 크면 1로 변환")
for col in sclaer_cols:
    # 0보다 크면 1, 아니면 0
    df[col] = df[col].apply(lambda x: 1 if x > 0 else 0)

# 확인용
df.to_csv("processed_data_check.csv", index=False)

# 4. 모델 학습
X = df[inven ]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 성능 평가 및 저장
acc = accuracy_score(y_test, model.predict(X_test))
print(f" 정확도: {acc:.2%}")

joblib.dump(model, "sub1_cancer_model.pkl")

print(" 모델 저장 완료: sub1_cancer_model.pkl")