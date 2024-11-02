# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 22:59:51 2024

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer

# 데이터 로드 및 전처리
df = pd.read_excel("train.xlsx")  # train.xlsx 파일 로드
df_clean = df.dropna()
df_encoded_clean = pd.get_dummies(df_clean.drop('NObeyesdad', axis=1))
y = df_clean['NObeyesdad']

# 10-폴드 교차 검증 설정
kf = KFold(n_splits=10, shuffle=True, random_state=42)

# 기본 모델 설정 (랜덤 포레스트와 그라디언트 부스팅)
rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=1, min_samples_split=2, random_state=42)
gb_model = GradientBoostingClassifier(learning_rate = 0.05, max_depth = 7, min_samples_leaf = 1, min_samples_split=5, n_estimators=200, subsample = 0.8,random_state=42)

# F1 스코어를 위한 커스텀 스코러 설정
f1_scorer = make_scorer(f1_score, average='weighted')

# 교차 검증을 사용하여 스태킹 평가
f1_scores = []
accuracy_scores = []
meta_model = LogisticRegression()

for train_index, test_index in kf.split(df_encoded_clean):
    # 훈련 및 테스트 데이터 분할
    X_train, X_test = df_encoded_clean.iloc[train_index], df_encoded_clean.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    # 개별 모델 학습 및 예측
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    rf_pred = rf_model.predict_proba(X_test)  # 확률 예측 사용
    
    gb_pred = gb_model.predict_proba(X_test)
    
    
    # 예측 결과 결합 (각 모델의 확률 예측을 스택)
    stacked_features = np.hstack((rf_pred, gb_pred))
    
    # 메타 모델 학습
    meta_model.fit(stacked_features, y_test)
    
    # 최종 예측 수행
    final_pred = meta_model.predict(stacked_features)
    
    # 평가
    f1_scores.append(f1_score(y_test, final_pred, average='weighted'))
    accuracy_scores.append((final_pred == y_test).mean())

print(f'교차 검증 F1 스코어 평균: {np.mean(f1_scores)}')
print(f'교차 검증 정확도 평균: {np.mean(accuracy_scores)}')

# 테스트 데이터 로드 및 전처리
X_test = pd.read_excel("X_test.xlsx")

# df_encoded_clean의 열과 동일하게 인코딩
X_test_encoded = pd.get_dummies(X_test)

# df_encoded_clean과 동일한 열을 유지하기 위해 재색인
X_test_encoded = X_test_encoded.reindex(columns=df_encoded_clean.columns, fill_value=0)

# 랜덤 포레스트 및 그라디언트 부스팅 모델 학습 (전체 데이터 사용)
rf_model.fit(df_encoded_clean, y)
gb_model.fit(df_encoded_clean, y)

# 개별 모델 예측 확률
rf_pred_test = rf_model.predict_proba(X_test_encoded)
gb_pred_test = gb_model.predict_proba(X_test_encoded)

# 스택된 특징 결합
stacked_features_test = np.hstack((rf_pred_test, gb_pred_test))

# 메타 모델 학습 (전체 데이터 사용)
meta_model.fit(np.hstack((rf_model.predict_proba(df_encoded_clean), gb_model.predict_proba(df_encoded_clean))), y)

# 최종 예측 수행
final_pred_test = meta_model.predict(stacked_features_test)

# 예측 결과를 데이터프레임으로 저장
output_df = pd.DataFrame({'Predicted': final_pred_test})


# 예측 결과를 엑셀 파일로 저장
output_df.to_excel('result4.xlsx', index=False)

