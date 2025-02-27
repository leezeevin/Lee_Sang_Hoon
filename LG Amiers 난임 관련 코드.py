import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 데이터 로드
df = pd.read_csv('train.csv').drop(columns=['ID'])
test = pd.read_csv('test.csv').drop(columns=['ID'])
sample_submission = pd.read_csv('sample_submission.csv')

count_mapping = {
    "0회": 0, "1회": 1, "2회": 2, "3회": 3, "4회": 4, "5회": 5, "6회 이상": 6
}

# 적용
for col in ['총 시술 횟수', 'IVF 시술 횟수', 'DI 시술 횟수', '총 임신 횟수', 'IVF 임신 횟수', 'DI 임신 횟수', '총 출산 횟수', 'IVF 출산 횟수', 'DI 출산 횟수','클리닉 내 총 시술 횟수']:
    df[col] = df[col].map(count_mapping)
    test[col] = test[col].map(count_mapping)

# 파생 변수 생성
def create_features(data):
    data['총 시술 경험'] = data['총 시술 횟수'] / data['클리닉 내 총 시술 횟수']
    data['ivf 대비 di 비율'] = data['IVF 시술 횟수'] / (data['IVF 시술 횟수'] + data['DI 시술 횟수'])
    data['주된 불임 원인 개수'] = (data['남성 주 불임 원인'] + data['여성 주 불임 원인'] + data['부부 주 불임 원인'])
    data['시술당 임신율'] = data['총 임신 횟수'] / data['총 시술 횟수']
    data['IVF 임신 성공률'] = data['IVF 임신 횟수'] / data['IVF 시술 횟수']
    data['출산 성공률'] = data['총 출산 횟수'] / data['총 임신 횟수']
    data['배아 생존율'] = data['이식된 배아 수'] / data['총 생성 배아 수']
    data['ICSI 배아 비율'] = data['미세주입에서 생성된 배아 수'] / data['총 생성 배아 수']
    data['배아 저장율'] = data['저장된 배아 수'] / data['총 생성 배아 수']
    return data

df = create_features(df)
test = create_features(test)

# IVF와 DI 데이터 분리
df1 = df[df['시술 유형'] == 'IVF']
df2 = df[df['시술 유형'] == 'DI']
test1 = test[test['시술 유형'] == 'IVF']
test2 = test[test['시술 유형'] == 'DI']

# 특정 컬럼의 결측치를 -1로 채우기
fillna_cols = [
    '임신 시도 또는 마지막 임신 경과 연수', '착상 전 유전 검사 사용 여부', 'PGS 시술 여부',
    'PGD 시술 여부', '난자 해동 경과일', '배아 해동 경과일'
]
df[fillna_cols] = df[fillna_cols].fillna(-1)
df1[fillna_cols] = df1[fillna_cols].fillna(-1)
df2[fillna_cols] = df2[fillna_cols].fillna(-1)

# 소수 결측치 Unknown으로 치환
df1['특정 시술 유형'].fillna('Unknown', inplace=True)

#DI 모델 중 필요 없는 열 제거(모두 결측치)
df2 = df2[df2.isnull().sum(axis=1) != 6291]

# 범주형 변수 설정
categorical_columns = [
    '시술 시기 코드', '시술 당시 나이', '시술 유형', '특정 시술 유형', '배란 자극 여부',
    '배란 유도 유형', '단일 배아 이식 여부', '착상 전 유전 검사 사용 여부',
    '착상 전 유전 진단 사용 여부', '남성 주 불임 원인', '남성 부 불임 원인',
    '여성 주 불임 원인', '여성 부 불임 원인', '부부 주 불임 원인', '부부 부 불임 원인',
    '불명확 불임 원인', '불임 원인 - 난관 질환', '불임 원인 - 남성 요인', '불임 원인 - 배란 장애',
    '불임 원인 - 여성 요인', '불임 원인 - 자궁경부 문제', '불임 원인 - 자궁내막증',
    '불임 원인 - 정자 농도', '불임 원인 - 정자 면역학적 요인', '불임 원인 - 정자 운동성',
    '불임 원인 - 정자 형태', '배아 생성 주요 이유', '난자 출처', '정자 출처',
    '난자 기증자 나이', '정자 기증자 나이', '동결 배아 사용 여부', '신선 배아 사용 여부',
    '기증 배아 사용 여부', '대리모 여부', 'PGD 시술 여부', 'PGS 시술 여부','총 시술 경험','주된 불임 원인 개수'
]

# 범주형 변수 문자열 변환
df[categorical_columns] = df[categorical_columns].astype(str)
test[categorical_columns] = test[categorical_columns].astype(str)
df1[categorical_columns] = df1[categorical_columns].astype(str)
df2[categorical_columns] = df2[categorical_columns].astype(str)
test1[categorical_columns] = test1[categorical_columns].astype(str).fillna('nan')
test2[categorical_columns] = test2[categorical_columns].astype(str).fillna('nan')

# 종속변수 (임신 성공 여부)
X1, y1 = df1.drop(columns=['임신 성공 여부']), df1['임신 성공 여부']
X2, y2 = df2.drop(columns=['임신 성공 여부']), df2['임신 성공 여부']
X_test1 = test1.drop(columns=['임신 성공 여부'], errors='ignore')
X_test2 = test2.drop(columns=['임신 성공 여부'], errors='ignore')

# IVF 모델 학습 및 예측 (5-Fold 교차 검증)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []
final_preds1 = np.zeros(len(X_test1))
for train_idx, val_idx in skf.split(X1, y1):
    X_train, X_val = X1.iloc[train_idx], X1.iloc[val_idx]
    y_train, y_val = y1.iloc[train_idx], y1.iloc[val_idx]

    model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, cat_features=categorical_columns, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    auc_scores.append(auc)
    final_preds1 += model.predict_proba(X_test1)[:, 1] / skf.n_splits
print(f'IVF 모델 평균 ROC AUC: {np.mean(auc_scores):.4f}')

# 결과 저장
sample_submission.loc[test1.index, '임신 성공 여부'] = final_preds1

# DI 모델 학습 및 예측 (10-Fold 교차 검증)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
auc_scores = []
final_preds2 = np.zeros(len(X_test2))
for train_idx, val_idx in skf.split(X2, y2):
    X_train, X_val = X2.iloc[train_idx], X2.iloc[val_idx]
    y_train, y_val = y2.iloc[train_idx], y2.iloc[val_idx]

    model = CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.05, cat_features=categorical_columns, verbose=0)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=0)
    preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, preds)
    auc_scores.append(auc)
    final_preds2 += model.predict_proba(X_test2)[:, 1] / skf.n_splits
print(f'DI 모델 평균 ROC AUC: {np.mean(auc_scores):.4f}')

# 예측값을 원래 index로 복원하여 sample_submission.csv 저장
test1['probability'] = final_preds1
test2['probability'] = final_preds2

# test1과 test2를 합치고 인덱스를 정렬
final_submission = pd.concat([test1[['probability']], test2[['probability']]]).sort_index()

# sample_submission 불러오기
sample_submission = pd.read_csv('sample_submission.csv')

# 확률값을 sample_submission에 넣기
sample_submission['probability'] = final_submission['probability'].values

# 결과를 새로운 파일에 저장
sample_submission.to_csv('submission.csv', index=False)
