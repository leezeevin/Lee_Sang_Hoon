import pandas as pd
train = pd.read_csv('C:\\Users\\USER\\Downloads\\기업성공\\train.csv')
test = pd.read_csv('C:\\Users\\USER\\Downloads\\기업성공\\test.csv')

#eda 
#기초통계량 확인
print(train.describe())

#타겟 변수의 분포 확인
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(train['성공확률'], kde=True)
plt.xlabel('success percentage')
plt.show()

#상관계수 확인
cor = train.drop(['ID','국가','분야','투자단계','인수여부','상장여부','기업가치(백억원)'], axis=1).corr()['성공확률'].sort_values()
print(cor)

#결측치 확인
print(train.isnull().sum().sort_values(ascending=False))

#범주형 분포 확인
categorical = ['국가','분야','투자단계','인수여부','상장여부','기업가치(백억원)']
for col in categorical:
    print(f"\n🔹 {col} value counts:")
    print(train[col].value_counts(dropna=False))

#결측치 처리
train['고객수(백만명)'].fillna(train['고객수(백만명)'].median(), inplace=True)
train['직원 수'].fillna(train['직원 수'].median(), inplace=True)
train['기업가치(백억원)'].fillna(train['기업가치(백억원)'].mode()[0], inplace=True)
train['분야'].fillna(train['분야'].mode()[0], inplace=True)

#데이터 분할
y = train['성공확률']
X = train.drop(['성공확률','ID'], axis=1)

#원핫 인코딩
X_dum = pd.get_dummies(X, columns = ['국가','분야','투자단계','인수여부', '상장여부','기업가치(백억원)'], drop_first=True)

import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# 데이터 준비 (X_dum과 y는 이미 준비된 상태로 가정)
# X_dum = pd.get_dummies(X, columns=['국가','분야','투자단계','인수여부', '상장여부','기업가치(백억원)'], drop_first=True)

# 1. K-Fold Cross Validation 설정 (k=10)
kf = KFold(n_splits=10, shuffle=True, random_state=1)

# 2. Weighted MAE 계산 함수 정의
def weighted_mae(y_true, y_pred, weights):
    abs_errors = np.abs(y_true - y_pred)
    return np.sum(abs_errors * weights) / np.sum(weights)

# 3. 모델 훈련 및 교차검증 수행
def cross_validate_model(X, y):
    fold_weights = []
    fold_errors = []

    # KFold 교차검증 수행
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # 4. 랜덤포레스트 모델 학습
        rf = RandomForestRegressor(random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # 5. 빈도 기반 가중치 계산
        y_val_rounded = y_val.round(2)  # 소수점 두 자리 반올림
        value_counts = pd.Series(y_val_rounded).value_counts()
        freq_map = value_counts.to_dict()
        weights = np.array([1 / freq_map[val] for val in y_val_rounded])

        # 6. Weighted MAE 계산
        wmae = weighted_mae(y_val, y_pred, weights)
        fold_weights.append(weights)
        fold_errors.append(wmae)

    # 7. 평균 가중치 MAE 반환
    return np.mean(fold_errors)

# 8. 하이퍼파라미터 최적화 (Bayesian Optimization)
search_space = {
    'n_estimators': Integer(100, 1000),  # 트리 개수
    'max_depth': Integer(3, 20),          # 트리 최대 깊이
    'min_samples_split': Integer(2, 10),  # 분할에 필요한 최소 샘플 수
    'min_samples_leaf': Integer(1, 10),   # 리프 노드 최소 샘플 수
}

opt = BayesSearchCV(
    RandomForestRegressor(random_state=1),
    search_space,
    n_iter=50,  # 탐색할 횟수
    cv=3,  # 내부 교차검증 (3폴드)
    n_jobs=-1,  # 병렬 처리
    scoring=None,  # scoring은 None으로 두고 내부 MAE 계산
    verbose=1,
)

# 9. 베이지안 최적화 수행 (하이퍼파라미터 조정)
opt.fit(X_dum, y)

# 10. 최적의 하이퍼파라미터 출력
print("Best Hyperparameters:", opt.best_params_)

# 11. 최적 모델로 K-Fold 교차검증 후 평가
best_rf = opt.best_estimator_
final_wmae = cross_validate_model(X_dum, y)
print("Final Weighted MAE after Hyperparameter Optimization:", final_wmae)

# 12. 중요도 0.01 이상인 피처만 선택
importances = best_rf.feature_importances_
feature_names = X_dum.columns
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# 11. 중요도 0.01 이상인 피처만 선택
selected_features = importance_df[importance_df['Importance'] >= 0.01]['Feature'].tolist()
X_selected = X_dum[selected_features]

# 12. 선택된 피처로 다시 교차검증 평가
final_wmae_selected = cross_validate_model(X_selected, y)
print("Weighted MAE with Selected Features (Importance ≥ 0.01):", final_wmae_selected)

best_rf_selected = RandomForestRegressor(
    n_estimators=opt.best_params_['n_estimators'],
    max_depth=opt.best_params_['max_depth'],
    min_samples_split=opt.best_params_['min_samples_split'],
    min_samples_leaf=opt.best_params_['min_samples_leaf'],
    random_state=1
)
best_rf_selected.fit(X_selected, y)

# 13. test 데이터에 대해서도 dummies 처리
test.drop('ID', axis=1, inplace=True)
X_test_dum = pd.get_dummies(test, columns=['국가','분야','투자단계','인수여부', '상장여부','기업가치(백억원)'], drop_first=True)

X_test_dum_1 = X_test_dum[selected_features]

# 14. test 데이터에 대해 예측 수행
y_test_pred = best_rf_selected.predict(X_test_dum_1)

# 15. sample_submission.csv 파일 불러오기
sample_submission = pd.read_csv('C:\\Users\\USER\\Downloads\\기업성공\\sample_submission.csv')

# 16. 예측값을 '성공확률' 컬럼에 삽입
sample_submission['성공확률'] = y_test_pred

# 17. 결과 저장
sample_submission.to_csv('submission성공12.csv', index=False)

print("결과가 'submission.csv' 파일에 저장되었습니다.")
