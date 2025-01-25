import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import optuna
from optuna.samplers import TPESampler

# 데이터 불러오기
df = pd.read_csv('train.csv')
df = df.drop(['ID'], axis=1)

# 범주형 데이터 인코딩
categorical_columns = ['모델', '제조사', '차량상태', '구동방식', '사고이력']
encoder = OrdinalEncoder()
df[categorical_columns] = encoder.fit_transform(df[categorical_columns])

# 피처/타겟 데이터 분리
y = df['가격(백만원)']
X = df.drop(['가격(백만원)'], axis=1)

# Optuna로 하이퍼파라미터 튜닝
def objective(trial):
    # 하이퍼파라미터 범위 정의
    param = {
        'max_depth': trial.suggest_int('max_depth', 5, 6),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.25, 0.3),
        'subsample': trial.suggest_uniform('subsample', 0.9, 1.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.9, 1.0),
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 2),
        'max_delta_step': trial.suggest_int('max_delta_step', 0, 1),
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'seed': 42  # XGBoost 랜덤 시드 고정
    }

    # 교차검증을 위한 KFold
    cv = KFold(n_splits=10, shuffle=True, random_state=42)  # KFold 랜덤 시드 고정

    # 교차검증을 통해 모델 평가
    scores = []
    for train_idx, val_idx in cv.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # DMatrix로 변환
        train_dmatrix = xgb.DMatrix(X_train, label=y_train)
        val_dmatrix = xgb.DMatrix(X_val, label=y_val)

        # xgb.train()으로 모델 학습 (early_stopping_rounds=50 추가)
        model = xgb.train(
            params=param,
            dtrain=train_dmatrix,
            num_boost_round=1000,
            evals=[(val_dmatrix, 'eval')],
            early_stopping_rounds=50,
            verbose_eval=False
        )

        # 검증 데이터로 예측 및 성능 평가
        y_pred = model.predict(val_dmatrix)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        scores.append(rmse)

    # 교차검증 평균 RMSE 반환
    return np.mean(scores)

# 랜덤 시드 고정
sampler = TPESampler(seed=42)

# Optuna Study 생성 및 실행
study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=50)

# 최적 파라미터 출력
print("Best Parameters from Optuna:", study.best_params)
print("Best RMSE from Optuna:", study.best_value)

# 최적 하이퍼파라미터로 모델 재학습 및 최종 평가
best_params = study.best_params
final_model = xgb.XGBRegressor(random_state=42, **best_params, n_estimators=1000)

# 전체 데이터로 다시 학습하고 평가 (교차검증 외의 별도 분할 없이)
final_model.fit(X, y)
y_pred_final = final_model.predict(X)
final_rmse = np.sqrt(mean_squared_error(y, y_pred_final))

print(f"Final RMSE (on the full dataset): {final_rmse:.4f}")

