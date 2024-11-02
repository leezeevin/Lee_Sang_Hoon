# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 12:38:06 2024

@author: USER
"""

import pandas as pd


train = pd.read_csv("train.csv")
smile = train['Smiles']
smile_list  = list(smile)
test = pd.read_csv("test.csv")
sample = pd.read_csv('sample_submission.csv')

import numpy as np
train['log_ic50'] = np.log(train['IC50_nM'])

pip install rdkit pandas scikit-learn

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split


# 피처 추출: SMILES로부터 분자량(Molecular Weight) 계산
def get_molecular_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol)

def calculate_logp(smiles):
    # SMILES 문자열을 RDKit Mol 객체로 변환
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None  # 유효하지 않은 SMILES 문자열일 경우
    
    # LogP 계산
    logp = Descriptors.MolLogP(mol)
    return logp
def calculate_hbd(smiles):
    # SMILES 문자열을 RDKit Mol 객체로 변환
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None  # 유효하지 않은 SMILES 문자열일 경우
    
    # 수소기부체 수 계산
    hbd = Descriptors.NumHDonors(mol)
    return hbd
def calculate_hba(smiles):
    # SMILES 문자열을 RDKit Mol 객체로 변환
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None  # 유효하지 않은 SMILES 문자열일 경우
    
    # 수소결합 원자 수 계산
    hba = Descriptors.NumHAcceptors(mol)
    return hba
def calculate_psa(smiles):
    # SMILES 문자열을 통해 분자 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    
    # PSA (Polar Surface Area) 계산
    psa = Descriptors.TPSA(mol)
    
    return psa
def calculate_rotatable_bonds(smiles):
    # SMILES 문자열을 RDKit Mol 객체로 변환
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return None  # 유효하지 않은 SMILES 문자열일 경우
    
    # 회전 가능한 결합 수 계산
    rotatable_bonds = Descriptors.NumRotatableBonds(mol)
    return rotatable_bonds

def calculate_ring_count(smiles):
    # SMILES 문자열을 통해 분자 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    
    # Ring Count 계산
    ring_count = rdMolDescriptors.CalcNumRings(mol)
    
    return ring_count
def calculate_aromatic_rings(smiles):
    # SMILES 문자열을 통해 분자 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    
    # 분자 내 방향족 링의 수 계산
    aromatic_rings = Descriptors.NumAromaticRings(mol)
    
    return aromatic_rings
def calculate_bond_count(smiles):
    # SMILES 문자열을 통해 분자 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    
    # 결합 수 계산
    bond_count = mol.GetNumBonds()
    
    return bond_count
def calculate_aliphatic_bond_count(smiles):
    # SMILES 문자열을 통해 분자 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    
    # 알리파틱 결합 수 계산
    aliphatic_bond_count = 0
    
    for bond in mol.GetBonds():
        # 결합이 방향족이 아닌 경우 (알리파틱 결합)
        if not bond.GetIsAromatic():
            aliphatic_bond_count += 1
    
    return aliphatic_bond_count
def calculate_molecular_complexity(smiles):
    # SMILES 문자열을 통해 분자 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    
    # BertzCT 지표를 통해 분자 복잡성 계산
    molecular_complexity = Descriptors.BertzCT(mol)
    
    return molecular_complexity
def calculate_fraction_sp3_carbons(smiles):
    # SMILES 문자열을 통해 분자 객체 생성
    mol = Chem.MolFromSmiles(smiles)
    
    # Fraction of sp³ hybridized carbons 계산
    fraction_sp3_carbons = rdMolDescriptors.CalcFractionCSP3(mol)
    
    return fraction_sp3_carbons





# 분자량 계산
data = pd.DataFrame(smile_list, columns=['SMILES'])
data['Molecular_Weight'] = data['SMILES'].apply(get_molecular_weight)
train['Molecular_Weight'] = data['SMILES'].apply(get_molecular_weight)

# logp 계산
data['logp'] = data['SMILES'].apply(calculate_logp)
train['logp'] = data['SMILES'].apply(calculate_logp)

#hbd 계산
data['hbd'] = data['SMILES'].apply(calculate_hbd)
train['hbd'] = data['SMILES'].apply(calculate_hbd)

#hba 계산
data['hba'] = data['SMILES'].apply(calculate_hba)
train['hba'] = data['SMILES'].apply(calculate_hba)

#psa 계산
data['psa'] = data['SMILES'].apply(calculate_psa)
train['psa'] = data['SMILES'].apply(calculate_psa)

#회전가능한 결합수 계산
data['rotatable_bonds'] = data['SMILES'].apply(calculate_rotatable_bonds)
train['rotatable_bonds'] = data['SMILES'].apply(calculate_rotatable_bonds)

#ring_count 계산
data['ring_count'] = data['SMILES'].apply(calculate_ring_count)
train['ring_count'] = data['SMILES'].apply(calculate_ring_count)

#aromatic_rings 계산
data['aromatic_rings'] = data['SMILES'].apply(calculate_aromatic_rings)
train['aromatic_rings'] = data['SMILES'].apply(calculate_aromatic_rings)

#bond_count 계산
data['bond_count'] = data['SMILES'].apply(calculate_bond_count)
train['bond_count'] = data['SMILES'].apply(calculate_bond_count)

#aliphatic_bond_count 계산
data['aliphatic_bond_count'] = data['SMILES'].apply(calculate_aliphatic_bond_count)
train['aliphatic_bond_count'] = data['SMILES'].apply(calculate_aliphatic_bond_count)

#molecular_complexity 계산
data['molecular_complexity'] = data['SMILES'].apply(calculate_molecular_complexity)
train['molecular_complexity'] = data['SMILES'].apply(calculate_molecular_complexity)

#fraction_sp3_carbons 계산
data['fraction_sp3_carbons'] = data['SMILES'].apply(calculate_fraction_sp3_carbons)
train['fraction_sp3_carbons'] = data['SMILES'].apply(calculate_fraction_sp3_carbons)



# 분자량 계산
test['Molecular_Weight'] = test['Smiles'].apply(get_molecular_weight)

# logp 계산
test['logp'] = test['Smiles'].apply(calculate_logp)

#hbd 계산
test['hbd'] = test['Smiles'].apply(calculate_hbd)


#hba 계산
test['hba'] = test['Smiles'].apply(calculate_hba)


#psa 계산
test['psa'] = test['Smiles'].apply(calculate_psa)


#회전가능한 결합수 계산
test['rotatable_bonds'] = test['Smiles'].apply(calculate_rotatable_bonds)

#ring_count 계산
test['ring_count'] = test['Smiles'].apply(calculate_ring_count)

#aromatic_rings 계산
test['aromatic_rings'] = test['Smiles'].apply(calculate_aromatic_rings)

#bond_count 계산
test['bond_count'] = test['Smiles'].apply(calculate_bond_count)

#aliphatic_bond_count 계산
test['aliphatic_bond_count'] = test['Smiles'].apply(calculate_aliphatic_bond_count)

#molecular_complexity 계산
test['molecular_complexity'] = test['Smiles'].apply(calculate_molecular_complexity)

#fraction_sp3_carbons 계산
test['fraction_sp3_carbons'] = test['Smiles'].apply(calculate_fraction_sp3_carbons)





data.info()

# 가상의 타겟 값 (예: 예측하고자 하는 값)
# 실제 상황에서는 이 값이 실험 데이터 등에서 추출됨
data['Target'] = train['IC50_nM']

# 회귀 분석
X = train[['Molecular_Weight','logp','hbd','hba','psa','rotatable_bonds','ring_count','bond_count','aliphatic_bond_count','molecular_complexity','fraction_sp3_carbons']]
X1 = test[['Molecular_Weight','logp','hbd','hba','psa','rotatable_bonds','ring_count','aromatic_rings','bond_count','aliphatic_bond_count','molecular_complexity','fraction_sp3_carbons']]
y = train['log_ic50']
feature_names = [['Molecular_Weight','logp','hbd','hba','psa','rotatable_bonds','ring_count','aromatic_rings','bond_count','aliphatic_bond_count','molecular_complexity','fraction_sp3_carbons']]
feature_names = pd.DataFrame(feature_names).T

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,  random_state=42)


#랜덤포레스팅

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# 랜덤 포레스트 모델 생성
model = RandomForestRegressor(n_estimators=300, max_depth = None, min_samples_split = 2, random_state=42)

# 모델 학습
data.to_csv('data.csv')
X1.to_csv('tsts.csv')
import os
os.getcwd()

model.fit(X_train, y_train)

# 예측
y_pred = model.predict(X_test)

y_pred_sub = model.predict(X1)
y_pred_sub_exp = np.exp(y_pred_sub)
sample['IC50_nM'] = y_pred_sub_exp
sample.to_csv('sample_submission.csv')


y_pred_exp = np.exp(y_pred)
y_test_exp = np.exp(y_test)
mse_og = mean_squared_error(y_test_exp, y_pred_exp)
a = (mse_og)**(1/2)/(train['IC50_nM'].max()  - train['IC50_nM'].min())
pred_pic50 = -np.log10(y_pred_exp*(1e-9))
test_pic50 = -np.log10(y_test_exp*(1e-9))
ab_error = np.abs(test_pic50 - pred_pic50)
correct_ratio = np.mean(ab_error <= 0.5)
0.5*(1-min(a,1)) + 0.5*correct_ratio

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = model.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

importances = model.feature_importances_

# 중요도와 변수 이름을 정렬
indices = np.argsort(importances)[::-1]
sorted_importances = importances[indices]
sorted_features = np.array(feature_names)[indices]

import matplotlib.pyplot as plt
# 변수 중요도 시각화
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), sorted_importances, align="center")
plt.xticks(range(X.shape[1]), sorted_features, rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()




from sklearn.model_selection import GridSearchCV

# 하이퍼파라미터 그리드 설정
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}


# GridSearchCV를 통한 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)


# 최적의 파라미터 출력
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')


# 스태킹 -----------------------------------------------#

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 개별 모델 학습
rf = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=2, random_state=42)
gbr = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)

rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)

# 개별 모델의 예측 결과를 메타 모델의 입력으로 사용
rf_pred = rf.predict(X_test)
gbr_pred = gbr.predict(X_test)

stacked_predictions = np.column_stack((rf_pred, gbr_pred))

# 메타 모델 학습
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_test)

# 최종 예측
final_predictions = meta_model.predict(stacked_predictions)

# 모델 평가
mse = mean_squared_error(y_test, final_predictions)
r2 = meta_model.score(stacked_predictions, y_test)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

from sklearn.model_selection import GridSearchCV

# 랜덤 포레스트 하이퍼파라미터 그리드 설정
param_grid_rf = {
    'n_estimators': [100, 200,300],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

# 그라디언트 부스팅 하이퍼파라미터 그리드 설정
param_grid_gbr = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1]
}

# GridSearchCV를 통해 최적의 하이퍼파라미터 탐색
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)

grid_search_gbr = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=param_grid_gbr, cv=5, n_jobs=-1, verbose=2)
grid_search_gbr.fit(X_train, y_train)

# 최적의 모델 저장
best_rf = grid_search_rf.best_estimator_
best_gbr = grid_search_gbr.best_estimator_


# 최적의 파라미터 출력
print(f'Best parameters: {best_rf}')
print(f'Best parameters: {best_gbr}')

# 개별 모델의 예측 결과를 메타 모델의 입력으로 사용
rf_pred = best_rf.predict(X_test)
gbr_pred = best_gbr.predict(X_test)

stacked_predictions = np.column_stack((rf_pred, gbr_pred))

# 메타 모델 학습
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_test)

# 최종 예측
final_predictions = meta_model.predict(stacked_predictions)

# 모델 평가
mse = mean_squared_error(y_test, final_predictions)
r2 = meta_model.score(stacked_predictions, y_test)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')









#svm-------------------------------------------#
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# 데이터 준비
X = train['Molecular_Weight'].values.reshape(-1, 1)
y = train['log_ic50']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVR 모델 정의
svr = SVR()

# 하이퍼파라미터 그리드 설정
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.3]
}

# GridSearchCV를 통한 하이퍼파라미터 튜닝
grid_search = GridSearchCV(estimator=svr, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# 최적의 모델 저장
best_svr = grid_search.best_estimator_

# 모델 학습
best_svr.fit(X_train, y_train)

# 예측
y_pred = best_svr.predict(X_test)

# 모델 평가
mse = mean_squared_error(y_test, y_pred)
r2 = best_svr.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
print(f'Best parameters found: {grid_search.best_params_}')
