import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pymc as pm
import arviz as az

# ==========================================================
# 0. 환경 세팅 및 Matplotlib 그래픽 엔진 안전 모드
# ==========================================================
print("=== 0. 환경 세팅 및 시각화 엔진 초기화 ===")
plt.rc('font', family='sans-serif')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


# ==========================================================
# 1. 데이터 로드 및 통합 (신한카드 종로구 데이터)
# ==========================================================
print("\n=== 1. 신한카드 종로구 데이터 통합 중... ===")
card_files = sorted(glob.glob('./신한카드_서울_종로구_2020*.csv'))
print(f"-> 발견된 카드 파일: {card_files}")

card_df_list = []
for file in card_files:
    df = pd.read_csv(file, encoding='utf-8-sig')
    card_df_list.append(df)

total_card = pd.concat(card_df_list, axis=0, ignore_index=True)
total_card['기준년월'] = total_card['기준년월'].astype(str)
print(f"-> 카드 데이터 통합 완료! 총 행 수: {total_card.shape[0]}개")


# ==========================================================
# 2. 한국은행 기준금리 데이터 전처리
# ==========================================================
print("\n=== 2. 한국은행 기준금리 데이터 전처리 중... ===")
interest_file = glob.glob('./*기준금리*.csv')[0]
print(f"-> 매칭된 기준금리 파일: {interest_file}")

interest_df = pd.read_csv(interest_file, encoding='utf-8')
interest_df = interest_df[interest_df['변환'] == '원자료']
interest_melt = interest_df.melt(id_vars=['계정항목', '단위', '변환'], var_name='기준년월', value_name='기준금리')

interest_macro = interest_melt[['기준년월', '기준금리']].copy()
interest_macro['기준년월'] = interest_macro['기준년월'].str.replace('/', '')
interest_macro = interest_macro[interest_macro['기준년월'].str.isnumeric() == True]
interest_macro['기준금리'] = pd.to_numeric(interest_macro['기준금리'], errors='coerce')
print("-> 기준금리 전처리 완료!")


# ==========================================================
# 3. 소비자심리지수(CCSI) 데이터 전처리
# ==========================================================
print("\n=== 3. 소비자심리지수(CCSI) 데이터 전처리 중... ===")
ccsi_file = glob.glob('./*소비자동향조사*.csv')[0]
print(f"-> 매칭된 소비자심리 파일: {ccsi_file}")

try:
    ccsi_df = pd.read_csv(ccsi_file, encoding='utf-8')
except UnicodeDecodeError:
    ccsi_df = pd.read_csv(ccsi_file, encoding='cp949')

ccsi_all = ccsi_df[ccsi_df['CSI분류코드'].str.strip() == '전체']
ccsi_melt = ccsi_all.melt(id_vars=['통계표', 'CSI코드', 'CSI분류코드', '단위', '변환'], var_name='기준년월', value_name='소비자심리지수')

ccsi_macro = ccsi_melt[['기준년월', '소비자심리지수']].copy()
ccsi_macro['기준년월'] = ccsi_macro['기준년월'].str.replace('/', '')
ccsi_macro = ccsi_macro[ccsi_macro['기준년월'].str.isnumeric() == True]
ccsi_macro['소비자심리지수'] = pd.to_numeric(ccsi_macro['소비자심리지수'], errors='coerce')
print("-> 소비자심리지수 전처리 완료!")


# ==========================================================
# 4. 소비자물가지수(CPI) 데이터 전처리
# ==========================================================
print("\n=== 4. 소비자물가지수(CPI) 데이터 전처리 중... ===")
cpi_file = glob.glob('./*소비자물가지수*.csv')[0]
print(f"-> 매칭된 소비자물가 파일: {cpi_file}")

try:
    cpi_df = pd.read_csv(cpi_file, encoding='utf-8')
except UnicodeDecodeError:
    cpi_df = pd.read_csv(cpi_file, encoding='cp949')

cpi_all = cpi_df[cpi_df['시도별'] == '전국']
cpi_melt = cpi_all.melt(id_vars=['시도별'], var_name='기준년월', value_name='소비자물가지수')

cpi_macro = cpi_melt[['기준년월', '소비자물가지수']].copy()
cpi_macro['기준년월'] = cpi_macro['기준년월'].str.replace('.', '', regex=False)
cpi_macro = cpi_macro[cpi_macro['기준년월'].str.isnumeric() == True]
cpi_macro['소비자물가지수'] = pd.to_numeric(cpi_macro['소비자물가지수'], errors='coerce')
print("-> 소비자물가지수 전처리 완료!")


# ==========================================================
# 5. 데이터 최종 결합 및 2차 정제·스케일링
# ==========================================================
print("\n=== 5. 거시경제 지표 통합 및 데이터 정제/스케일링 중... ===")
macro_df = pd.merge(interest_macro, ccsi_macro, on='기준년월', how='inner')
macro_df = pd.merge(macro_df, cpi_macro, on='기준년월', how='inner')
macro_df['기준년월'] = macro_df['기준년월'].astype(str)

final_df = pd.merge(total_card, macro_df, on='기준년월', how='left')

# 마이너스 매출 및 오류 데이터 제거 (0원 이하 제거)
clean_df = final_df[final_df['카드매출금액'] > 0].copy()

# 로그 변환 및 Z-score 표준화
clean_df['log_매출금액'] = np.log1p(clean_df['카드매출금액'])

def standardize(col):
    return (col - col.mean()) / col.std()

clean_df['scaled_매출'] = standardize(clean_df['log_매출금액'])
clean_df['scaled_금리'] = standardize(clean_df['기준금리'])
clean_df['scaled_심리'] = standardize(clean_df['소비자심리지수'])
clean_df['scaled_물가'] = standardize(clean_df['소비자물가지수'])

# 업종명 영문 매핑
industry_eng = {
    '문화레져': 'Culture/Leisure',
    '생활서비스': 'Life Service',
    '음식': 'Food/F&B',
    '일반유통': 'General Retail',
    '전문서비스': 'Professional',
    '종합유통': 'Total Retail'
}
clean_df['Industry_ENG'] = clean_df['업종대분류'].map(industry_eng)
print(f"-> 최종 정제 완료! 분석 대상 데이터 수: {clean_df.shape[0]}개")


# ==========================================================
# 6. 베이지안 EDA (시각화)
# ==========================================================
print("\n=== 6. 베이지안 분석용 EDA 시각화 생성... ===")

# ① 업종별 매출 분포 Boxplot
plt.figure(figsize=(10, 5))
sns.boxplot(data=clean_df, x='Industry_ENG', y='log_매출금액', hue='Industry_ENG', palette='muted', legend=False)
plt.title('Distribution of Log Sales by Industry (For Intercept Prior)', fontsize=14, pad=15)
plt.xlabel('Industry Category', fontsize=12)
plt.ylabel('Log Sales Amount', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=15)
plt.show()

# ② 기준금리 vs 업종별 매출 추세 FacetGrid
grid = sns.FacetGrid(clean_df, col="Industry_ENG", col_wrap=3, height=4, sharey=False)
grid.map(sns.regplot, "기준금리", "log_매출금액",
         scatter_kws={'alpha':0.4, 'color':'gray'},
         line_kws={'color':'firebrick', 'linewidth':2})

for ax, title in zip(grid.axes.flat, sorted(clean_df['Industry_ENG'].unique())):
    ax.set_title(title, fontsize=12, weight='bold')
    ax.set_xlabel('Interest Rate', fontsize=10)
    ax.set_ylabel('Log Sales', fontsize=10)
plt.subplots_adjust(top=0.85, hspace=0.4)
grid.fig.suptitle('Industry Sales Trend by Changes in Interest Rates', fontsize=15, weight='bold')
plt.show()

# ③ 거시지표 간 다중공선성 점검 Heatmap
plt.figure(figsize=(6, 5))
macro_corr = clean_df[['기준금리', '소비자심리지수', '소비자물가지수']].corr()
macro_corr.index = ['Interest Rate', 'CCSI(Sentiment)', 'CPI(Inflation)']
macro_corr.columns = ['Interest Rate', 'CCSI(Sentiment)', 'CPI(Inflation)']

sns.heatmap(macro_corr, annot=True, cmap='RdYlBu_r', fmt=".2f", vmin=-1, vmax=1, linewidths=0.5)
plt.title('Macroeconomic Indicators Correlation Matrix', fontsize=13, pad=15)
plt.show()


# ==========================================================
# 7. PyMC 베이지안 계층 모델 구조 정의 및 샘플링
# ==========================================================
print("\n=== 7. 베이지안 계층 모델 설계 및 MCMC 샘플링 ===")
# 범주형 인덱싱 생성
clean_df['업종_code'] = clean_df['업종대분류'].astype('category').cat.codes
industry_categories = clean_df['업종대분류'].astype('category').cat.categories.tolist()
industry_eng_list = [industry_eng[cat] for cat in industry_categories]

# 모델 입력용 벡터 추출
sub_idx = clean_df['업종_code'].values
X_rate = clean_df['scaled_금리'].values
X_sentiment = clean_df['scaled_심리'].values
X_cpi = clean_df['scaled_물가'].values
Y_sales = clean_df['scaled_매출'].values
num_industries = len(industry_categories)

with pm.Model() as hierarchical_model:
    # [Hyper-priors] 대한민국 전체 공통 거시경제 흐름 (상위 계층)
    mu_beta_rate = pm.Normal('mu_beta_rate', mu=0.0, sigma=1.0)
    sigma_beta_rate = pm.Exponential('sigma_beta_rate', lam=1.0)

    # [Priors] 6개 개별 업종 고유 파라미터 (하위 계층)
    alpha = pm.Normal('alpha', mu=0.0, sigma=1.0, shape=num_industries)
    
    # 상위 계층의 정보 공유를 받는 부분 수축(Partial Pooling) 적용 변수
    beta_rate = pm.Normal('beta_rate', mu=mu_beta_rate, sigma=sigma_beta_rate, shape=num_industries)

    # 나머지 통제 변수 (업종별 독립 추정)
    beta_sentiment = pm.Normal('beta_sentiment', mu=0.0, sigma=1.0, shape=num_industries)
    beta_cpi = pm.Normal('beta_cpi', mu=0.0, sigma=1.0, shape=num_industries)

    # [Linear Regression Equation] 회귀 예측식
    mu_y = (alpha[sub_idx] +
            beta_rate[sub_idx] * X_rate +
            beta_sentiment[sub_idx] * X_sentiment +
            beta_cpi[sub_idx] * X_cpi)

    # 관측 오차 및 Likelihood 결합
    sigma_y = pm.Exponential('sigma_y', lam=1.0)
    y_obs = pm.Normal('y_obs', mu=mu_y, sigma=sigma_y, observed=Y_sales)

    # MCMC Sampling 가동
    print("\n🚀 NUTS Sampler 기반 MCMC 사후분포 시뮬레이션 시작... (약 30초 소요)")
    trace = pm.sample(draws=1000, tune=1000, chains=2, random_seed=42, return_inferencedata=True)

print("\n🎉 MCMC 샘플링 완료! 사후분포 수집에 성공했습니다.")


# ==========================================================
# 8. 모델 수렴 검증 (MCMC Diagnostics)
# ==========================================================
print("\n=== 8. 모델 수렴 검증 및 Trace Plot ===")
summary_convergence = az.summary(trace, var_names=['mu_beta_rate', 'beta_rate'])
print(summary_convergence[['mean', 'sd', 'r_hat']].round(4))
print("\n💡 [체크 포인트] 모든 r_hat 값이 1.00xx 부근이면 성공적으로 수렴된 것입니다.")

# Trace Plot 생성
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
az.plot_trace(trace, var_names=['beta_rate'], compact=True, axes=axes)
plt.suptitle("Industry-level Slopes Trace Plot (Convergence Check)", fontsize=14, y=1.02, weight='bold')
plt.tight_layout()
plt.show()


# ==========================================================
# 9. 최종 시각화 및 결과 저장 (Forest Plot & 사후분포 성적표)
# ==========================================================
print("\n=== 9. 최종 결론 도출 및 결과 리포팅 ===")

# ① 기준금리 민감도 Forest Plot 생성
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_forest(trace, var_names=['beta_rate'], combined=True, ax=ax, colors='firebrick', linewidth=3, markersize=8)
ax.set_title("Posterior Distributions of Interest Rate Sensitivity by Industry", fontsize=14, pad=15, weight='bold')
ax.set_yticklabels([f"Beta: {name}" for name in reversed(industry_eng_list)], fontsize=11)
ax.set_xlabel("Effect Size (Standardized Coefficient)", fontsize=12)
ax.grid(True, alpha=0.3, linestyle='--')
plt.axvline(x=0, color='black', linestyle=':', linewidth=1.5)
plt.show()

# ② 거시경제 지표(금리·심리·물가) 전체 성적표 취합
target_vars = ['beta_rate', 'beta_sentiment', 'beta_cpi']
summary_all = az.summary(trace, var_names=target_vars)
result_df = summary_all[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']]

# 인덱스를 직관적인 영문 업종명 규칙으로 재정의
new_indices = []
for var in target_vars:
    for name in industry_eng_list:
        new_indices.append(f"{var} ({name})")
result_df.index = new_indices

print("\n==========================================================================")
print("📊 [최종 결과] 업종별 거시경제 지표(금리·심리·물가) 민감도 사후분포 성적표")
print("==========================================================================")
display(result_df.round(4))

# ③ PPT 작성을 위한 CSV 파일 자동 저장
output_filename = "industry_macro_sensitivity_results.csv"
result_df.to_csv(output_filename, encoding='utf-8-sig')
print(f"\n💾 성적표가 '{output_filename}' 파일로 저장되었습니다. PPT 표 작성 시 활용하세요!")
