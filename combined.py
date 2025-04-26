import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
import datetime

# 1. 날짜 설정
start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.today()

# 2. FRED 데이터 불러오기
gdp = pdr.DataReader('GDPC1', 'fred', start, end)         # 실질 GDP
pot_gdp = pdr.DataReader('GDPPOT', 'fred', start, end)    # 잠재 GDP
pce = pdr.DataReader('PCEPILFE', 'fred', start, end)      # Core PCE 지수

# 3. log변환
gdp['log_gdp'] = np.log(gdp['GDPC1'])
pot_gdp['log_pot'] = np.log(pot_gdp['GDPPOT'])

# 4. output gap 계산: log GDP - log POT
output_gap = gdp['log_gdp'] - pot_gdp['log_pot']
output_gap.name = 'output_gap'

# 5. YoY 인플레이션 계산 (quarterly 기준 4분기 차이)
pce['YoY_inflation'] = pce['PCEPILFE'].pct_change(periods=4) * 100

# 6. 데이터 통합
data = pd.concat([output_gap, pce['YoY_inflation']], axis=1).dropna()

# 7. 시각화 (선택적)
data.plot(title='Output Gap and YoY Core PCE Inflation', figsize=(10, 4), grid=True)
plt.ylabel('Output Gap / Inflation (%)')
plt.tight_layout()
plt.show()

# 8. 결과 확인
print(data.tail())

data.to_excel(r'C:\Users\ann\Desktop\BK\data.xlsx')

import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# 1. VAR 모델 생성
model = VAR(data)

# 2. VAR(1) 추정
results = model.fit(1)

# 3. A 행렬 출력
A = results.coefs[0]
print("Estimated A matrix:")
print(A)

# 4. 예측값 계산 (fitted values)
fitted = results.fittedvalues

# 5. 실제 vs 예측값 시각화
fig, ax = plt.subplots(2, 1, figsize=(10, 6))

# Output Gap
ax[0].plot(data.index, data['output_gap'], label='Actual Output Gap')
ax[0].plot(fitted.index, fitted['output_gap'], linestyle='--', label='Fitted Output Gap')
ax[0].legend()
ax[0].set_title('Output Gap: Actual vs Fitted')
ax[0].grid(True)

# YoY Inflation
ax[1].plot(data.index, data['YoY_inflation'], label='Actual YoY Inflation')
ax[1].plot(fitted.index, fitted['YoY_inflation'], linestyle='--', label='Fitted YoY Inflation')
ax[1].legend()
ax[1].set_title('YoY Inflation: Actual vs Fitted')
ax[1].grid(True)

plt.tight_layout()

# 6. 그래프 저장
plt.savefig(r'C:\Users\ann\Desktop\BK\var_fit_results.png')
plt.show()

# 1. A 행렬 저장
A_df = pd.DataFrame(A, columns=data.columns, index=data.columns)
A_df.to_excel(r'C:\Users\ann\Desktop\BK\A_matrix.xlsx')

# 2. 고유값 저장
eigvals = np.linalg.eigvals(A)
moduli = np.abs(eigvals)

eig_df = pd.DataFrame({
    'Eigenvalue': eigvals,
    'Modulus': moduli
})
eig_df.to_excel(r'C:\Users\ann\Desktop\BK\eigenvalues.xlsx')

# 3. fitted values 저장
results.fittedvalues.to_excel(r'C:\Users\ann\Desktop\BK\fitted_values.xlsx')

# 4. VAR 회귀요약 저장
with open(r'C:\Users\ann\Desktop\BK\var_summary.txt', 'w') as f:
    f.write(str(results.summary()))

    from statsmodels.tsa.api import VAR
import numpy as np

# BK 만족하는 VAR 탐색
max_p = 10  # 최대 시차 p=10까지 시도
num_forward_looking = 1  # forward-looking 변수 개수 (ex: inflation)

# 3. 반복 탐색
found = False

for p in range(1, max_p + 1):
    try:
        model = VAR(data)
        results = model.fit(p)
        
        # A_total = coefs[0] + coefs[1] + ... + coefs[p-1]
        A_total = results.coefs.sum(axis=0)
        
        eigvals = np.linalg.eigvals(A_total)
        moduli = np.abs(eigvals)
        num_unstable = np.sum(moduli > 1)
        
        print(f"p = {p}, unstable eigenvalues = {num_unstable}")

        if num_unstable == num_forward_looking:
            print(f"\n✅ BK condition satisfied at p = {p}!")
            found = True
            break

    except Exception as e:
        print(f"Error at p = {p}: {e}")
        continue

if not found:
    print("\n❌ No p found satisfying BK condition up to p =", max_p)

save_dir = r'C:\Users\ann\Desktop\BK'  # 저장 경로

### 1. 실제 데이터 기반 IRF
model = VAR(data)
results = model.fit(1)
irf_actual = results.irf(10)
fig1 = irf_actual.plot(orth=False)
plt.suptitle('IRF from Actual Estimated A (BK NOT satisfied)')
plt.tight_layout()
fig1.savefig(f'{save_dir}\\irf_actual.png')
plt.close()

### 2. BK 조건 만족 A (1 unstable eigenvalue)
A_bk = np.array([[0.7, 0.2], [0.3, 1.1]])
n_periods = 50
np.random.seed(0)
x_bk = np.zeros((n_periods, 2))
shock = np.random.normal(size=(n_periods, 2)) * 0.1

for t in range(1, n_periods):
    x_bk[t] = A_bk @ x_bk[t-1] + shock[t]

data_bk = pd.DataFrame(x_bk, columns=['output_gap', 'YoY_inflation'])
model_bk = VAR(data_bk)
results_bk = model_bk.fit(1)
irf_bk = results_bk.irf(10)
fig2 = irf_bk.plot(orth=False)
plt.suptitle('IRF from Artificial BK-satisfied A')
plt.tight_layout()
fig2.savefig(f'{save_dir}\\irf_bk_satisfied.png')
plt.close()

### 3. BK 조건 위반 A (2 unstable eigenvalues → no solution)
A_nosol = np.array([[1.2, 0.4], [0.3, 1.1]])
np.random.seed(1)
x_nosol = np.zeros((n_periods, 2))
shock = np.random.normal(size=(n_periods, 2)) * 0.1

for t in range(1, n_periods):
    x_nosol[t] = A_nosol @ x_nosol[t-1] + shock[t]

data_nosol = pd.DataFrame(x_nosol, columns=['output_gap', 'YoY_inflation'])
model_nosol = VAR(data_nosol)
results_nosol = model_nosol.fit(1)
irf_nosol = results_nosol.irf(10)
fig3 = irf_nosol.plot(orth=False)
plt.suptitle('IRF from Artificial A (BK NO SOLUTION)')
plt.tight_layout()
fig3.savefig(f'{save_dir}\\irf_bk_no_solution.png')
plt.close()

# 1. 인위적으로 불안정한 A 행렬 설계 (실험 목적)
A_nosol = np.array([
    [2, 0.8],
    [0.6, 2.5]
])

# 2. 시뮬레이션
n_periods = 50
np.random.seed(42)
x_nosol = np.zeros((n_periods, 2))
shock = np.random.normal(size=(n_periods, 2)) * 0.1

for t in range(1, n_periods):
    x_nosol[t] = A_nosol @ x_nosol[t-1] + shock[t]

sim_data = pd.DataFrame(x_nosol, columns=['output_gap', 'YoY_inflation'])

# 3. VAR 추정
model = VAR(sim_data)
results = model.fit(1)

# 4. BK 조건 체크
num_fwd = 1  # forward-looking variable 수
A_est = results.coefs[0]
eigvals = np.linalg.eigvals(A_est)
moduli = np.abs(eigvals)
num_unstable = np.sum(moduli > 1)

print("Estimated A matrix:\n", A_est)
print("Eigenvalues:", eigvals)
print(f"Unstable eigenvalues: {num_unstable}, Forward-looking vars: {num_fwd}")

# 5. BK 조건 위반 여부에 따라 IRF 처리
if num_unstable > num_fwd:
    print("❌ BK condition violated – NO SOLUTION (system is explosive)")
else:
    try:
        print("✅ BK condition ok – generating IRF")
        irf = results.irf(10)
        fig = irf.plot(orth=False)
        plt.suptitle('IRF (BK condition OK or indeterminate)')
        plt.tight_layout()
        fig.savefig(r'C:\Users\ann\Desktop\BK\irf_checked.png')
        plt.show()
    except np.linalg.LinAlgError as e:
        print("❌ IRF computation failed due to singular matrix:", e)

if num_unstable > num_fwd:
    print("❌ BK condition violated – NO SOLUTION (system is explosive)")
else:
    print("✅ BK condition ok – generating IRF")
    irf = results.irf(10)  
    fig = irf.plot(orth=False)
    plt.suptitle('IRF (BK condition OK or indeterminate)')
    plt.tight_layout()
    fig.savefig(r'C:\Users\ann\Desktop\BK\irf_checked.png')
    plt.show()

# 4. 고유값 확인
A_est = results.coefs[0]
eigvals = np.linalg.eigvals(A_est)
print("Estimated A matrix:")
print(A_est)
print("Eigenvalues:", eigvals)

# 5. IRF 계산 및 저장
irf = results.irf(10)
fig = irf.plot(orth=False)
plt.suptitle('IRF from Strongly Unstable A (BK Violated: NO SOLUTION)')
plt.tight_layout()
fig.savefig(r'C:\Users\ann\Desktop\BK\irf_bk_strong_nosolution.png')
plt.show()