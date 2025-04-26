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