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