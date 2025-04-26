import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt

# 1. VAR �� ����
model = VAR(data)

# 2. VAR(1) ����
results = model.fit(1)

# 3. A ��� ���
A = results.coefs[0]
print("Estimated A matrix:")
print(A)

# 4. ������ ��� (fitted values)
fitted = results.fittedvalues

# 5. ���� vs ������ �ð�ȭ
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

# 6. �׷��� ����
plt.savefig(r'C:\Users\ann\Desktop\BK\var_fit_results.png')
plt.show()

# 7. Fitted values�� ������ ����
fitted.to_excel(r'C:\Users\ann\Desktop\BK\var_fitted_values.xlsx')