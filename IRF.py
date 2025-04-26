### 1. ���� �����ͷ� VAR(1) �����ϰ� IRF ����

# VAR(1) ����
model = VAR(data)
results = model.fit(1)

# IRF ���
irf_actual = results.irf(10)

# IRF �׸��� �� ����
fig1 = irf_actual.plot(orth=False)
plt.suptitle('IRF from Actual Estimated A (BK NOT satisfied)')
plt.tight_layout()
fig1.savefig(r'C:\Users\ann\Desktop\BK\irf_actual.png')
plt.show()

### 2. BK ���� �����ϴ� ������ A ����� IRF ����

# BK ���� ���� A ����
A_bk = np.array([
    [0.7, 0.2],
    [0.3, 1.1]
])

# ������ ������ �ùķ��̼�
n_periods = 50
np.random.seed(0)
x_simulated = np.zeros((n_periods, 2))
shock = np.random.normal(size=(n_periods, 2)) * 0.1

for t in range(1, n_periods):
    x_simulated[t] = A_bk @ x_simulated[t-1] + shock[t]

# �ùķ��̼� �����ͷ� VAR(1) ����
simulated_data = pd.DataFrame(x_simulated, columns=['output_gap', 'YoY_inflation'])

model_sim = VAR(simulated_data)
results_sim = model_sim.fit(1)

# IRF ���
irf_bk = results_sim.irf(10)

# IRF �׸��� �� ����
fig2 = irf_bk.plot(orth=False)
plt.suptitle('IRF from Artificial BK-satisfied A')
plt.tight_layout()
fig2.savefig(r'C:\Users\ann\Desktop\BK\irf_bk_satisfied.png')
plt.show()