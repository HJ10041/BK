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