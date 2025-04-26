# 1. �� �Ҿ����� A ����
A_nosol = np.array([
    [1.5, 0.8],
    [0.6, 1.3]
])

# 2. �ùķ��̼�
n_periods = 50
np.random.seed(42)
x_nosol = np.zeros((n_periods, 2))
shock = np.random.normal(size=(n_periods, 2)) * 0.1

for t in range(1, n_periods):
    x_nosol[t] = A_nosol @ x_nosol[t-1] + shock[t]

sim_data = pd.DataFrame(x_nosol, columns=['output_gap', 'YoY_inflation'])

# 3. VAR(1) ����
model = VAR(sim_data)
results = model.fit(1)

# 4. ������ Ȯ��
A_est = results.coefs[0]
eigvals = np.linalg.eigvals(A_est)
print("Estimated A matrix:")
print(A_est)
print("Eigenvalues:", eigvals)

# 5. IRF ��� �� ����
irf = results.irf(10)
fig = irf.plot(orth=False)
plt.suptitle('IRF from Strongly Unstable A (BK Violated: NO SOLUTION)')
plt.tight_layout()
fig.savefig(r'C:\Users\ann\Desktop\BK\irf_bk_strong_nosolution.png')
plt.show()