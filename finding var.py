from statsmodels.tsa.api import VAR
import numpy as np

# 2. 기본 설정
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