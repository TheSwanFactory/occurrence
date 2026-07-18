import numpy as np

# --- rebuild the verified sedenion structure tensor (same as independent_check.py) ---
def cd_double(mult_table):
    n = mult_table.shape[0]
    dim = 2*n
    C = np.zeros((dim, dim, dim))
    conj_sign = np.array([1.0] + [-1.0]*(n-1))
    def basis_pair(idx):
        if idx < n:
            a = np.zeros(n); a[idx] = 1.0; b = np.zeros(n)
        else:
            a = np.zeros(n); b = np.zeros(n); b[idx-n] = 1.0
        return a,b
    def mult_base(u, v):
        out = np.zeros(n)
        for i in range(n):
            if u[i]==0: continue
            for j in range(n):
                if v[j]==0: continue
                out += u[i]*v[j]*mult_table[i,j]
        return out
    def conj_base(u): return u*conj_sign
    for I in range(dim):
        a1,b1 = basis_pair(I)
        for J in range(dim):
            a2,b2 = basis_pair(J)
            real_part = mult_base(a1,a2) - mult_base(conj_base(b2), b1)
            imag_part = mult_base(b2,a1) + mult_base(b1, conj_base(a2))
            for k in range(n):
                if abs(real_part[k])>1e-15: C[I,J,k] += real_part[k]
                if abs(imag_part[k])>1e-15: C[I,J,n+k] += imag_part[k]
    return C

C = np.ones((1,1,1))
for _ in range(4):
    C = cd_double(C)
n = C.shape[0]

def multiply(x, y, C):
    out = np.zeros(n)
    for i in range(n):
        if x[i]==0: continue
        out += x[i]*(y @ C[i])
    return out

def conj(x):
    s = np.ones(n); s[1:] = -1.0
    return x*s

def norm(x): return np.sqrt(np.dot(x,x))

# crack (verified 84 elements)
crack = []
for i in range(1,8):
    for j in range(1,8):
        for s in (1.0,-1.0):
            v = np.zeros(n); v[i]=1.0/np.sqrt(2); v[8+j]=s/np.sqrt(2)
            Lv = np.zeros((n,n))
            for k in range(n):
                ek = np.zeros(n); ek[k]=1.0
                Lv[:,k] = multiply(v, ek, C)
            if np.linalg.matrix_rank(Lv, tol=1e-9) < n:
                crack.append(v)
crack = np.array(crack)
assert len(crack) == 84

# === TEST A: exact trajectory-level coupling via conjugation (does Prop 4.2 hold pointwise?) ===
rng = np.random.default_rng(42)
x0 = crack[rng.integers(84)].copy()  # arbitrary unit start (any unit vector works; use a crack pt for concreteness... actually should be generic, not on crack)
x0 = rng.normal(size=n); x0 /= norm(x0)

T = 200
z_seq = crack[rng.integers(0, 84, size=T)]

# standard chain: x_{t+1} = normalize(z_t * x_t)   [z on LEFT, x retained on RIGHT]
x = x0.copy()
xs_standard = [x.copy()]
for t in range(T):
    x = multiply(z_seq[t], x, C)
    x = x / norm(x)
    xs_standard.append(x.copy())

# swapped chain: y_{t+1} = normalize(y_t * z'_t) where z'_t = conj(z_t)  [retained on LEFT]
y = conj(x0)
ys_swapped = [y.copy()]
for t in range(T):
    zp = conj(z_seq[t])
    y = multiply(y, zp, C)
    y = y / norm(y)
    ys_swapped.append(y.copy())

# Prop 4.2 claims: y_t should equal conj(x_t) exactly, for all t, if conj(ab)=conj(b)conj(a) and
# conj is an isometry (both true for sedenion conjugation)
max_dev = 0.0
for t in range(T+1):
    dev = norm(ys_swapped[t] - conj(xs_standard[t]))
    max_dev = max(max_dev, dev)
print(f"[Test A] max_t || y_t - conj(x_t) || over {T} steps: {max_dev:.3e}")
print("         (if ~0: Prop 4.2's exact coupling holds pointwise, trajectory by trajectory)")

# spine share is conjugation-invariant: check directly
def spine_share(v): return v[0]**2 + v[8]**2
dev_spine = max(abs(spine_share(xs_standard[t]) - spine_share(ys_swapped[t])) for t in range(T+1))
print(f"[Test A] max_t |spine(x_t) - spine(y_t)|: {dev_spine:.3e}  (should be ~0 by conj-invariance of spine share)")

# === Does conjugation permute the crack onto itself (so mu is conj-invariant)? ===
crack_set = set(tuple(np.round(v, 9)) for v in crack)
conj_crack = np.array([conj(v) for v in crack])
conj_in_crack = sum(1 for v in conj_crack if tuple(np.round(v,9)) in crack_set)
print(f"\n[Test B] conj(crack) subset of crack: {conj_in_crack}/84 points land back in the crack")

# === Empirical Monte Carlo comparison at MUCH higher N, T, multiple seeds, both orientations ===
def run_chain(x0, z_indices, retained_on_left):
    x = x0.copy()
    for zi in z_indices:
        z = crack[zi]
        if retained_on_left:
            x = multiply(x, z, C)   # x * z  (swapped: retained on left)
        else:
            x = multiply(z, x, C)   # z * x  (standard: retained on right)
        x = x / norm(x)
    return x

def spine_share(v): return v[0]**2 + v[8]**2

def estimate_spine_share(retained_on_left, N, T, burn_in, seed):
    rng = np.random.default_rng(seed)
    shares = []
    for traj in range(N):
        x0 = rng.normal(size=n); x0 /= norm(x0)
        z_idx = rng.integers(0, 84, size=T)
        x = x0.copy()
        for t in range(T):
            z = crack[z_idx[t]]
            if retained_on_left:
                x = multiply(x, z, C)
            else:
                x = multiply(z, x, C)
            x = x / norm(x)
            if t >= burn_in:
                shares.append(spine_share(x))
    shares = np.array(shares)
    return shares.mean(), shares.std()/np.sqrt(len(shares))

print("\n[Test C] Empirical spine share, matched to cabarius methodology (N=8000,T=1200,burn=200), 3 seeds each:")
for seed in (1,2,3):
    m_std, se_std = estimate_spine_share(False, N=8000, T=1200, burn_in=200, seed=seed)
    m_swp, se_swp = estimate_spine_share(True,  N=8000, T=1200, burn_in=200, seed=seed+100)
    diff = m_std - m_swp
    combined_se = np.sqrt(se_std**2 + se_swp**2)
    print(f"  seed {seed}: standard = {m_std:.6f} +/- {se_std:.6f}   swapped = {m_swp:.6f} +/- {se_swp:.6f}   "
          f"diff = {diff:+.6f}  ({diff/combined_se:+.2f} sigma)")

print("\n[Test D] Reported Test-14-D3 scale under the gauge-equivalent discrete design (N=1000, T=30):")
m_std_orig, se_std_orig = estimate_spine_share(False, N=1000, T=30, burn_in=0, seed=7)
m_swp_orig, se_swp_orig = estimate_spine_share(True,  N=1000, T=30, burn_in=0, seed=7)
print(f"  standard = {m_std_orig:.5f} +/- {se_std_orig:.5f}   swapped = {m_swp_orig:.5f} +/- {se_swp_orig:.5f}")
print(f"  (the historical run used a different continuum, endpoint-only, uncoupled protocol)")
