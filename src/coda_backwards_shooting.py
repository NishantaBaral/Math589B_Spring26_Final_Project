import numpy as np
from scipy.linalg import expm

# ═══════════════════════════════════════════════════════════════
# 1. ARE Solver
# ═══════════════════════════════════════════════════════════════
def solve_are(alpha):
    A   = np.array([[0.0, 1.0], [1.0, -alpha]])
    BBT = np.array([[0.0, 0.0], [0.0, 1.0]])
    Q   = np.eye(2)
    H   = np.block([[A, -BBT], [-Q, -A.T]])
    vals, vecs = np.linalg.eig(H)
    mask = vals.real < -1e-10
    V = vecs[:, mask]
    S = np.real(V[2:] @ np.linalg.inv(V[:2]))
    return 0.5 * (S + S.T)

# ═══════════════════════════════════════════════════════════════
# 2. PMP right-hand side
# ═══════════════════════════════════════════════════════════════
def pmp_rhs(y, alpha):
    th, ph, l1, l2 = y
    s, c = np.sin(th), np.cos(th)
    return np.array([
        ph,
        s - alpha * ph - l2 * c * c,
        -s - l2 * c - l2 * l2 * s * c,
        -ph - l1 + alpha * l2,
    ])

# ═══════════════════════════════════════════════════════════════
# 3. Backward RK4 with cost accumulation
#
#    Integrates dy/ds = -f(y), s ∈ [0, T_back].
#    In forward time this traces the trajectory from x₀ (at s=T_back)
#    back to the patch ξ (at s=0).
#    Cost is accumulated along the path (same points, same integral).
# ═══════════════════════════════════════════════════════════════
def rk4_backward(y0, alpha, T, n):
    h = T / n
    y = y0.astype(float).copy()
    J = 0.0
    for k in range(n):
        th, ph, l1, l2 = y
        c = np.cos(th)
        u = -l2 * c
        L = 0.5 * ph**2 + (1 - c) + 0.5 * u**2

        k1 = -pmp_rhs(y, alpha)
        k2 = -pmp_rhs(y + 0.5 * h * k1, alpha)
        k3 = -pmp_rhs(y + 0.5 * h * k2, alpha)
        k4 = -pmp_rhs(y + h * k3, alpha)
        y = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        th2, ph2, _, l22 = y
        c2 = np.cos(th2)
        u2 = -l22 * c2
        L2 = 0.5 * ph2**2 + (1 - c2) + 0.5 * u2**2
        J += 0.5 * h * (L + L2)

        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e8:
            return y, J, False
    return y, J, True

# ═══════════════════════════════════════════════════════════════
# 4. Newton solver (central-diff Jacobian + Armijo)
# ═══════════════════════════════════════════════════════════════
def newton(F, x0, tol=1e-12, max_iter=25):
    x = x0.astype(float).copy()
    Fx = F(x)
    for it in range(max_iter):
        nrm = float(np.linalg.norm(Fx))
        if nrm < tol:
            break
        eps = max(1e-8, 1e-6 * max(1.0, np.linalg.norm(x)))
        J = np.zeros((2, 2))
        for i in range(2):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            Fp, Fm = F(xp), F(xm)
            if np.all(np.isfinite(Fp + Fm)):
                J[:, i] = (Fp - Fm) / (2 * eps)
            else:
                J[:, i] = (F(xp) - Fx) / eps
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx = -0.01 * Fx
        step = 1.0
        for _ in range(12):
            Ft = F(x + step * dx)
            if np.all(np.isfinite(Ft)) and np.linalg.norm(Ft) < nrm * (1 - 1e-4 * step):
                break
            step *= 0.5
        x = x + step * dx
        Fx = F(x)
    return x, float(np.linalg.norm(Fx))

# ═══════════════════════════════════════════════════════════════
# 5. Full solver
#
#    Backward shooting from the stable-manifold patch:
#    1. Near origin, stable manifold ≈ λ = S·x (LQR).
#    2. Pick small ξ, set patch point y = (ξ, S·ξ).
#    3. Integrate PMP backward for T_back seconds.
#       Off-manifold errors DECAY → self-correcting.
#    4. Newton finds ξ s.t. x_backward(T_back) = x₀.
#    5. λ(0) = costate at backward endpoint.
#    6. Cost = backward path integral + ξᵀSξ (LQR tail).
# ═══════════════════════════════════════════════════════════════
def solve(theta0, phi0, alpha):
    S = solve_are(alpha)
    x0 = np.array([theta0, phi0])

    # Trivial case
    if np.linalg.norm(x0) < 1e-15:
        lam = S @ x0
        return lam[0], lam[1], float(x0 @ S @ x0)

    # Closed-loop system for initial guess
    A   = np.array([[0., 1.], [1., -alpha]])
    Acl = A - np.outer([0., 1.], S[1, :])
    poles = np.linalg.eigvals(Acl)
    sigma_slow = min(abs(p.real) for p in poles)

    # Backward horizon: long enough for |ξ| to be tiny
    T_back = max(6.0, 5.0 / sigma_slow + 0.3 * abs(phi0))
    T_back = min(T_back, 20.0)
    n_back = max(5000, int(T_back * 800))

    # Initial guess: ξ = exp(Acl·T_back)·x₀ (tiny near origin)
    xi_init = expm(Acl * T_back) @ x0

    # Newton: find ξ s.t. backward(ξ, S·ξ) lands on x₀
    def F(xi):
        lam_xi = S @ xi
        y_patch = np.array([xi[0], xi[1], lam_xi[0], lam_xi[1]])
        y_end, _, ok = rk4_backward(y_patch, alpha, T_back, n_back)
        if not ok:
            return np.array([1e6, 1e6])
        return y_end[:2] - x0

    xi_opt, res = newton(F, xi_init)

    # Final backward integration: extract λ(0) and cost
    lam_xi = S @ xi_opt
    y_patch = np.array([xi_opt[0], xi_opt[1], lam_xi[0], lam_xi[1]])
    y_end, J_path, ok = rk4_backward(y_patch, alpha, T_back, n_back)

    l1 = float(y_end[2])
    l2 = float(y_end[3])

    # Total cost = path integral (x₀ → ξ) + LQR tail (ξ → origin)
    J_tail = float(xi_opt @ S @ xi_opt)
    J = J_path + J_tail

    return l1, l2, J

# ═══════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════
def main():
    alpha  = 0.2
    theta0 = 0
    phi0   = 1
    l1, l2, J = solve(theta0, phi0, alpha)
    print(f"{l1:.10f} {l2:.10f} {J:.10f}")

if __name__ == "__main__":
    main()