import numpy as np

#firstly solving the Ricatti equation.
def solve_are(alpha):
    A   = np.array([[0.0, 1.0], [1.0, -alpha]])
    BBT = np.array([[0.0, 0.0], [0.0, 1.0]])
    Q   = np.eye(2)
    H   = np.block([[A, -BBT], [-Q, -A.T]])

    vals, vecs = np.linalg.eig(H)
    mask = vals.real < -1e-10
    if mask.sum() != 2:
        raise RuntimeError(
            f"Expected 2 stable Hamiltonian eigenvalues, got {mask.sum()}.\n"
            "Ensure α > 0 so the system is stabilisable.")

    V  = vecs[:, mask]
    S  = np.real(V[2:] @ np.linalg.inv(V[:2]))
    return 0.5 * (S + S.T)

#state costate equation
def pmp_rhs(y, alpha):
    """ẏ = f(y),   y = [θ, φ, λ₁, λ₂]"""
    th, ph, l1, l2 = y
    s, c = np.sin(th), np.cos(th)
    return np.array([
        ph,
        s  - alpha*ph - l2*c*c,
        -s - l2*c     - l2*l2*s*c,
        -ph - l1      + alpha*l2,
    ])


# ═══════════════════════════════════════════════════════════════
# 3.  Fixed-step RK4
# ═══════════════════════════════════════════════════════════════

def rk4(alpha, y0, T, n):
    """
    Integrate  ẏ = f(y)  from 0 to T with n equal steps.
    Returns (t, Y, ok) where ok=False if overflow/NaN occurs.
    """
    h = T / n
    t = np.linspace(0.0, T, n + 1)
    Y = np.empty((4, n + 1));  Y[:, 0] = y0
    y = y0.astype(float).copy()

    for k in range(n):
        k1 = pmp_rhs(y,             alpha)
        k2 = pmp_rhs(y + 0.5*h*k1, alpha)
        k3 = pmp_rhs(y + 0.5*h*k2, alpha)
        k4 = pmp_rhs(y +     h*k3, alpha)
        y  = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        if not np.all(np.isfinite(y)) or np.max(np.abs(y)) > 1e6:
            return t, Y, False

        Y[:, k+1] = y

    return t, Y, True


# ═══════════════════════════════════════════════════════════════
# 4.  Forward shooting residual
#     F(λ₀) = λ(T) − S·x(T)  →  0  on the stable manifold
# ═══════════════════════════════════════════════════════════════

def shoot_residual(lam0, x0, alpha, S, T, n):
    y0 = np.array([x0[0], x0[1], lam0[0], lam0[1]])
    t, Y, ok = rk4(alpha, y0, T, n)
    if not ok:
        return np.array([1e6, 1e6])
    xT, lamT = Y[:2, -1], Y[2:, -1]
    return lamT - S @ xT          # terminal costate condition


# ═══════════════════════════════════════════════════════════════
# 5.  Newton with central-diff Jacobian + Armijo line search
# ═══════════════════════════════════════════════════════════════

def newton(F, lam0, tol=1e-9, max_iter=20):
    x  = lam0.astype(float).copy()
    Fx = F(x)

    for _ in range(max_iter):
        nrm = float(np.linalg.norm(Fx))
        if nrm < tol:
            break

        # Central-difference Jacobian (fresh each iteration)
        eps = max(1e-7, 1e-5 * np.linalg.norm(x))
        J   = np.zeros((2, 2))
        for i in range(2):
            xp = x.copy();  xp[i] += eps
            xm = x.copy();  xm[i] -= eps
            Fp, Fm = F(xp), F(xm)
            if np.all(np.isfinite(Fp + Fm)):
                J[:, i] = (Fp - Fm) / (2 * eps)
            else:
                J[:, i] = (F(xp) - Fx) / eps

        try:
            dx = np.linalg.solve(J + 1e-12 * np.eye(2), -Fx)
        except np.linalg.LinAlgError:
            dx = -0.01 * Fx

        # Armijo backtracking line search
        step = 1.0
        for _ in range(15):
            Ft = F(x + step * dx)
            if np.all(np.isfinite(Ft)) and np.linalg.norm(Ft) < nrm * (1 - 1e-4*step):
                break
            step *= 0.5

        x  = x + step * dx
        Fx = F(x)

    return x, float(np.linalg.norm(Fx))


# ═══════════════════════════════════════════════════════════════
# 6.  T-continuation:  T_min → T_max  (warm-started Newton)
#
#     Why this works:
#       • At T_min≈0.5s: exp(σ·T)≈2, Newton converges in 1–2 steps.
#       • At each step: ΔT small → previous solution is a good warm start.
#       • At T_max: x(T)≈0, so λ(T)=S·x(T) is accurate.
# ═══════════════════════════════════════════════════════════════

def t_continuation(x0, alpha, S, T_min, T_max, K_steps, n_per_s):
    """
    Solve  F(λ₀; T) = λ(T) − S·x(T) = 0  for increasing T.

    Parameters
    ----------
    T_min, T_max : float — horizon range
    K_steps      : int   — number of continuation steps
    n_per_s      : int   — integration steps per second

    Returns
    -------
    lam_opt : (2,) array — converged λ(0)
    res     : float      — final residual norm
    """
    lam = S @ x0                      # LQR initial guess (accurate for T→0)

    Ts  = np.linspace(T_min, T_max, K_steps)

    for k, T in enumerate(Ts):
        n = max(40, int(T * n_per_s))  # steps proportional to horizon

        def F(lam0, T=T, n=n):
            return shoot_residual(lam0, x0, alpha, S, T, n)

        lam, res = newton(F, lam, tol=1e-9, max_iter=20)

        # Progress every few steps
        if (k + 1) % max(1, K_steps // 6) == 0 or k == K_steps - 1:
            flag = "ok" if res < 1e-6 else ("~" if res < 1e-3 else "!!")
            print(f"    [{flag}] T={T:.2f}s  n={n}  "
                  f"‖F‖={res:.2e}  λ=[{lam[0]:+.4f}, {lam[1]:+.4f}]")

        if res > 0.5:
            # Large residual — try LQR restart at this T
            lam_lqr = S @ x0
            lam_new, res_new = newton(F, lam_lqr, tol=1e-9, max_iter=20)
            if res_new < res:
                lam, res = lam_new, res_new

    return lam, res


# ═══════════════════════════════════════════════════════════════
# 7.  Accumulated cost  J = ∫ L dt  (trapezoid)
# ═══════════════════════════════════════════════════════════════

def accumulated_cost(t, Y):
    th, ph, _, l2 = Y[0], Y[1], Y[2], Y[3]
    u  = -l2 * np.cos(th)
    L  = 0.5*ph**2 + (1.0 - np.cos(th)) + 0.5*u**2
    return float(np.sum(0.5 * (L[:-1] + L[1:]) * np.diff(t)))


# ═══════════════════════════════════════════════════════════════
# 8.  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    SEP = "=" * 64
    print(SEP)
    print("  Pendulum Optimal Control  (numpy only, T-continuation shooting)")
    print("  L = ½φ² + (1−cosθ) + ½u²")
    print(SEP)

    # ── Inputs ──────────────────────────────────────────────────────
    alpha  = 0.1
    theta0 = 0
    phi0   = 1
    x0     = np.array([theta0, phi0])

    # ── 1. ARE ──────────────────────────────────────────────────────
    print(f"\n  [1/4]  Solving ARE (Hamiltonian eigendecomposition) …")
    S = solve_are(alpha)
    K = S[1, :]
    A   = np.array([[0., 1.], [1., -alpha]])
    Acl = A - np.outer([0., 1.], K)
    poles = np.linalg.eigvals(Acl)

    print(f"         S  = | {S[0,0]:+.6f}  {S[0,1]:+.6f} |")
    print(f"               | {S[1,0]:+.6f}  {S[1,1]:+.6f} |")
    print(f"         K  = [ {K[0]:+.6f}  {K[1]:+.6f} ]")
    print(f"         CL poles: {poles[0].real:+.6f},  {poles[1].real:+.6f}")
    print(f"         LQR initial guess λ₀ = [{(S@x0)[0]:+.4f}, {(S@x0)[1]:+.4f}]")

    # ── 2. Continuation parameters ───────────────────────────────────
    slowest = float(np.min(np.abs(poles.real)))
    fastest = float(np.max(np.abs(poles.real)))

    # T_min: small enough that exp(σ·T_min) ≈ 3  →  T_min ≈ ln(3)/σ
    T_min = float(np.log(3.0) / fastest)            # ≈ 0.7 s
    # T_max: large enough that |x(T_max)| ≈ 0  →  T_max ≈ 5/|slowest pole|
    T_max = float(np.clip(5.0 / slowest, 5.0, 15.0))
    K_steps   = 18      # 18 continuation steps (smooth ramp)
    n_per_s   = 80      # 80 RK4 steps per second of integration

    print(f"\n  [2/4]  Continuation parameters:")
    print(f"         T_min = {T_min:.2f} s  →  T_max = {T_max:.2f} s  "
          f"({K_steps} steps, {n_per_s} pts/s)")
    print(f"         exp(σ_max · T_min) = {np.exp(fastest*T_min):.1f}  "
          f"(manageable for Newton)")
    print(f"         |x(T_max)| / |x₀| ≈ {np.exp(-slowest*T_max):.4f}  "
          f"(LQR terminal cond. accurate)")

    # ── 3. T-continuation + Newton ───────────────────────────────────
    print(f"\n  [3/4]  T-continuation shooting  (F(λ₀)=λ(T)−S·x(T)=0):")
    lam_opt, res = t_continuation(x0, alpha, S, T_min, T_max, K_steps, n_per_s)
    print(f"\n         Final residual ‖λ(T)−S·x(T)‖ = {res:.2e}")

    if res > 1e-4:
        print("  WARNING: residual is large.  "
              "Consider increasing K_steps or n_per_s.")

    # ── 4. Cost ──────────────────────────────────────────────────────
    print(f"\n  [4/4]  Computing accumulated cost J* …")
    y0_fwd = np.array([x0[0], x0[1], lam_opt[0], lam_opt[1]])
    n_cost = int(T_max * 300)        # fine grid for cost integration
    t_fwd, Y_fwd, ok_fwd = rk4(alpha, y0_fwd, T_max, n_cost)

    if not ok_fwd:
        # Trajectory diverged — use what we have
        print("  WARNING: forward trajectory became unstable; "
              "cost is a lower bound.")
    J = accumulated_cost(t_fwd, Y_fwd)

    l1    = float(lam_opt[0])
    l2    = float(lam_opt[1])
    u0    = -l2 * np.cos(theta0)
    lqr_v = float(x0 @ S @ x0)

    # ── Results ─────────────────────────────────────────────────────
    print()
    print(SEP)
    print("  RESULTS")
    print(SEP)
    print(f"  λ₁(0)                 =  {l1:+.8f}")
    print(f"  λ₂(0)                 =  {l2:+.8f}")
    print(f"  u*(0) = −λ₂·cosθ₀    =  {u0:+.8f}")
    print(f"  J*    (accum. cost)   =  {J:.8f}")
    print(f"  x₀ᵀ S x₀ (LQR val.)  =  {lqr_v:.8f}  ← linearised lower bound")
    print(SEP)

    return l1, l2, J


if __name__ == "__main__":
    main()