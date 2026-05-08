import numpy as np
from scipy.linalg import expm

#Ricatti eqn solver
# H = [[ A,-B B^T ][-Q,-A^T]] and the stable eigenspace of H gives the Ricatti solution
def solve_are(alpha):
    A   = np.array([[0.0, 1.0], [1.0, -alpha]])
    BBT = np.array([[0.0, 0.0], [0.0, 1.0]])
    Q   = np.eye(2)
    H   = np.block([[A, -BBT], [-Q, -A.T]])
    vals, vecs = np.linalg.eig(H)
    mask = vals.real < -1e-10
    V = vecs[:, mask]
    #first two rows of V are x part, last two rows are lambda part. So S = lambda*x^{-1}
    S = np.real(V[2:] @ np.linalg.inv(V[:2]))
    #force S to be symmetric
    return 0.5 * (S + S.T)

#state costate dynamics. Good ol' PMP. 
def pmp_rhs(y, alpha):
    th, ph, l1, l2 = y
    sin, cos = np.sin(th), np.cos(th)
    return np.array([
        ph,
        sin - alpha * ph - l2 * cos * cos,
        -sin - l2 * cos - l2 * l2 * sin * cos,
        -ph - l1 + alpha * l2,
    ])

#rk4 backwards
def rk4_backward(y0, alpha, T, n):
    #here y0 is the patch point on the stable manifold
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

#Newton+Armijo
#F is the resiual fn. We want to solve F(x) = 0 where x is a point on the stable manifold patch. 
#F(x) = (theta \tilde - theta0, phi \tilde - phi0) where
#  So we want to adjust the patch coordinate x until the backward-integrated stable-manifold point lands on give initial condition (theta0, phi0).
def newton(F, x0, tol=1e-12, max_iter=25):
    x = x0.astype(float).copy()
    Fx = F(x)
    for it in range(max_iter):
        nrm = float(np.linalg.norm(Fx))
        #square the norm and see if jt works 
        if nrm < tol:
            break
        #we are choosing a finite-difference step size epsilon for approximating the Jacobian.
        eps = max(1e-8, 1e-6 * max(1.0, np.linalg.norm(x)))
        J = np.zeros((2, 2))
        #loop over the two dimensions of x to compute the Jacobian.
        #  We perturb each coordinate of x by epsilon and evaluate F at the perturbed points to estimate the Jacobian.
        for i in range(2):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            #evlaute the residual at two perturbed points to get a finite-difference approximation of the Jacobian column.
            Fp, Fm = F(xp), F(xm)
            #use central difference if Fp and Fm dont blow up, otherwise use forward difference to avoid NaNs.
            if np.all(np.isfinite(Fp + Fm)):
                J[:, i] = (Fp - Fm) / (2 * eps)
            else:
                J[:, i] = (F(xp) - Fx) / eps
        try:
            dx = np.linalg.solve(J, -Fx) #solve J dx = -Fx to get the Newton step
        except np.linalg.LinAlgError:
            #some error protection against singular Jacobian
            dx = -0.01 * Fx
        #Armijo line search: we want to ensure that the new point x + step*dx actually reduces the norm of the residual F.
        step = 1.0
        #do Armijo backtracking with 12 step sizes
        for _ in range(12):
            Ft = F(x + step * dx)
            if np.all(np.isfinite(Ft)) and np.linalg.norm(Ft) < nrm * (1 - 1e-4 * step):
                break
            step *= 0.5
        x = x + step * dx #Actal update Newton w/ accepted step size
        Fx = F(x) #evaluate the residual at the new point for the next iteration
    return x, float(np.linalg.norm(Fx))

# Full solver
def solve(theta0, phi0, alpha):
    S = solve_are(alpha)
    x0 = np.array([theta0, phi0])

    # Trivial case solve ARE directly
    if np.linalg.norm(x0) < 1e-15:
        lam = S @ x0
        return lam[0], lam[1], 0.5*float(x0 @ S @ x0)

    # Closed-loop system for initial guess
    A   = np.array([[0., 1.], [1., -alpha]])
    #Acl​=A−BBTS. In our case, BBT = [[0, 0], [0, 1]], so BBT @ S just picks out the second row of S and zeros out the first row.
    # So Acl is A with the second row modified by subtracting S[1, :].
    Acl = A - np.outer([0., 1.], S[1, :])
    #eigenvalues
    poles = np.linalg.eigvals(Acl)
    sigma_slow = min(abs(p.real) for p in poles)

    #choose backward integration time

    T_back = max(6.0, 5.0 / sigma_slow + 0.3 * abs(phi0))
    T_back = min(T_back, 20.0)
    n_back = max(5000, int(T_back * 800))

    # Computes an initial guess for the near-origin patch coordinate
    xi_init = expm(Acl * T_back) @ x0

    # define theshooting residual function
    def F(xi):
        #Compute the local costate on the Riccati patch
        lam_xi = S @ xi
        #build the pmp point 
        y_patch = np.array([xi[0], xi[1], lam_xi[0], lam_xi[1]])
        #Integrates this patch point backward through the PMP system.
        y_end, _, ok = rk4_backward(y_patch, alpha, T_back, n_back)
        if not ok:
            return np.array([1e6, 1e6])
        return y_end[:2] - x0
    
    #run newton
    xi_opt, res = newton(F, xi_init)

    # Final backward integration: extract lambda(0) and cost
    lam_xi = S @ xi_opt
    #Build the refined patch point on the stable manifold  
    y_patch = np.array([xi_opt[0], xi_opt[1], lam_xi[0], lam_xi[1]])
    #integrate back one last time
    y_end, J_path, ok = rk4_backward(y_patch, alpha, T_back, n_back)

    #extract intial cosate 
    l1 = float(y_end[2])
    l2 = float(y_end[3])

    # compute cost + add tail cost straight from Ricatti eqn. 
    J_tail = 0.5*float(xi_opt @ S @ xi_opt)
    J = J_path + J_tail

    return l1, l2, J

#driver code 
def main():
    alpha  = 0.2
    theta0 = 2
    phi0   = -2
    l1, l2, J = solve(theta0, phi0, alpha)
    print(f"{l1:.10f} {l2:.10f} {J:.10f}")

if __name__ == "__main__":
    main()