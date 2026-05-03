#include "solver.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <complex>
#include <cuda_runtime.h>

// Eigen for CPU-side ARE solve (4×4 eigendecomposition)
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

#define MAX_NEWTON_ITERS   20
#define MAX_ARMIJO_ITERS   15
#define NEWTON_TOL         1.0e-9
#define FD_REL_EPS         5.0e-6
#define FD_ABS_EPS         1.0e-7
#define BLOWUP_NORM        1.0e7
#define PHI_THRESH         4.0
#define K_T_STEPS          22     // T-continuation steps
#define MAX_PHI_STEPS      16     // max φ-continuation steps
#define PHI_STEP_SIZE      2.5    // Δφ per continuation step

// ═══════════════════════════════════════════════════════════════════
// Structs passed to GPU
// ═══════════════════════════════════════════════════════════════════

// ARE solution + continuation parameters, computed per unique α on CPU
struct AREParams {
    double S[2][2];       // ARE solution matrix
    double sigma_slow;    // |slowest CL pole|
    double sigma_fast;    // |fastest CL pole|
    double alpha;
};

// ═══════════════════════════════════════════════════════════════════
// 1. CPU: ARE solver via Hamiltonian eigendecomposition (Eigen)
//
//    ARE:  Aᵀ S + S A − S B Bᵀ S + Q = 0
//    H = [[A, −BBᵀ], [−Q, −Aᵀ]]
//    Stable eigenvectors → S = V₂ V₁⁻¹
// ═══════════════════════════════════════════════════════════════════

static AREParams solve_are_cpu(double alpha) {
    Eigen::Matrix4d H;
    //  A = [0, 1; 1, -α]
    //  BBᵀ = [0, 0; 0, 1]
    //  Q = I₂
    //
    //  H = [ A    -BBᵀ ]   = [ 0    1    0    0  ]
    //      [ -Q   -Aᵀ  ]     [ 1   -α    0   -1  ]
    //                         [-1    0    0   -1  ]
    //                         [ 0   -1   -1    α  ]
    H << 0.0,     1.0,    0.0,    0.0,
         1.0,    -alpha,  0.0,   -1.0,
        -1.0,     0.0,    0.0,   -1.0,
         0.0,    -1.0,   -1.0,    alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(H);
    auto vals = es.eigenvalues();
    auto vecs = es.eigenvectors();

    // Pick the 2 eigenvectors with Re(λ) < 0
    int stable[2];
    int count = 0;
    for (int i = 0; i < 4 && count < 2; i++) {
        if (vals(i).real() < -1.0e-10) {
            stable[count++] = i;
        }
    }
    if (count != 2) {
        fprintf(stderr, "ARE: expected 2 stable eigenvalues, got %d\n", count);
        // fallback: return identity-like S
        AREParams p{};
        p.S[0][0] = 1.0; p.S[0][1] = 0.0;
        p.S[1][0] = 0.0; p.S[1][1] = 1.0;
        p.sigma_slow = 1.0; p.sigma_fast = 1.0;
        p.alpha = alpha;
        return p;
    }

    // V = [V₁ (top 2 rows); V₂ (bottom 2 rows)]
    Eigen::Matrix2cd V1, V2;
    for (int j = 0; j < 2; j++) {
        V1(0, j) = vecs(0, stable[j]);
        V1(1, j) = vecs(1, stable[j]);
        V2(0, j) = vecs(2, stable[j]);
        V2(1, j) = vecs(3, stable[j]);
    }

    Eigen::Matrix2cd Sc = V2 * V1.inverse();

    // S should be real and symmetric; take Re and symmetrise
    AREParams p{};
    p.S[0][0] = 0.5 * (Sc(0,0).real() + Sc(0,0).real());
    p.S[0][1] = 0.5 * (Sc(0,1).real() + Sc(1,0).real());
    p.S[1][0] = p.S[0][1];
    p.S[1][1] = 0.5 * (Sc(1,1).real() + Sc(1,1).real());
    p.alpha = alpha;

    // Closed-loop poles: eigenvalues of Acl = A - B K, K = S[1,:]
    // Acl = [0, 1; 1, -α] - [0; 1][S10, S11] = [0, 1; 1-S10, -α-S11]
    double a = 0.0, b = 1.0;
    double c = 1.0 - p.S[1][0], d = -alpha - p.S[1][1];
    double tr = a + d;
    double det = a * d - b * c;
    double disc = tr * tr - 4.0 * det;

    if (disc >= 0) {
        double sq = sqrt(disc);
        double p1 = 0.5 * (tr + sq);
        double p2 = 0.5 * (tr - sq);
        p.sigma_slow = std::min(fabs(p1), fabs(p2));
        p.sigma_fast = std::max(fabs(p1), fabs(p2));
    } else {
        // Complex poles: |Re| is the same for both
        p.sigma_slow = fabs(0.5 * tr);
        p.sigma_fast = fabs(0.5 * tr);
    }

    return p;
}


// ═══════════════════════════════════════════════════════════════════
// 2. GPU: Device functions
// ═══════════════════════════════════════════════════════════════════

// ── wrap θ to [−π, π] ────────────────────────────────────────────

__device__ __forceinline__
double wrap_theta(double theta) {
    return theta - 2.0 * M_PI * round(theta / (2.0 * M_PI));
}

// ── PMP state: [θ, φ, λ₁, λ₂, J] ───────────────────────────────

struct State {
    double th, ph, l1, l2, cost;
};

__device__ __forceinline__
State pmp_rhs(State s, double alpha) {
    double sin_th = sin(s.th);
    double cos_th = cos(s.th);
    double u = -s.l2 * cos_th;

    State ds;
    ds.th   = s.ph;
    ds.ph   = sin_th - alpha * s.ph - s.l2 * cos_th * cos_th;
    ds.l1   = -sin_th - s.l2 * cos_th - s.l2 * s.l2 * sin_th * cos_th;
    ds.l2   = -s.ph - s.l1 + alpha * s.l2;
    ds.cost = 0.5 * s.ph * s.ph + (1.0 - cos_th) + 0.5 * u * u;
    return ds;
}

__device__ __forceinline__
State add_scaled(State s, State k, double a) {
    State out;
    out.th   = s.th   + a * k.th;
    out.ph   = s.ph   + a * k.ph;
    out.l1   = s.l1   + a * k.l1;
    out.l2   = s.l2   + a * k.l2;
    out.cost = s.cost  + a * k.cost;
    return out;
}

__device__ __forceinline__
State rk4_step(State s, double h, double alpha) {
    State k1 = pmp_rhs(s, alpha);
    State k2 = pmp_rhs(add_scaled(s, k1, 0.5 * h), alpha);
    State k3 = pmp_rhs(add_scaled(s, k2, 0.5 * h), alpha);
    State k4 = pmp_rhs(add_scaled(s, k3, h), alpha);

    State out;
    out.th   = s.th   + (h / 6.0) * (k1.th   + 2.0*k2.th   + 2.0*k3.th   + k4.th);
    out.ph   = s.ph   + (h / 6.0) * (k1.ph   + 2.0*k2.ph   + 2.0*k3.ph   + k4.ph);
    out.l1   = s.l1   + (h / 6.0) * (k1.l1   + 2.0*k2.l1   + 2.0*k3.l1   + k4.l1);
    out.l2   = s.l2   + (h / 6.0) * (k1.l2   + 2.0*k2.l2   + 2.0*k3.l2   + k4.l2);
    out.cost = s.cost  + (h / 6.0) * (k1.cost + 2.0*k2.cost + 2.0*k3.cost + k4.cost);
    return out;
}

// ── Integrate forward from t=0 to t=T ────────────────────────────

__device__
State integrate(double th0, double ph0, double l1_0, double l2_0,
                double alpha, double T, int n_steps) {
    State s;
    s.th = th0;  s.ph = ph0;  s.l1 = l1_0;  s.l2 = l2_0;  s.cost = 0.0;

    double h = T / (double)n_steps;

    for (int i = 0; i < n_steps; i++) {
        s = rk4_step(s, h, alpha);

        if (!isfinite(s.th) || !isfinite(s.ph) ||
            !isfinite(s.l1) || !isfinite(s.l2) ||
            fabs(s.th) > BLOWUP_NORM || fabs(s.ph) > BLOWUP_NORM ||
            fabs(s.l1) > BLOWUP_NORM || fabs(s.l2) > BLOWUP_NORM) {
            s.th = 1e10; s.ph = 1e10; s.l1 = 1e10; s.l2 = 1e10;
            s.cost = 1e10;
            return s;
        }
    }
    return s;
}

// ── Shooting residual:  F(λ₀) = λ(T) − S · wrap(x(T)) ──────────

__device__
void shoot_residual(double th0, double ph0, double l1_0, double l2_0,
                    double alpha, double T, int n_steps,
                    const double S[2][2],
                    double* r1, double* r2) {
    State s = integrate(th0, ph0, l1_0, l2_0, alpha, T, n_steps);

    if (fabs(s.th) > 1e9) {
        *r1 = 1e6;
        *r2 = 1e6;
        return;
    }

    double xT_w0 = wrap_theta(s.th);
    double xT_w1 = s.ph;

    // terminal condition: λ(T) = S · wrap(x(T))
    *r1 = s.l1 - (S[0][0] * xT_w0 + S[0][1] * xT_w1);
    *r2 = s.l2 - (S[1][0] * xT_w0 + S[1][1] * xT_w1);
}

// ── Newton solver with central-diff Jacobian + Armijo ────────────

__device__
void newton_solve(double th0, double ph0,
                  double& l1, double& l2,
                  double alpha, double T, int n_steps,
                  const double S[2][2],
                  int max_iter) {

    double r1, r2;
    shoot_residual(th0, ph0, l1, l2, alpha, T, n_steps, S, &r1, &r2);

    for (int iter = 0; iter < max_iter; iter++) {
        double nrm = sqrt(r1 * r1 + r2 * r2);
        if (nrm < NEWTON_TOL) break;

        // Adaptive FD epsilon
        double lnrm = sqrt(l1 * l1 + l2 * l2);
        double eps = fmax(FD_ABS_EPS, FD_REL_EPS * fmax(1.0, lnrm));

        // Central-difference Jacobian (5 residual evaluations)
        double r1_p1, r2_p1, r1_m1, r2_m1;
        double r1_p2, r2_p2, r1_m2, r2_m2;

        shoot_residual(th0, ph0, l1 + eps, l2, alpha, T, n_steps, S, &r1_p1, &r2_p1);
        shoot_residual(th0, ph0, l1 - eps, l2, alpha, T, n_steps, S, &r1_m1, &r2_m1);
        shoot_residual(th0, ph0, l1, l2 + eps, alpha, T, n_steps, S, &r1_p2, &r2_p2);
        shoot_residual(th0, ph0, l1, l2 - eps, alpha, T, n_steps, S, &r1_m2, &r2_m2);

        double J11, J21, J12, J22;

        if (isfinite(r1_p1 + r1_m1)) {
            J11 = (r1_p1 - r1_m1) / (2.0 * eps);
            J21 = (r2_p1 - r2_m1) / (2.0 * eps);
        } else {
            J11 = (r1_p1 - r1) / eps;
            J21 = (r2_p1 - r2) / eps;
        }

        if (isfinite(r1_p2 + r1_m2)) {
            J12 = (r1_p2 - r1_m2) / (2.0 * eps);
            J22 = (r2_p2 - r2_m2) / (2.0 * eps);
        } else {
            J12 = (r1_p2 - r1) / eps;
            J22 = (r2_p2 - r2) / eps;
        }

        // Solve J · δ = −F  via 2×2 Cramer
        double det = J11 * J22 - J12 * J21;
        double d1, d2;
        if (fabs(det) < 1e-14) {
            d1 = -0.01 * r1;
            d2 = -0.01 * r2;
        } else {
            d1 = ( J22 * (-r1) - J12 * (-r2)) / det;
            d2 = (-J21 * (-r1) + J11 * (-r2)) / det;
        }

        // Armijo backtracking line search
        double step = 1.0;
        for (int ls = 0; ls < MAX_ARMIJO_ITERS; ls++) {
            double tl1 = l1 + step * d1;
            double tl2 = l2 + step * d2;
            double tr1, tr2;
            shoot_residual(th0, ph0, tl1, tl2, alpha, T, n_steps, S, &tr1, &tr2);
            double tnrm = sqrt(tr1 * tr1 + tr2 * tr2);

            if (isfinite(tnrm) && tnrm < nrm * (1.0 - 1e-4 * step)) {
                break;
            }
            step *= 0.5;
        }

        l1 += step * d1;
        l2 += step * d2;

        if (!isfinite(l1) || !isfinite(l2)) {
            l1 = 0.0;
            l2 = 0.0;
            return;
        }

        // Re-evaluate residual for next iteration
        shoot_residual(th0, ph0, l1, l2, alpha, T, n_steps, S, &r1, &r2);
    }
}

// ═══════════════════════════════════════════════════════════════════
// 3. GPU Kernel: one thread per problem
//    Runs full T-continuation + optional φ-continuation
// ═══════════════════════════════════════════════════════════════════

__global__
void solve_kernel(
    const double* __restrict__ theta_arr,
    const double* __restrict__ phi_arr,
    const AREParams* __restrict__ are_arr,
    Result* __restrict__ result_arr,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double theta0 = theta_arr[idx];
    double phi0   = phi_arr[idx];
    AREParams are = are_arr[idx];
    double alpha  = are.alpha;

    // Copy S to local registers
    double S[2][2];
    S[0][0] = are.S[0][0]; S[0][1] = are.S[0][1];
    S[1][0] = are.S[1][0]; S[1][1] = are.S[1][1];

    double sigma_slow = are.sigma_slow;
    double sigma_fast = are.sigma_fast;

    // ── Continuation parameters ──────────────────────────────────
    double T_min = log(3.0) / sigma_fast;
    double T_max = fmin(22.0, fmax(5.0,
                        5.0 / sigma_slow + 0.35 * fabs(phi0)));

    int n_per_s = max(80, (int)(9.0 * fabs(phi0)));

    // ── Determine whether φ-continuation is needed ───────────────
    bool large_phi = fabs(phi0) > PHI_THRESH;

    // Working φ target for the seed phase
    double phi_work = large_phi
        ? fmin(PHI_THRESH, fmax(-PHI_THRESH, phi0))
        : phi0;

    // ── LQR initial guess: S · [wrap(θ₀), φ_work] ───────────────
    double eff_th = wrap_theta(theta0);
    double l1 = S[0][0] * eff_th + S[0][1] * phi_work;
    double l2 = S[1][0] * eff_th + S[1][1] * phi_work;

    // ═════════════════════════════════════════════════════════════
    // LEVEL A: T-continuation  T_min → T_max
    //   At each T step, warm-start Newton from previous solution.
    // ═════════════════════════════════════════════════════════════

    for (int k = 0; k < K_T_STEPS; k++) {
        double T = T_min + (T_max - T_min) * (double)k / (double)(K_T_STEPS - 1);
        int n_steps = max(50, (int)(T * n_per_s));

        newton_solve(theta0, phi_work, l1, l2,
                     alpha, T, n_steps, S, MAX_NEWTON_ITERS);
    }

    // ═════════════════════════════════════════════════════════════
    // LEVEL B: φ-continuation  φ_seed → φ_target  (if needed)
    //   Walk φ in steps of PHI_STEP_SIZE, Newton at T_max each step.
    // ═════════════════════════════════════════════════════════════

    if (large_phi) {
        double phi_seed = phi_work;
        int N_phi = (int)ceil(fabs(phi0 - phi_seed) / PHI_STEP_SIZE);
        if (N_phi < 3) N_phi = 3;
        if (N_phi > MAX_PHI_STEPS) N_phi = MAX_PHI_STEPS;

        for (int i = 1; i <= N_phi; i++) {
            double phi_c = phi_seed + (phi0 - phi_seed) * (double)i / (double)N_phi;

            // Recompute T_max for this φ
            double T_c = fmin(22.0, fmax(5.0,
                              5.0 / sigma_slow + 0.35 * fabs(phi_c)));
            int n_c = max(120, (int)(T_c * fmax(8.0, 1.5 * fabs(phi_c))));
            n_c = min(n_c, 3000);

            newton_solve(theta0, phi_c, l1, l2,
                         alpha, T_c, n_c, S, 8);
        }

        // Final refinement at full resolution
        int n_fine = max(300, (int)(T_max * n_per_s));
        newton_solve(theta0, phi0, l1, l2,
                     alpha, T_max, n_fine, S, MAX_NEWTON_ITERS);
    }

    // ═════════════════════════════════════════════════════════════
    // Cost computation:  J = ∫₀ᵀ L dt  +  LQR tail
    // ═════════════════════════════════════════════════════════════

    int n_cost = max(600, (int)(T_max * fmax((double)n_per_s, 80.0)));
    State sf = integrate(theta0, phi0, l1, l2, alpha, T_max, n_cost);

    double J = sf.cost;

    // LQR tail:  J_tail = wrap(x(T))ᵀ S wrap(x(T))
    if (fabs(sf.th) < 1e9) {
        double xw0 = wrap_theta(sf.th);
        double xw1 = sf.ph;
        J += S[0][0]*xw0*xw0 + 2.0*S[0][1]*xw0*xw1 + S[1][1]*xw1*xw1;
    }

    result_arr[idx].l1   = l1;
    result_arr[idx].l2   = l2;
    result_arr[idx].cost = J;
}


// ═══════════════════════════════════════════════════════════════════
// 4. Host-side solve_many: CPU ARE + GPU kernel launch
// ═══════════════════════════════════════════════════════════════════

std::vector<Result> solve_many(
    const std::vector<double>& theta,
    const std::vector<double>& phi,
    const std::vector<double>& alpha
) {
    int N = (int)theta.size();
    std::vector<Result> results(N);

    // ── CPU: solve ARE for each problem ──────────────────────────
    //    (In practice, α is often shared — could cache, but this is
    //     negligible cost vs the GPU kernel.)
    std::vector<AREParams> are_params(N);
    for (int i = 0; i < N; i++) {
        are_params[i] = solve_are_cpu(alpha[i]);
    }

    // ── Allocate device memory ───────────────────────────────────
    double*     d_theta   = nullptr;
    double*     d_phi     = nullptr;
    AREParams*  d_are     = nullptr;
    Result*     d_results = nullptr;

    cudaMalloc(&d_theta,   N * sizeof(double));
    cudaMalloc(&d_phi,     N * sizeof(double));
    cudaMalloc(&d_are,     N * sizeof(AREParams));
    cudaMalloc(&d_results, N * sizeof(Result));

    cudaMemcpy(d_theta, theta.data(),       N * sizeof(double),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi,   phi.data(),         N * sizeof(double),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_are,   are_params.data(),  N * sizeof(AREParams), cudaMemcpyHostToDevice);

    // ── Launch kernel ────────────────────────────────────────────
    int threads = 128;
    int blocks  = (N + threads - 1) / threads;

    solve_kernel<<<blocks, threads>>>(
        d_theta, d_phi, d_are, d_results, N
    );

    cudaDeviceSynchronize();

    // ── Copy results back ────────────────────────────────────────
    cudaMemcpy(results.data(), d_results, N * sizeof(Result), cudaMemcpyDeviceToHost);

    cudaFree(d_theta);
    cudaFree(d_phi);
    cudaFree(d_are);
    cudaFree(d_results);

    return results;
}

Result solve(double theta, double phi, double alpha) {
    return solve_many({theta}, {phi}, {alpha})[0];
}
