#include "solver.hpp"
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>

// ═══════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════

static constexpr int    MAX_NEWTON      = 25;
static constexpr int    MAX_ARMIJO      = 12;
static constexpr double NEWTON_TOL      = 1.0e-12;
static constexpr double BLOWUP          = 1.0e8;
static constexpr double FD_ABS          = 1.0e-8;
static constexpr double FD_REL          = 1.0e-6;

// ═══════════════════════════════════════════════════════════════════
// AREParams: CPU → GPU transfer struct
// ═══════════════════════════════════════════════════════════════════

struct AREParams {
    double S[2][2];
    double sigma_slow;
    double sigma_fast;
    double alpha;
};

// ═══════════════════════════════════════════════════════════════════
// 1. CPU: ARE solver + initial guess computation
// ═══════════════════════════════════════════════════════════════════

static AREParams solve_are_cpu(double alpha) {
    Eigen::Matrix4d H;
    H << 0.0,    1.0,   0.0,   0.0,
         1.0,   -alpha, 0.0,  -1.0,
        -1.0,    0.0,   0.0,  -1.0,
         0.0,   -1.0,  -1.0,   alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(H);
    auto vals = es.eigenvalues();
    auto vecs = es.eigenvectors();

    int stable[2]; int cnt = 0;
    for (int i = 0; i < 4 && cnt < 2; i++)
        if (vals(i).real() < -1e-10) stable[cnt++] = i;

    AREParams p{};
    p.alpha = alpha;

    if (cnt != 2) {
        p.S[0][0] = p.S[1][1] = 1.0;
        p.sigma_slow = p.sigma_fast = 1.0;
        return p;
    }

    Eigen::Matrix2cd V1, V2;
    for (int j = 0; j < 2; j++) {
        V1(0,j) = vecs(0, stable[j]); V1(1,j) = vecs(1, stable[j]);
        V2(0,j) = vecs(2, stable[j]); V2(1,j) = vecs(3, stable[j]);
    }
    Eigen::Matrix2cd Sc = V2 * V1.inverse();
    p.S[0][0] = Sc(0,0).real();
    p.S[0][1] = 0.5 * (Sc(0,1).real() + Sc(1,0).real());
    p.S[1][0] = p.S[0][1];
    p.S[1][1] = Sc(1,1).real();

    // Closed-loop poles
    double tr = -(alpha + p.S[1][1]);  // trace of Acl
    double det = -(1.0 - p.S[1][0]);   // det of Acl (careful with signs)
    // Acl = [0, 1; 1-S10, -α-S11]
    // tr = 0 + (-α-S11) = -α-S11
    // det = 0*(-α-S11) - 1*(1-S10) = -(1-S10)
    double disc = tr*tr - 4.0*det;
    if (disc >= 0) {
        double sq = sqrt(disc);
        p.sigma_slow = fmin(fabs(0.5*(tr+sq)), fabs(0.5*(tr-sq)));
        p.sigma_fast = fmax(fabs(0.5*(tr+sq)), fabs(0.5*(tr-sq)));
    } else {
        p.sigma_slow = p.sigma_fast = fabs(0.5*tr);
    }

    return p;
}

// Compute xi_guess = exp(Acl * T) * x0
static void compute_xi_guess(const AREParams& p, double T,
                             double x0, double x1,
                             double& xi0, double& xi1) {
    Eigen::Matrix2d Acl;
    Acl(0,0) = 0.0;             Acl(0,1) = 1.0;
    Acl(1,0) = 1.0 - p.S[1][0]; Acl(1,1) = -p.alpha - p.S[1][1];

    Eigen::Matrix2d eAT = (Acl * T).exp();
    Eigen::Vector2d x0v; x0v << x0, x1;
    Eigen::Vector2d xi = eAT * x0v;
    xi0 = xi(0);
    xi1 = xi(1);
}


// ═══════════════════════════════════════════════════════════════════
// 2. GPU device functions
// ═══════════════════════════════════════════════════════════════════

struct State { double th, ph, l1, l2, cost; };

__device__ __forceinline__
void pmp_deriv(double th, double ph, double l1, double l2, double alpha,
               double& dth, double& dph, double& dl1, double& dl2, double& dcost) {
    double si = sin(th), co = cos(th);
    double u = -l2 * co;
    dth   = ph;
    dph   = si - alpha*ph - l2*co*co;
    dl1   = -si - l2*co - l2*l2*si*co;
    dl2   = -ph - l1 + alpha*l2;
    dcost = 0.5*ph*ph + (1.0 - co) + 0.5*u*u;
}

// Backward RK4: dy/ds = -f(y), s ∈ [0, T]
// Starts at patch point (xi, S*xi), arrives at (x0, lam0)
__device__
void integrate_backward(double xi0, double xi1,
                        double lxi0, double lxi1,
                        double alpha, double T, int n,
                        double& out_th, double& out_ph,
                        double& out_l1, double& out_l2) {
    double th = xi0, ph = xi1, l1 = lxi0, l2 = lxi1;
    double h = T / (double)n;

    for (int i = 0; i < n; i++) {
        // k1 = -f(y)
        double d1th, d1ph, d1l1, d1l2, dc;
        pmp_deriv(th, ph, l1, l2, alpha, d1th, d1ph, d1l1, d1l2, dc);
        d1th = -d1th; d1ph = -d1ph; d1l1 = -d1l1; d1l2 = -d1l2;

        // k2
        double d2th, d2ph, d2l1, d2l2;
        pmp_deriv(th+0.5*h*d1th, ph+0.5*h*d1ph, l1+0.5*h*d1l1, l2+0.5*h*d1l2, alpha,
                  d2th, d2ph, d2l1, d2l2, dc);
        d2th = -d2th; d2ph = -d2ph; d2l1 = -d2l1; d2l2 = -d2l2;

        // k3
        double d3th, d3ph, d3l1, d3l2;
        pmp_deriv(th+0.5*h*d2th, ph+0.5*h*d2ph, l1+0.5*h*d2l1, l2+0.5*h*d2l2, alpha,
                  d3th, d3ph, d3l1, d3l2, dc);
        d3th = -d3th; d3ph = -d3ph; d3l1 = -d3l1; d3l2 = -d3l2;

        // k4
        double d4th, d4ph, d4l1, d4l2;
        pmp_deriv(th+h*d3th, ph+h*d3ph, l1+h*d3l1, l2+h*d3l2, alpha,
                  d4th, d4ph, d4l1, d4l2, dc);
        d4th = -d4th; d4ph = -d4ph; d4l1 = -d4l1; d4l2 = -d4l2;

        double h6 = h / 6.0;
        th += h6*(d1th + 2*d2th + 2*d3th + d4th);
        ph += h6*(d1ph + 2*d2ph + 2*d3ph + d4ph);
        l1 += h6*(d1l1 + 2*d2l1 + 2*d3l1 + d4l1);
        l2 += h6*(d1l2 + 2*d2l2 + 2*d3l2 + d4l2);

        if (!isfinite(th) || fabs(th) > BLOWUP ||
            !isfinite(ph) || fabs(ph) > BLOWUP) {
            out_th = 1e10; out_ph = 1e10; out_l1 = 1e10; out_l2 = 1e10;
            return;
        }
    }
    out_th = th; out_ph = ph; out_l1 = l1; out_l2 = l2;
}

// Forward RK4 for cost
__device__
void integrate_forward_cost(double th0, double ph0, double l1_0, double l2_0,
                            double alpha, double T, int n,
                            double& final_th, double& final_ph, double& total_cost) {
    double th = th0, ph = ph0, l1 = l1_0, l2 = l2_0;
    double cost = 0.0;
    double h = T / (double)n;

    for (int i = 0; i < n; i++) {
        double d1th, d1ph, d1l1, d1l2, d1c;
        pmp_deriv(th, ph, l1, l2, alpha, d1th, d1ph, d1l1, d1l2, d1c);

        double d2th, d2ph, d2l1, d2l2, d2c;
        pmp_deriv(th+0.5*h*d1th, ph+0.5*h*d1ph, l1+0.5*h*d1l1, l2+0.5*h*d1l2, alpha,
                  d2th, d2ph, d2l1, d2l2, d2c);

        double d3th, d3ph, d3l1, d3l2, d3c;
        pmp_deriv(th+0.5*h*d2th, ph+0.5*h*d2ph, l1+0.5*h*d2l1, l2+0.5*h*d2l2, alpha,
                  d3th, d3ph, d3l1, d3l2, d3c);

        double d4th, d4ph, d4l1, d4l2, d4c;
        pmp_deriv(th+h*d3th, ph+h*d3ph, l1+h*d3l1, l2+h*d3l2, alpha,
                  d4th, d4ph, d4l1, d4l2, d4c);

        double h6 = h / 6.0;
        th   += h6*(d1th + 2*d2th + 2*d3th + d4th);
        ph   += h6*(d1ph + 2*d2ph + 2*d3ph + d4ph);
        l1   += h6*(d1l1 + 2*d2l1 + 2*d3l1 + d4l1);
        l2   += h6*(d1l2 + 2*d2l2 + 2*d3l2 + d4l2);
        cost += h6*(d1c  + 2*d2c  + 2*d3c  + d4c);

        if (!isfinite(cost) || fabs(th) > BLOWUP) {
            final_th = th; final_ph = ph; total_cost = cost;
            return;
        }
    }
    final_th = th; final_ph = ph; total_cost = cost;
}

// Backward residual: F(ξ) = x_backward(T_back) - x0
__device__
void back_residual(double xi0, double xi1,
                   const double S[2][2], double alpha,
                   double T, int n,
                   double x0_0, double x0_1,
                   double* r0, double* r1) {
    double lxi0 = S[0][0]*xi0 + S[0][1]*xi1;
    double lxi1 = S[1][0]*xi0 + S[1][1]*xi1;
    double oth, oph, ol1, ol2;
    integrate_backward(xi0, xi1, lxi0, lxi1, alpha, T, n, oth, oph, ol1, ol2);
    if (fabs(oth) > 1e9) { *r0 = 1e6; *r1 = 1e6; return; }
    *r0 = oth - x0_0;
    *r1 = oph - x0_1;
}

// Newton for backward shooting
__device__
void newton_backward(double& xi0, double& xi1,
                     const double S[2][2], double alpha,
                     double T, int n,
                     double x0_0, double x0_1) {
    double r0, r1;
    back_residual(xi0, xi1, S, alpha, T, n, x0_0, x0_1, &r0, &r1);

    for (int iter = 0; iter < MAX_NEWTON; iter++) {
        double nrm = sqrt(r0*r0 + r1*r1);
        if (nrm < NEWTON_TOL) break;

        double xnrm = sqrt(xi0*xi0 + xi1*xi1);
        double eps = fmax(FD_ABS, FD_REL * fmax(1.0, xnrm));

        double rp00, rp01, rm00, rm01, rp10, rp11, rm10, rm11;
        back_residual(xi0+eps, xi1, S, alpha, T, n, x0_0, x0_1, &rp00, &rp01);
        back_residual(xi0-eps, xi1, S, alpha, T, n, x0_0, x0_1, &rm00, &rm01);
        back_residual(xi0, xi1+eps, S, alpha, T, n, x0_0, x0_1, &rp10, &rp11);
        back_residual(xi0, xi1-eps, S, alpha, T, n, x0_0, x0_1, &rm10, &rm11);

        double J00 = (rp00-rm00)/(2*eps), J10 = (rp01-rm01)/(2*eps);
        double J01 = (rp10-rm10)/(2*eps), J11 = (rp11-rm11)/(2*eps);

        double det = J00*J11 - J01*J10;
        double d0, d1;
        if (fabs(det) < 1e-30) { d0 = -0.01*r0; d1 = -0.01*r1; }
        else {
            d0 = ( J11*(-r0) - J01*(-r1)) / det;
            d1 = (-J10*(-r0) + J00*(-r1)) / det;
        }

        double step = 1.0;
        for (int ls = 0; ls < MAX_ARMIJO; ls++) {
            double tr0, tr1;
            back_residual(xi0+step*d0, xi1+step*d1, S, alpha, T, n, x0_0, x0_1, &tr0, &tr1);
            double tnrm = sqrt(tr0*tr0 + tr1*tr1);
            if (isfinite(tnrm) && tnrm < nrm*(1.0 - 1e-4*step)) break;
            step *= 0.5;
        }

        xi0 += step*d0;
        xi1 += step*d1;
        if (!isfinite(xi0) || !isfinite(xi1)) { xi0 = xi1 = 0; return; }

        back_residual(xi0, xi1, S, alpha, T, n, x0_0, x0_1, &r0, &r1);
    }
}


// ═══════════════════════════════════════════════════════════════════
// 3. GPU Kernel
// ═══════════════════════════════════════════════════════════════════

__global__
void solve_kernel(
    const double* __restrict__ theta_arr,
    const double* __restrict__ phi_arr,
    const AREParams* __restrict__ are_arr,
    const double* __restrict__ xi0_arr,
    const double* __restrict__ xi1_arr,
    Result* __restrict__ result_arr,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double theta0 = theta_arr[idx];
    double phi0   = phi_arr[idx];
    AREParams are = are_arr[idx];
    double alpha  = are.alpha;

    double S[2][2];
    S[0][0] = are.S[0][0]; S[0][1] = are.S[0][1];
    S[1][0] = are.S[1][0]; S[1][1] = are.S[1][1];

    double xnrm = sqrt(theta0*theta0 + phi0*phi0);
    if (xnrm < 1e-15) {
        result_arr[idx].l1   = S[0][0]*theta0 + S[0][1]*phi0;
        result_arr[idx].l2   = S[1][0]*theta0 + S[1][1]*phi0;
        result_arr[idx].cost = S[0][0]*theta0*theta0 + 2*S[0][1]*theta0*phi0 + S[1][1]*phi0*phi0;
        return;
    }

    double T_back = fmax(6.0, 5.0/are.sigma_slow + 0.3*fabs(phi0));
    T_back = fmin(T_back, 20.0);
    int n_back = max(3000, (int)(T_back * 500.0));

    double xi0 = xi0_arr[idx];
    double xi1 = xi1_arr[idx];

    newton_backward(xi0, xi1, S, alpha, T_back, n_back, theta0, phi0);

    // Extract lambda
    double lxi0 = S[0][0]*xi0 + S[0][1]*xi1;
    double lxi1 = S[1][0]*xi0 + S[1][1]*xi1;
    double oth, oph, ol1, ol2;
    integrate_backward(xi0, xi1, lxi0, lxi1, alpha, T_back, n_back, oth, oph, ol1, ol2);

    double l1 = ol1;
    double l2 = ol2;

    // Cost: forward integrate + LQR tail
    double T_cost = fmin(T_back, 5.0);
    int n_cost = max(3000, (int)(T_cost * 1000.0));
    double fth, fph, J;
    integrate_forward_cost(theta0, phi0, l1, l2, alpha, T_cost, n_cost, fth, fph, J);

    if (fabs(fth) < 1e9) {
        J += S[0][0]*fth*fth + 2*S[0][1]*fth*fph + S[1][1]*fph*fph;
    }

    result_arr[idx].l1   = l1;
    result_arr[idx].l2   = l2;
    result_arr[idx].cost = J;
}


// ═══════════════════════════════════════════════════════════════════
// 4. Host interface
// ═══════════════════════════════════════════════════════════════════

std::vector<Result> solve_many(
    const std::vector<double>& theta,
    const std::vector<double>& phi,
    const std::vector<double>& alpha
) {
    int N = (int)theta.size();
    std::vector<Result> results(N);

    std::vector<AREParams> are_params(N);
    std::vector<double> xi0_vec(N), xi1_vec(N);

    for (int i = 0; i < N; i++) {
        are_params[i] = solve_are_cpu(alpha[i]);
        double T_back = fmax(6.0, 5.0/are_params[i].sigma_slow + 0.3*fabs(phi[i]));
        T_back = fmin(T_back, 20.0);
        compute_xi_guess(are_params[i], T_back, theta[i], phi[i],
                         xi0_vec[i], xi1_vec[i]);
    }

    double *d_theta, *d_phi, *d_xi0, *d_xi1;
    AREParams *d_are;
    Result *d_results;

    cudaMalloc(&d_theta, N*sizeof(double));
    cudaMalloc(&d_phi,   N*sizeof(double));
    cudaMalloc(&d_xi0,   N*sizeof(double));
    cudaMalloc(&d_xi1,   N*sizeof(double));
    cudaMalloc(&d_are,   N*sizeof(AREParams));
    cudaMalloc(&d_results, N*sizeof(Result));

    cudaMemcpy(d_theta, theta.data(),       N*sizeof(double),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi,   phi.data(),         N*sizeof(double),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_xi0,   xi0_vec.data(),     N*sizeof(double),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_xi1,   xi1_vec.data(),     N*sizeof(double),    cudaMemcpyHostToDevice);
    cudaMemcpy(d_are,   are_params.data(),  N*sizeof(AREParams), cudaMemcpyHostToDevice);

    int threads = 128;
    int blocks  = (N + threads - 1) / threads;
    solve_kernel<<<blocks, threads>>>(d_theta, d_phi, d_are, d_xi0, d_xi1, d_results, N);
    cudaDeviceSynchronize();

    cudaMemcpy(results.data(), d_results, N*sizeof(Result), cudaMemcpyDeviceToHost);

    cudaFree(d_theta); cudaFree(d_phi); cudaFree(d_xi0); cudaFree(d_xi1);
    cudaFree(d_are); cudaFree(d_results);

    return results;
}

Result solve(double theta, double phi, double alpha) {
    return solve_many({theta}, {phi}, {alpha})[0];
}
