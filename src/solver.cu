#include "solver.hpp"
#include <cmath>
#include <cuda_runtime.h>

#define MAX_NEWTON_ITERS 20
#define NEWTON_TOL 1.0e-8
#define FD_EPS 1.0e-6
#define BLOWUP_TOL 1.0e8

struct State {
    double theta;
    double phi;
    double lambda1;
    double lambda2;
    double cost;
};

__device__
State dynamics(State s, double alpha) {
    State ds;

    double sin_theta = sin(s.theta);
    double cos_theta = cos(s.theta);

    double u = -s.lambda2 * cos_theta;

    ds.theta = s.phi;

    ds.phi =
        sin_theta
        - alpha * s.phi
        - s.lambda2 * cos_theta * cos_theta;

    ds.lambda1 =
        -sin_theta
        - s.lambda2 * cos_theta
        - s.lambda2 * s.lambda2 * sin_theta * cos_theta;

    ds.lambda2 =
        -s.phi - s.lambda1 + alpha * s.lambda2;

    ds.cost =
        (1.0 - cos_theta)
        + 0.5 * s.phi * s.phi
        + 0.5 * u * u;

    return ds;
}

__device__
State add_scaled(State s, State k, double a) {
    State out;

    out.theta   = s.theta   + a * k.theta;
    out.phi     = s.phi     + a * k.phi;
    out.lambda1 = s.lambda1 + a * k.lambda1;
    out.lambda2 = s.lambda2 + a * k.lambda2;
    out.cost    = s.cost    + a * k.cost;

    return out;
}

__device__
State rk4_step(State s, double h, double alpha) {
    State k1 = dynamics(s, alpha);
    State k2 = dynamics(add_scaled(s, k1, 0.5 * h), alpha);
    State k3 = dynamics(add_scaled(s, k2, 0.5 * h), alpha);
    State k4 = dynamics(add_scaled(s, k3, h), alpha);

    State out;

    out.theta =
        s.theta + (h / 6.0) *
        (k1.theta + 2.0 * k2.theta + 2.0 * k3.theta + k4.theta);

    out.phi =
        s.phi + (h / 6.0) *
        (k1.phi + 2.0 * k2.phi + 2.0 * k3.phi + k4.phi);

    out.lambda1 =
        s.lambda1 + (h / 6.0) *
        (k1.lambda1 + 2.0 * k2.lambda1 + 2.0 * k3.lambda1 + k4.lambda1);

    out.lambda2 =
        s.lambda2 + (h / 6.0) *
        (k1.lambda2 + 2.0 * k2.lambda2 + 2.0 * k3.lambda2 + k4.lambda2);

    out.cost =
        s.cost + (h / 6.0) *
        (k1.cost + 2.0 * k2.cost + 2.0 * k3.cost + k4.cost);

    return out;
}

__device__
State integrate(
    double theta0,
    double phi0,
    double lambda1_0,
    double lambda2_0,
    double alpha,
    double T,
    double h
) {
    State s;

    s.theta = theta0;
    s.phi = phi0;
    s.lambda1 = lambda1_0;
    s.lambda2 = lambda2_0;
    s.cost = 0.0;

    int N = (int)(T / h);

    for (int i = 0; i < N; i++) {
        s = rk4_step(s, h, alpha);

        if (!isfinite(s.theta) ||
            !isfinite(s.phi) ||
            !isfinite(s.lambda1) ||
            !isfinite(s.lambda2) ||
            fabs(s.theta) > BLOWUP_TOL ||
            fabs(s.phi) > BLOWUP_TOL ||
            fabs(s.lambda1) > BLOWUP_TOL ||
            fabs(s.lambda2) > BLOWUP_TOL) {

            s.theta = 1.0e10;
            s.phi = 1.0e10;
            s.lambda1 = 1.0e10;
            s.lambda2 = 1.0e10;
            s.cost = 1.0e10;
            return s;
        }
    }

    return s;
}

__device__
void residual(
    double theta0,
    double phi0,
    double lambda1_0,
    double lambda2_0,
    double alpha,
    double T,
    double h,
    double* r1,
    double* r2
) {
    State s = integrate(theta0, phi0, lambda1_0, lambda2_0, alpha, T, h);

    *r1 = s.theta;
    *r2 = s.phi;
}

__device__
int solve_2x2(
    double a,
    double b,
    double c,
    double d,
    double rhs1,
    double rhs2,
    double* x1,
    double* x2
) {
    double det = a * d - b * c;

    if (fabs(det) < 1.0e-14) {
        *x1 = 0.0;
        *x2 = 0.0;
        return 0;
    }

    *x1 = ( d * rhs1 - b * rhs2) / det;
    *x2 = (-c * rhs1 + a * rhs2) / det;

    return 1;
}

__global__
void solve_kernel(
    const double* theta_array,
    const double* phi_array,
    const double* alpha_array,
    Result* result_array,
    int N,
    double T,
    double h
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N) {
        return;
    }

    double theta0 = theta_array[idx];
    double phi0 = phi_array[idx];
    double alpha = alpha_array[idx];

    double lambda1 = 0.0;
    double lambda2 = 0.0;

    for (int iter = 0; iter < MAX_NEWTON_ITERS; iter++) {
        double r1;
        double r2;

        residual(theta0, phi0, lambda1, lambda2, alpha, T, h, &r1, &r2);

        double norm_r = sqrt(r1 * r1 + r2 * r2);

        if (norm_r < NEWTON_TOL) {
            break;
        }

        double r1_l1;
        double r2_l1;
        double r1_l2;
        double r2_l2;

        residual(theta0, phi0, lambda1 + FD_EPS, lambda2, alpha, T, h, &r1_l1, &r2_l1);
        residual(theta0, phi0, lambda1, lambda2 + FD_EPS, alpha, T, h, &r1_l2, &r2_l2);

        double J11 = (r1_l1 - r1) / FD_EPS;
        double J21 = (r2_l1 - r2) / FD_EPS;

        double J12 = (r1_l2 - r1) / FD_EPS;
        double J22 = (r2_l2 - r2) / FD_EPS;

        double delta1;
        double delta2;

        int ok = solve_2x2(
            J11, J12,
            J21, J22,
            -r1, -r2,
            &delta1, &delta2
        );

        if (!ok) {
            break;
        }

        double damping = 0.5;

        lambda1 += damping * delta1;
        lambda2 += damping * delta2;

        if (!isfinite(lambda1) || !isfinite(lambda2)) {
            lambda1 = 0.0;
            lambda2 = 0.0;
            break;
        }
    }

    State final_state = integrate(theta0, phi0, lambda1, lambda2, alpha, T, h);

    result_array[idx].l1 = lambda1;
    result_array[idx].l2 = lambda2;
    result_array[idx].cost = final_state.cost;
}

std::vector<Result> solve_many(
    const std::vector<double>& theta,
    const std::vector<double>& phi,
    const std::vector<double>& alpha
) {
    int N = (int)theta.size();

    std::vector<Result> results(N);

    double* d_theta = nullptr;
    double* d_phi = nullptr;
    double* d_alpha = nullptr;
    Result* d_results = nullptr;

    cudaMalloc((void**)&d_theta, N * sizeof(double));
    cudaMalloc((void**)&d_phi, N * sizeof(double));
    cudaMalloc((void**)&d_alpha, N * sizeof(double));
    cudaMalloc((void**)&d_results, N * sizeof(Result));

    cudaMemcpy(d_theta, theta.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_phi, phi.data(), N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_alpha, alpha.data(), N * sizeof(double), cudaMemcpyHostToDevice);

    double T = 5.0;
    double h = 1.0e-3;

    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    solve_kernel<<<blocks, threads_per_block>>>(
        d_theta,
        d_phi,
        d_alpha,
        d_results,
        N,
        T,
        h
    );

    cudaDeviceSynchronize();

    cudaMemcpy(results.data(), d_results, N * sizeof(Result), cudaMemcpyDeviceToHost);

    cudaFree(d_theta);
    cudaFree(d_phi);
    cudaFree(d_alpha);
    cudaFree(d_results);

    return results;
}

Result solve(double theta, double phi, double alpha) {
    std::vector<double> theta_vec(1);
    std::vector<double> phi_vec(1);
    std::vector<double> alpha_vec(1);

    theta_vec[0] = theta;
    phi_vec[0] = phi;
    alpha_vec[0] = alpha;

    std::vector<Result> results = solve_many(theta_vec, phi_vec, alpha_vec);

    return results[0];
}