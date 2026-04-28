#!/usr/bin/env python3

import sys
import math
import numpy as np
from numba import cuda


MAX_NEWTON_ITERS = 20
NEWTON_TOL = 1.0e-8
FD_EPS = 1.0e-6
BLOWUP_TOL = 1.0e8


@cuda.jit(device=True)
def rhs(theta, phi, lambda1, lambda2, cost, alpha):
    s = math.sin(theta)
    c = math.cos(theta)

    u = -lambda2 * c

    theta_dot = phi
    phi_dot = s - alpha * phi - lambda2 * c * c

    lambda1_dot = -s - lambda2 * c - lambda2 * lambda2 * s * c
    lambda2_dot = -phi - lambda1 + alpha * lambda2

    cost_dot = (1.0 - c) + 0.5 * phi * phi + 0.5 * u * u

    return theta_dot, phi_dot, lambda1_dot, lambda2_dot, cost_dot


@cuda.jit(device=True)
def rk4_step(theta, phi, lambda1, lambda2, cost, alpha, h):
    k1_t, k1_p, k1_l1, k1_l2, k1_c = rhs(
        theta, phi, lambda1, lambda2, cost, alpha
    )

    k2_t, k2_p, k2_l1, k2_l2, k2_c = rhs(
        theta + 0.5 * h * k1_t,
        phi + 0.5 * h * k1_p,
        lambda1 + 0.5 * h * k1_l1,
        lambda2 + 0.5 * h * k1_l2,
        cost + 0.5 * h * k1_c,
        alpha
    )

    k3_t, k3_p, k3_l1, k3_l2, k3_c = rhs(
        theta + 0.5 * h * k2_t,
        phi + 0.5 * h * k2_p,
        lambda1 + 0.5 * h * k2_l1,
        lambda2 + 0.5 * h * k2_l2,
        cost + 0.5 * h * k2_c,
        alpha
    )

    k4_t, k4_p, k4_l1, k4_l2, k4_c = rhs(
        theta + h * k3_t,
        phi + h * k3_p,
        lambda1 + h * k3_l1,
        lambda2 + h * k3_l2,
        cost + h * k3_c,
        alpha
    )

    theta_new = theta + (h / 6.0) * (
        k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t
    )

    phi_new = phi + (h / 6.0) * (
        k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
    )

    lambda1_new = lambda1 + (h / 6.0) * (
        k1_l1 + 2.0 * k2_l1 + 2.0 * k3_l1 + k4_l1
    )

    lambda2_new = lambda2 + (h / 6.0) * (
        k1_l2 + 2.0 * k2_l2 + 2.0 * k3_l2 + k4_l2
    )

    cost_new = cost + (h / 6.0) * (
        k1_c + 2.0 * k2_c + 2.0 * k3_c + k4_c
    )

    return theta_new, phi_new, lambda1_new, lambda2_new, cost_new


@cuda.jit(device=True)
def integrate(theta0, phi0, lambda1_0, lambda2_0, alpha, T, h):
    theta = theta0
    phi = phi0
    lambda1 = lambda1_0
    lambda2 = lambda2_0
    cost = 0.0

    nsteps = int(T / h)

    for i in range(nsteps):
        theta, phi, lambda1, lambda2, cost = rk4_step(
            theta, phi, lambda1, lambda2, cost, alpha, h
        )

        if (
            math.isnan(theta) or math.isnan(phi) or
            math.isnan(lambda1) or math.isnan(lambda2) or
            math.isinf(theta) or math.isinf(phi) or
            math.isinf(lambda1) or math.isinf(lambda2) or
            abs(theta) > BLOWUP_TOL or
            abs(phi) > BLOWUP_TOL or
            abs(lambda1) > BLOWUP_TOL or
            abs(lambda2) > BLOWUP_TOL
        ):
            return 1.0e10, 1.0e10, 1.0e10, 1.0e10, 1.0e10

    return theta, phi, lambda1, lambda2, cost


@cuda.jit
def solve_kernel(theta_array, phi_array, alpha_array, output_array, T, h):
    idx = cuda.grid(1)

    if idx >= theta_array.shape[0]:
        return

    theta0 = theta_array[idx]
    phi0 = phi_array[idx]
    alpha = alpha_array[idx]

    lambda1 = 0.0
    lambda2 = 0.0

    for newton_iter in range(MAX_NEWTON_ITERS):
        theta_T, phi_T, l1_T, l2_T, cost_T = integrate(
            theta0, phi0, lambda1, lambda2, alpha, T, h
        )

        r1 = theta_T
        r2 = phi_T

        norm_r = math.sqrt(r1 * r1 + r2 * r2)

        if norm_r < NEWTON_TOL:
            break

        theta_l1, phi_l1, _, _, _ = integrate(
            theta0, phi0, lambda1 + FD_EPS, lambda2, alpha, T, h
        )

        theta_l2, phi_l2, _, _, _ = integrate(
            theta0, phi0, lambda1, lambda2 + FD_EPS, alpha, T, h
        )

        J11 = (theta_l1 - r1) / FD_EPS
        J21 = (phi_l1 - r2) / FD_EPS

        J12 = (theta_l2 - r1) / FD_EPS
        J22 = (phi_l2 - r2) / FD_EPS

        det = J11 * J22 - J12 * J21

        if abs(det) < 1.0e-14:
            break

        delta1 = (J22 * (-r1) - J12 * (-r2)) / det
        delta2 = (-J21 * (-r1) + J11 * (-r2)) / det

        damping = 0.5

        lambda1 = lambda1 + damping * delta1
        lambda2 = lambda2 + damping * delta2

        if math.isnan(lambda1) or math.isnan(lambda2) or math.isinf(lambda1) or math.isinf(lambda2):
            lambda1 = 0.0
            lambda2 = 0.0
            break

    theta_T, phi_T, l1_T, l2_T, cost_T = integrate(
        theta0, phi0, lambda1, lambda2, alpha, T, h
    )

    output_array[idx, 0] = lambda1
    output_array[idx, 1] = lambda2
    output_array[idx, 2] = cost_T


def solve_many(theta_values, phi_values, alpha_values):
    theta_values = np.asarray(theta_values, dtype=np.float64)
    phi_values = np.asarray(phi_values, dtype=np.float64)
    alpha_values = np.asarray(alpha_values, dtype=np.float64)

    n = theta_values.shape[0]

    output = np.zeros((n, 3), dtype=np.float64)

    d_theta = cuda.to_device(theta_values)
    d_phi = cuda.to_device(phi_values)
    d_alpha = cuda.to_device(alpha_values)
    d_output = cuda.to_device(output)

    T = 5.0
    h = 1.0e-3

    threads_per_block = 128
    blocks_per_grid = (n + threads_per_block - 1) // threads_per_block

    solve_kernel[blocks_per_grid, threads_per_block](
        d_theta,
        d_phi,
        d_alpha,
        d_output,
        T,
        h
    )

    cuda.synchronize()

    output = d_output.copy_to_host()

    return output


def main():
    if len(sys.argv) == 4:
        theta = float(sys.argv[1])
        phi = float(sys.argv[2])
        alpha = float(sys.argv[3])

        output = solve_many(
            np.array([theta]),
            np.array([phi]),
            np.array([alpha])
        )

        print(f"{output[0,0]:.10f} {output[0,1]:.10f} {output[0,2]:.10f}")

    else:
        print("usage: ./solver theta phi alpha", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()