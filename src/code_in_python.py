import numpy as np
from scipy.optimize import least_squares


def pendulum_dynamics(state, alpha):
    theta, phi, lambda_1, lambda_2 = state
    theta_dot = phi
    phi_dot = np.sin(theta) - alpha * phi  - lambda_2 * np.cos(theta) * np.cos(theta)
    lambda_1_dot = -np.sin(theta)- lambda_2 * np.cos(theta)- lambda_2**2 * np.sin(theta) * np.cos(theta)
    lambda_2_dot = -phi - lambda_1 + alpha * lambda_2

    return np.array([theta_dot,phi_dot,lambda_1_dot,lambda_2_dot], dtype=float)


def rk4(state, h, alpha):
    k1 = pendulum_dynamics(state, alpha)
    k2 = pendulum_dynamics(state + 0.5 * h * k1, alpha)
    k3 = pendulum_dynamics(state + 0.5 * h * k2, alpha)
    k4 = pendulum_dynamics(state + h * k3, alpha)

    new_state = state + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return new_state

def shooting(lambda_guess, theta_0, phi_0, alpha, T, h):
    lambda_1_0 = lambda_guess[0]
    lambda_2_0 = lambda_guess[1]

    state = np.array([
        theta_0,
        phi_0,
        lambda_1_0,
        lambda_2_0
    ], dtype=float)

    N = int(T / h)

    for i in range(N):
        state = rk4(state, h, alpha)

        if not np.all(np.isfinite(state)):
            return np.array([1e10, 1e10])

    theta_T = state[0]
    phi_T = state[1]

    return np.array([theta_T, phi_T])
if __name__ == "__main__":
    theta_0 = 0.5
    phi_0 = 0.2

    alpha = 1.0
    T = 5.0
    h = 1e-3

    initial_costate_guess = np.array([0.0, 0.0])

    result = least_squares(
        shooting,
        initial_costate_guess,
        args=(theta_0, phi_0, alpha, T, h),
        xtol=1e-10,
        ftol=1e-10,
        gtol=1e-10,
        max_nfev=100
    )

    lambda_1_0 = result.x[0]
    lambda_2_0 = result.x[1]

    final_residual = shooting(
        result.x,
        theta_0,
        phi_0,
        alpha,
        T,
        h
    )

    print("Final result")
    print("lambda_1(0) =", lambda_1_0)
    print("lambda_2(0) =", lambda_2_0)

    print()
    print("Terminal residual")
    print("theta(T) =", final_residual[0])
    print("phi(T)   =", final_residual[1])

    print()
    print("least_squares success:", result.success)
    print("message:", result.message)
