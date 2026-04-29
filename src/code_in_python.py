import numpy as np

def pendulum_dynamics(y, alpha):
    theta, phi, lambda_1, lambda_2, cost = y
    theta_dot = phi
    phi_dot = np.sin(theta) - alpha * phi - lambda_2 * np.cos(theta) ** 2
    lambda_1_dot = -np.sin(theta) - lambda_2 * np.cos(theta) - lambda_2**2 * np.sin(theta) * np.cos(theta)
    lambda_2_dot = -phi - lambda_1 + alpha * lambda_2
    cost_dot = (1.0 - np.cos(theta)) + 0.5 * phi**2 + 0.5 * (-lambda_2 * np.cos(theta))**2

    return np.array([theta_dot, phi_dot, lambda_1_dot, lambda_2_dot, cost_dot], dtype=float)

def rk4_step(y, h, alpha):
    k1 = pendulum_dynamics(y, alpha)
    k2 = pendulum_dynamics(y + 0.5 * h * k1, alpha)
    k3 = pendulum_dynamics(y + 0.5 * h * k2, alpha)
    k4 = pendulum_dynamics(y + h * k3, alpha)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def simulate_backward_pass(terminal_costates, initial_condition, T, h, alpha):
    # Start at t=T: state is [0,0], cost is 0, costates are our current guess
    state = np.array([0.0, 0.0, terminal_costates[0], terminal_costates[1], 0.0])
    
    num_steps = int(T / h)
    
    # Integrate backwards
    for _ in range(num_steps):
        state = rk4_step(state, -h, alpha)
        
    # error
    error = np.array([state[0] - initial_condition[0], state[1] - initial_condition[1]])
    return error, state

def backwards_shooting(initial_condition, T, alpha, h=1e-3, tol=1e-5):
    guess_terminal_costates = np.array([1e-3, 1e-3], dtype=float)
    eps = 1e-6
    
    # 2. Newton's Method loop to adjust the guess
    for i in range(50):
        # Shoot backwards with current guess
        err, state_at_t0 = simulate_backward_pass(guess_terminal_costates, initial_condition, T, h, alpha)
        
        # If we hit the target, return the t=0 costates
        if np.linalg.norm(err) < tol:
            print(f"Target hit after {i} iterations!")
            return state_at_t0[2], state_at_t0[3] 
            
        # 3. Calculate how to adjust the guess (Jacobian via finite differences)
        J = np.zeros((2, 2))
        for j in range(2):
            perturbed_guess = guess_terminal_costates.copy()
            perturbed_guess[j] += eps
            perturbed_err, _ = simulate_backward_pass(perturbed_guess, initial_condition, T, h, alpha)
            J[:, j] = (perturbed_err - err) / eps
            
        # 4. Update the guess using the Jacobian
        guess_terminal_costates = guess_terminal_costates - np.linalg.solve(J, err)
        
    print("Warning: Did not converge within tolerance.")
    return state_at_t0[2], state_at_t0[3]

if __name__ == "__main__":
    initial_condition = np.array([np.pi, 1.1])  # Target state at t=0 (theta, phi)
    T = 5.0  # Time horizon
    alpha = 0.1  # Damping coefficient

    lambda_1_0, lambda_2_0 = backwards_shooting(initial_condition, T, alpha)
    print(f"Optimal initial costates: lambda_1(0) = {lambda_1_0:.6f}, lambda_2(0) = {lambda_2_0:.6f}")