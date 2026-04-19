# Math 589B Final Programming Project (Spring 2026)

## Problem

We consider the controlled pendulum:

$$
\ddot{\theta} = \sin(\theta) - \alpha \dot{\theta} + u \cos(\theta)
$$

State:
$x = (\theta, \phi)$

Cost:

$$
\int_0^\infty \left[(1 - \cos \theta) + \frac{1}{2}\phi^2 + \frac{1}{2}u^2\right] dt
$$

Optimal control:
$u^* = -\lambda_2 \cos(\theta)$

## Task

Given:
$\theta, \phi$

Return:
$\lambda_1, \lambda_2, J$

## Program Interface

`./solver theta phi`

Output:
`lambda1 lambda2 J`

More details coming this week.
