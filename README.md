# Math 589B Final Programming Project (Spring 2026)

## Problem

We consider the controlled pendulum:

```math
\ddot{\theta} = \sin(\theta) - \alpha \dot{\theta} + u \cos(\theta).
```

State:

```math
x = (\theta,\phi),
\qquad
\phi = \dot{\theta}.
```

Cost:

```math
J
=
\int_0^\infty
\left[
(1-\cos\theta)
+
\frac{1}{2}\phi^2
+
\frac{1}{2}u^2
\right]dt.
```

## Task

Given:

```math
\theta,\phi
```

Return:

```math
\lambda_1,\lambda_2,J
```

## Program Interface

```bash
./solver theta phi
```

Output:

```bash
lambda1 lambda2 J
```

---

## Analytical Derivation up to the Costate Equations

We consider the controlled pendulum

```math
\ddot{\theta}
=
\sin(\theta)-\alpha \dot{\theta}+u\cos(\theta).
```

Let

```math
\phi=\dot{\theta}.
```

Then the state is

```math
x=(\theta,\phi).
```

Therefore, the second-order equation becomes the first-order system

```math
\begin{aligned}
\dot{\theta} &= \phi,\\
\dot{\phi} &= \sin(\theta)-\alpha\phi+u\cos(\theta).
\end{aligned}
```

So

```math
\dot{x}=f(x,u),
```

where

```math
f(x,u)
=
\begin{pmatrix}
\phi\\
\sin(\theta)-\alpha\phi+u\cos(\theta)
\end{pmatrix}.
```

The running cost is

```math
L(\theta,\phi,u)
=
(1-\cos\theta)
+
\frac{1}{2}\phi^2
+
\frac{1}{2}u^2.
```

Let the costate vector be

```math
\lambda
=
\begin{pmatrix}
\lambda_1\\
\lambda_2
\end{pmatrix}.
```

The Hamiltonian is

```math
\mathbb{\mathcal{H}}=L+\lambda^T f.
```

Therefore,

```math
\mathbb{\mathcal{H}}(\theta,\phi,u,\lambda_1,\lambda_2)
=
(1-\cos\theta)
+
\frac{1}{2}\phi^2
+
\frac{1}{2}u^2
+
\lambda_1\phi
+
\lambda_2
\left(
\sin\theta-\alpha\phi+u\cos\theta
\right).
```

To find the optimal control, set

```math
\frac{\partial \mathbb{\mathcal{H}}}{\partial u}=0.
```

Since

```math
\frac{\partial \mathbb{\mathcal{H}}}{\partial u}
=
u+\lambda_2\cos\theta,
```

we get

```math
u+\lambda_2\cos\theta=0.
```

Therefore,

```math
u^*=-\lambda_2\cos\theta.
```

Now substitute the optimal control back into the Hamiltonian. Since

```math
\frac{1}{2}(u^*)^2
=
\frac{1}{2}\lambda_2^2\cos^2\theta
```

and

```math
\lambda_2u^*\cos\theta
=
-\lambda_2^2\cos^2\theta,
```

we have

```math
\frac{1}{2}(u^*)^2+\lambda_2u^*\cos\theta
=
-\frac{1}{2}\lambda_2^2\cos^2\theta.
```

Thus the effective Hamiltonian is

```math
H(\theta,\phi,\lambda_1,\lambda_2)
=
(1-\cos\theta)
+
\frac{1}{2}\phi^2
+
\lambda_1\phi
+
\lambda_2(\sin\theta-\alpha\phi)
-
\frac{1}{2}\lambda_2^2\cos^2\theta.
```

The costate equations are

```math
\dot{\lambda}_1
=
-\frac{\partial H}{\partial \theta},
\qquad
\dot{\lambda}_2
=
-\frac{\partial H}{\partial \phi}.
```

Now compute

```math
\frac{\partial H}{\partial \theta}
=
\sin\theta
+
\lambda_2\cos\theta
+
\lambda_2^2\sin\theta\cos\theta.
```

Therefore,

```math
\dot{\lambda}_1
=
-\sin\theta
-\lambda_2\cos\theta
-\lambda_2^2\sin\theta\cos\theta.
```

Also,

```math
\frac{\partial H}{\partial \phi}
=
\phi+\lambda_1-\alpha\lambda_2.
```

Therefore,

```math
\dot{\lambda}_2
=
-\phi-\lambda_1+\alpha\lambda_2.
```

Hence the state-costate system is

```math
\begin{aligned}
\dot{\theta}
&=
\phi,\\
\dot{\phi}
&=
\sin\theta-\alpha\phi-\lambda_2\cos^2\theta,\\
\dot{\lambda}_1
&=
-\sin\theta
-\lambda_2\cos\theta
-\lambda_2^2\sin\theta\cos\theta,\\
\dot{\lambda}_2
&=
-\phi-\lambda_1+\alpha\lambda_2.
\end{aligned}
```

with optimal control

```math
u^*=-\lambda_2\cos\theta.
```
