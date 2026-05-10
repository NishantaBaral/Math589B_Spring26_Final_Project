#define EIGEN_NO_CUDA
#define EIGEN_DONT_VECTORIZE


#include "solver.hpp"

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>


//store one PMP state (theta, phi, lambda1, lambda2, cost)
struct State {
    double th;
    double ph;
    double l1;
    double l2;
    double cost;
};

//This stores two 4D stable basis vectors, can create other canddidate initial conditions by linear combinations of these two vectors.
struct Basis {
    double v[8];
};

//store the result of one candidate trajectory. store patch coordinate, final PMP state, accumulated cost, squared distance to target, and whether the integration survived.
struct Candidate {
    double a;
    double b;
    double th;
    double ph;
    double l1;
    double l2;
    double cost;
    double dist2;
    int ok;
};
//some  CUDA error checking boilerplate, no idea what it does but it seems important to call after every CUDA API call so here it is.
static void gpu_check(cudaError_t e, const char* file, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n",
                     file, line, cudaGetErrorString(e));
        std::exit(2);
    }
}

#define GPU_CHECK(x) gpu_check((x), __FILE__, __LINE__)


//the state costate eqn, can run on both cpu and gpu
__host__ __device__
static State rhs_state(const State& y, double alpha) {
    const double st = sin(y.th);
    const double ct = cos(y.th);
    const double ct2 = ct * ct;
    const double l22 = y.l2 * y.l2;

    State f;
    f.th = y.ph;
    f.ph = st - alpha * y.ph - y.l2 * ct2;
    f.l1 = -st - y.l2 * ct - l22 * ct * st;
    f.l2 = -y.ph - y.l1 + alpha * y.l2;
    f.cost = 1.0 - ct + 0.5 * y.ph * y.ph + 0.5 * l22 * ct2;

    return f;
}

//cpu/gpu helper, add a scaled version of k to y, used in the RK4 integrator below
__host__ __device__
static State add_scaled(const State& y, const State& k, double h) {
    State z;
    z.th   = y.th   + h * k.th;
    z.ph   = y.ph   + h * k.ph;
    z.l1   = y.l1   + h * k.l1;
    z.l2   = y.l2   + h * k.l2;
    z.cost = y.cost + h * k.cost;
    return z;
}

//performs one step of good ol' rk4
__host__ __device__
static State rk4_step(const State& y, double alpha, double dt) {
    const State k1 = rhs_state(y, alpha);
    const State k2 = rhs_state(add_scaled(y, k1, 0.5 * dt), alpha);
    const State k3 = rhs_state(add_scaled(y, k2, 0.5 * dt), alpha);
    const State k4 = rhs_state(add_scaled(y, k3, dt), alpha);

    State z;
    const double h = dt / 6.0;

    z.th   = y.th   + h * (k1.th   + 2.0 * k2.th   + 2.0 * k3.th   + k4.th);
    z.ph   = y.ph   + h * (k1.ph   + 2.0 * k2.ph   + 2.0 * k3.ph   + k4.ph);
    z.l1   = y.l1   + h * (k1.l1   + 2.0 * k2.l1   + 2.0 * k3.l1   + k4.l1);
    z.l2   = y.l2   + h * (k1.l2   + 2.0 * k2.l2   + 2.0 * k3.l2   + k4.l2);
    z.cost = y.cost + h * (k1.cost + 2.0 * k2.cost + 2.0 * k3.cost + k4.cost);

    return z;
}

__host__ __device__
static double sqr(double x) {
    return x * x;
}

//the actual GPU kernel
__global__
static void patch_kernel(Candidate* out,
                         int grid_n,
                         double radius,
                         int steps,
                         double dt,
                         double alpha,
                         Basis B,
                         double target_th,
                         double target_ph) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; //this line means that each thread will have a unique idx, and the kernel will be launched with enough threads to cover all points in the grid. 
    //So each thread will be responsible for simulating one trajectory starting from a point in the patch.
    const int total = grid_n * grid_n;
    //total number of candidate in the grid, we will have it 49.

    if (idx >= total) {
        return;
    }
    //extra gpu threads will do nothing

    //convert idx to  to grid coordinate (i,j)
    const int i = idx / grid_n;
    const int j = idx - i * grid_n;

    //map grid coordinate to patch coordinate (xi, xj) in [-1, 1] x [-1, 1]
    const double xi = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(grid_n - 1);
    const double xj = -1.0 + 2.0 * static_cast<double>(j) / static_cast<double>(grid_n - 1);

    //scale patch coordinate by radius to get initial condition on the stable manifold
    const double a = radius * xi;
    const double b = radius * xj;

    //build initial state from patch coordinate using the stable basis
    State y;
    y.th   = B.v[0] * a + B.v[4] * b;
    y.ph   = B.v[1] * a + B.v[5] * b;
    y.l1   = B.v[2] * a + B.v[6] * b;
    y.l2   = B.v[3] * a + B.v[7] * b;
    y.cost = 0.0;

    int ok = 1;

    for (int k = 0; k < steps; ++k) {
        y = rk4_step(y, alpha, dt);

        if (!isfinite(y.th) || !isfinite(y.ph) ||
            !isfinite(y.l1) || !isfinite(y.l2) ||
            fabs(y.th) > 1.0e8 || fabs(y.ph) > 1.0e8 ||
            fabs(y.l1) > 1.0e8 || fabs(y.l2) > 1.0e8) {
            ok = 0;
            break;
        }
    }
    //store the result in the output array, we will copy this back to the CPU and sort by distance to target to get good initial conditions for the Newton refinement step.
    Candidate c;
    c.a = a;
    c.b = b;
    c.th = y.th;
    c.ph = y.ph;
    c.l1 = y.l1;
    c.l2 = y.l2;
    c.cost = -y.cost;
    c.ok = ok;
//Compute squared shooting residual and write this candidate into the gpu memory. This is all the gpu work.
    if (ok) {
        c.dist2 = sqr(y.th - target_th) + sqr(y.ph - target_ph);
    } else {
        c.dist2 = 1.0e300;
    }

    out[idx] = c;
}

//cpu function, create a 4*4 matrix from linearized PMP dynamics, compute eigval, eigvec, sort eigval by real part, and take the two most stable eigvecs. 
static Eigen::Matrix<double, 4, 2> stable_basis(double alpha) {
    Eigen::Matrix4d A;

    A << 0.0,    1.0,    0.0,    0.0,
         1.0,   -alpha,  0.0,   -1.0,
        -1.0,    0.0,    0.0,   -1.0,
         0.0,   -1.0,   -1.0,    alpha;

    Eigen::EigenSolver<Eigen::Matrix4d> es(A);

    std::vector<std::pair<double, int>> idx;

    for (int i = 0; i < 4; ++i) {
        idx.emplace_back(es.eigenvalues()(i).real(), i);
    }

    std::sort(idx.begin(), idx.end());

    Eigen::Matrix<std::complex<double>, 4, 2> Vc;
    Vc.col(0) = es.eigenvectors().col(idx[0].second);
    Vc.col(1) = es.eigenvectors().col(idx[1].second);

    Eigen::Matrix<double, 4, 2> Vs = Vc.real();

    for (int j = 0; j < 2; ++j) {
        const double n = Vs.col(j).norm();
        if (n > 0.0) {
            Vs.col(j) /= n;
        }
    }
//so we got our stable subsapce basis 
    return Vs;
}

//CPU function, do the same kind of integration as the GPU kernel, but for one candidate.
//very important to have this on the CPU so we can do the Newton refinement step which requires a lot of sequential steps and is not very parallelizable.
static Eigen::Matrix2d stable_gain(double alpha) {
    const Eigen::Matrix<double, 4, 2> Vs = stable_basis(alpha);

    Eigen::Matrix2d top = Vs.topRows<2>();
    Eigen::Matrix2d bot = Vs.bottomRows<2>();

    return bot * top.inverse();
}

static Basis make_basis_struct(double alpha) {
    const Eigen::Matrix<double, 4, 2> Vs = stable_basis(alpha);

    Basis B;

    for (int r = 0; r < 4; ++r) {
        B.v[r] = Vs(r, 0);
        B.v[4 + r] = Vs(r, 1);
    }

    return B;
}

static State integrate_ab(const Basis& B,
                          double a,
                          double b,
                          double alpha,
                          double T,
                          int steps) {
    State y;
    y.th   = B.v[0] * a + B.v[4] * b;
    y.ph   = B.v[1] * a + B.v[5] * b;
    y.l1   = B.v[2] * a + B.v[6] * b;
    y.l2   = B.v[3] * a + B.v[7] * b;
    y.cost = 0.0;

    const double dt = -T / static_cast<double>(steps);

    for (int k = 0; k < steps; ++k) {
        y = rk4_step(y, alpha, dt);

        if (!std::isfinite(y.th) || !std::isfinite(y.ph) ||
            !std::isfinite(y.l1) || !std::isfinite(y.l2) ||
            std::fabs(y.th) > 1.0e9 || std::fabs(y.ph) > 1.0e9 ||
            std::fabs(y.l1) > 1.0e9 || std::fabs(y.l2) > 1.0e9) {
            y.th = 1.0e100;
            y.ph = 1.0e100;
            y.l1 = 1.0e100;
            y.l2 = 1.0e100;
            y.cost = -1.0e100;
            return y;
        }
    }

    return y;
}

//newton's refinement 

static Candidate refine_ab_newton(const Basis& B,
                                  double a0,
                                  double b0,
                                  double target_th,
                                  double target_ph,
                                  double alpha) {
    const double T = 16.0;
    const int steps = 2600;
    const int max_iter = 14;

    double a = a0;
    double b = b0;
//start from a gpu seed
    State y = integrate_ab(B, a, b, alpha, T, steps);
    double best_dist2 = sqr(y.th - target_th) + sqr(y.ph - target_ph);

    for (int it = 0; it < max_iter; ++it) {
        if (!std::isfinite(best_dist2) || best_dist2 < 1.0e-16) {
            break;
        }

        const double ea = std::max(1.0e-12, 1.0e-5 * std::max(std::fabs(a), 1.0e-8));
        const double eb = std::max(1.0e-12, 1.0e-5 * std::max(std::fabs(b), 1.0e-8));
//cpu integration to compute the Jacobian, we will do two additional integrations with small perturbations in a and b to compute finite difference approximations of the Jacobian of the shooting function. 
//This is the main reason we need to do this on the CPU, since we need to do these sequential integrations and they are not very parallelizable.
//AI translated this from my python code, so it might look a bit weird since I was using a lot of numpy broadcasting and stuff in the python code, but here we have to write everything out explicitly.
        const State ya = integrate_ab(B, a + ea, b, alpha, T, steps);
        const State yb = integrate_ab(B, a, b + eb, alpha, T, steps);

        if (!std::isfinite(ya.th) || !std::isfinite(yb.th)) {
            break;
        }

        const double r1 = y.th - target_th;
        const double r2 = y.ph - target_ph;

        const double J11 = (ya.th - y.th) / ea;
        const double J21 = (ya.ph - y.ph) / ea;
        const double J12 = (yb.th - y.th) / eb;
        const double J22 = (yb.ph - y.ph) / eb;

        const double det = J11 * J22 - J12 * J21;

        if (!std::isfinite(det) || std::fabs(det) < 1.0e-14) {
            break;
        }

        const double da = (-r1 * J22 + J12 * r2) / det;
        const double db = ( J21 * r1 - J11 * r2) / det;

        bool accepted = false;
        double scale = 1.0;

        for (int ls = 0; ls < 10; ++ls) {
            const double na = a + scale * da;
            const double nb = b + scale * db;

            const State trial = integrate_ab(B, na, nb, alpha, T, steps);
            const double d2 = sqr(trial.th - target_th) + sqr(trial.ph - target_ph);

            if (std::isfinite(d2) && d2 < best_dist2) {
                a = na;
                b = nb;
                y = trial;
                best_dist2 = d2;
                accepted = true;
                break;
            }

            scale *= 0.5;
        }

        if (!accepted) {
            break;
        }
    }

    Candidate c;
    c.a = a;
    c.b = b;
    c.th = y.th;
    c.ph = y.ph;
    c.l1 = y.l1;
    c.l2 = y.l2;
    c.cost = -y.cost;
    c.dist2 = best_dist2;
    c.ok = std::isfinite(best_dist2) ? 1 : 0;

    return c;
}

//most imp part
//this function itself runs on CPU. It manages GPU work.
static std::vector<Candidate> gpu_patch_search(double theta,
                                               double phi,
                                               double alpha,
                                               const Basis& B) {
    const int grid_n = 49; 
    //we will search a 49x49 grid of initial conditions on the stable manifold patch, this number is somewhat arbitrary but it seems to give good coverage of the patch without being too slow.
    //also set backwards integration time and step size for coarse GPU search 
    const int n = grid_n * grid_n;
    const double T = 16.0;
    const int steps = 1000;
    const double dt = -T / static_cast<double>(steps);

    //search many patch radii
    const std::vector<double> radii = {
        1.0e-10, 3.0e-10,
        1.0e-9,  3.0e-9,
        1.0e-8,  3.0e-8,
        1.0e-7,  3.0e-7,
        1.0e-6,  3.0e-6,
        1.0e-5,  3.0e-5,
        1.0e-4,  3.0e-4,
        1.0e-3
    };

    //allocate output memory on GPU

    Candidate* d_out = nullptr;
    GPU_CHECK(cudaMalloc(&d_out, n * sizeof(Candidate)));

    std::vector<Candidate> h(n);
    std::vector<Candidate> all;
    all.reserve(n * radii.size());

    //cpu vectors to store results, we will copy the results from the GPU to these vectors and then sort them by distance to target to get good initial conditions for the Newton refinement step.
    const int block = 256;
    const int blocks = (n + block - 1) / block;
    
//This launches the GPU kernel. This is where the GPU actually starts doing the coarse candidate integrations.
    for (double radius : radii) {
        patch_kernel<<<blocks, block>>>(d_out,
                                        grid_n,
                                        radius,
                                        steps,
                                        dt,
                                        alpha,
                                        B,
                                        theta,
                                        phi);
        
        GPU_CHECK(cudaGetLastError());
        GPU_CHECK(cudaDeviceSynchronize());

        GPU_CHECK(cudaMemcpy(h.data(),
                             d_out,
                             n * sizeof(Candidate),
                             cudaMemcpyDeviceToHost));
//Copy candidate results from GPU memory back to CPU memory.
        for (const Candidate& c : h) {
            if (c.ok && std::isfinite(c.dist2)) {
                all.push_back(c);
            }
        }
    }

    GPU_CHECK(cudaFree(d_out));
//CPU sorts candidates by distance.So the GPU finds many crude candidates; the CPU decides which ones are good.
    std::sort(all.begin(), all.end(),
              [](const Candidate& x, const Candidate& y) {
                  return x.dist2 < y.dist2;
              });

    return all;
}

//kinda driver function.

static Candidate solve_for_well(const Basis& B,
                                double theta_eff,
                                double phi,
                                double alpha,
                                int max_trials) {
    std::vector<Candidate> seeds = gpu_patch_search(theta_eff, phi, alpha, B); //CPU asks GPU to generate seeds.

    Candidate best;
    best.a = 0.0; best.b = 0.0;
    best.th = 0.0; best.ph = 0.0;
    best.l1 = 0.0; best.l2 = 0.0;
    best.cost  = 1.0e300;
    best.dist2 = 1.0e300;
    best.ok = 0;

    const int trials = std::min<int>(max_trials, static_cast<int>(seeds.size()));
    //CPU refines the best few GPU seeds.
    for (int i = 0; i < trials; ++i) {
        Candidate c = refine_ab_newton(B,
                                       seeds[i].a,
                                       seeds[i].b,
                                       theta_eff,
                                       phi,
                                       alpha);

        if (c.ok && std::isfinite(c.dist2) && c.dist2 < best.dist2) {
            best = c;
        }
    }

    return best;
}

Result solve(double theta, double phi, double alpha) {
    //Main CPU solver. First check if we are already at the target, if so return zero cost solution.
    if (std::fabs(theta) < 1.0e-14 && std::fabs(phi) < 1.0e-14) {
        return {0.0, 0.0, 0.0};
    }

    const Basis B = make_basis_struct(alpha);
    //Because pendulum angle is periodic, it considers nearby wells:
    const double TWO_PI = 2.0 * M_PI;
    const int    k_round = static_cast<int>(std::lround(theta / TWO_PI));
    //Try several nearby angle shifts.
    int k_candidates_arr[] = {
        k_round,
        0,
        k_round - 1,
        k_round + 1,
        k_round - 2,
        k_round + 2
    };

    std::vector<int> k_candidates;
    for (int k : k_candidates_arr) {
        bool seen = false;
        for (int kk : k_candidates) {
            if (kk == k) { seen = true; break; }
        }
        if (!seen) k_candidates.push_back(k);
    }

    Candidate best_global;
    best_global.cost = 1.0e300;
    best_global.dist2 = 1.0e300;
    best_global.ok = 0;
    int best_k = 0;

    const double DIST2_OK = 1.0e-10;

    for (int k : k_candidates) {
        const double theta_eff = theta - TWO_PI * static_cast<double>(k);

        Candidate c = solve_for_well(B, theta_eff, phi, alpha, /*max_trials=*/6);

        const bool converged = c.ok
                            && std::isfinite(c.dist2)
                            && c.dist2 < DIST2_OK
                            && std::isfinite(c.cost);

        if (converged && c.cost < best_global.cost) {
            best_global = c;
            best_k = k;
        }
    }
    //Choose the converged trajectory with lowest cost.
    if (best_global.ok) {
        return {best_global.l1, best_global.l2, best_global.cost};
    }

    const Eigen::Matrix2d K = stable_gain(alpha);
    Eigen::Vector2d x;
    x << theta, phi;

    Eigen::Vector2d lambda = K * x;
    const double cost = 0.5 * (theta * lambda(0) + phi * lambda(1));

    return {lambda(0), lambda(1), cost};
}

std::vector<Result> solve_many(
    const std::vector<double>& theta,
    const std::vector<double>& phi,
    const std::vector<double>& alpha
) {
    int N = static_cast<int>(theta.size());
    std::vector<Result> results(N);

    for (int i = 0; i < N; ++i) {
        results[i] = solve(theta[i], phi[i], alpha[i]);
    }

    return results;
}
