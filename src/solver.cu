#include "solver.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

using namespace std;
using namespace Eigen;

Result solve(double theta, double phi) {
    Matrix2d A;
    A << 0,  1,
         1, -0.1;

    cout << "--- HPC Eigen Test ---" << endl;
    cout << "Inputs -> Theta: " << theta << ", Phi: " << phi << endl;
    cout << "Matrix A:" << endl << A << endl;

    EigenSolver<Matrix2d> solver(A);
    cout << "Eigenvalues:" << endl << solver.eigenvalues() << endl;
    cout << "----------------------" << endl;

    Result r;
    r.l1 = 0.0;
    r.l2 = 0.0;
    r.cost = 0.0;

    return r;
}