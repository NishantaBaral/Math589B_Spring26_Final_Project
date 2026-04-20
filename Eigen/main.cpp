#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <iostream>

int main() {
    Eigen::Matrix4d A;

    // Example matrix (students can replace this later)
    A << 0.0,  1.0,  0.0,  0.0,
         1.0, -0.1,  0.0, -1.0,
        -1.0,  0.0,  0.0, -1.0,
         0.0, -1.0, -1.0,  0.1;

    Eigen::EigenSolver<Eigen::Matrix4d> es(A);

    [[maybe_unused]] auto evals = es.eigenvalues();
    [[maybe_unused]] auto evecs = es.eigenvectors();

    std::cout << "Eigenvalues:\n";
    for (int i = 0; i < evals.size(); ++i) {
        std::cout << evals[i] << "\n";
    }

    std::cout << "\nStable eigenvalues (Re < 0):\n";
    for (int i = 0; i < evals.size(); ++i) {
        if (evals[i].real() < 0.0) {
            std::cout << evals[i] << "\n";
        }
    }

    return 0;
}
