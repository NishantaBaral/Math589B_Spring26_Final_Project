#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include "solver.hpp"

int main(int argc, char** argv) {
    if (argc == 4) {
        double theta = std::atof(argv[1]);
        double phi   = std::atof(argv[2]);
        double alpha = std::atof(argv[3]);
        Result r = solve(theta, phi, alpha);
        std::printf("%.10f %.10f %.10f\n", r.l1, r.l2, r.cost);
        return 0;
    }

    if (argc == 2) {
        std::ifstream infile(argv[1]);
        if (!infile) {
            std::fprintf(stderr, "could not open input file\n");
            return 1;
        }
        int N;
        infile >> N;
        std::vector<double> theta(N);
        std::vector<double> phi(N);
        std::vector<double> alpha(N);
        for (int i = 0; i < N; i++) {
            infile >> theta[i] >> phi[i] >> alpha[i];
        }
        std::vector<Result> results = solve_many(theta, phi, alpha);
        for (int i = 0; i < N; i++) {
            std::printf("%.10f %.10f %.10f\n",
                results[i].l1,
                results[i].l2,
                results[i].cost
            );
        }
        return 0;
    }

    std::fprintf(stderr, "usage:\n");
    std::fprintf(stderr, "  ./solver theta phi alpha\n");
    std::fprintf(stderr, "  ./solver input.txt\n");
    return 1;
}
