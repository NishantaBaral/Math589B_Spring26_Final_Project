#include <cstdio>
#include <cstdlib>
#include "solver.hpp"

int main(int argc, char** argv) {
    if (argc != 3) {
        std::fprintf(stderr, "usage: ./solver theta phi\n");
        return 1;
    }

    double theta = std::atof(argv[1]);
    double phi   = std::atof(argv[2]);

    Result r = solve(theta, phi);

    std::printf("%.10f %.10f %.10f\n", r.l1, r.l2, r.cost);
    return 0;
}
