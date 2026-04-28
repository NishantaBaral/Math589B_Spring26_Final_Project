#pragma once

#include <vector>

struct Result {
    double l1;
    double l2;
    double cost;
};

Result solve(double theta, double phi, double alpha);

std::vector<Result> solve_many(
    const std::vector<double>& theta,
    const std::vector<double>& phi,
    const std::vector<double>& alpha
);