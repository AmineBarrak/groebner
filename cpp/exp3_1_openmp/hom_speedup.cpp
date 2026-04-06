/*
 * cpp/exp3_1_openmp/hom_speedup.cpp
 * ==================================
 * EXPERIMENT 3.1 — Homogenization Speedup (OpenMP)
 *
 * Measures S(p) = T(1)/T(p) for p = 1, 2, 4, 8, 16
 * across n = 10, 20, 40, 80, 160 generators.
 *
 * Build:
 *   g++ -O3 -fopenmp -std=c++17 -I.. hom_speedup.cpp -o hom_speedup
 *
 * Run:
 *   OMP_NUM_THREADS=8 ./hom_speedup
 *   Results printed to stdout and written to ../../results/exp3_1_cpp.csv
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>     // for setw, setprecision
#include <filesystem>  // for create_directories
#include <map>         // for std::map
#include <omp.h>
#include "../polynomial.h"


/* ─────────────────────────────────────────────────────────────────────────
 * Sequential homogenization of one generator:
 *   x_i - t^{qi}  →  alpha^{qi} * x_i - t^{qi}
 *
 * For the monomial curve family this is trivial (just set the alpha exponent).
 * We simulate realistic work by also computing a substitution expansion.
 * ───────────────────────────────────────────────────────────────────────── */
static Poly homogenize_one(int i, int n_vars) {
    // Generator:  alpha^{i+1} * x_i  -  t^{i+1}
    Poly f(n_vars);
    {
        Monomial m(n_vars, 0);
        m[0]   = i + 1;   // alpha
        m[2+i] = 1;       // x_i
        f.add_term(m, 1.0);
    }
    {
        Monomial m(n_vars, 0);
        m[1] = i + 1;     // t
        f.add_term(m, -1.0);
    }
    // Simulate additional work proportional to degree (α-clearing step)
    // In a real implementation this involves rational arithmetic cancellation.
    volatile double acc = 0.0;
    for (int k = 0; k < (i+1) * 50; ++k) acc += std::sin(k * 0.001);
    (void)acc;
    return f;
}


/* ─────────────────────────────────────────────────────────────────────────
 * Sequential homogenization of all n generators
 * ───────────────────────────────────────────────────────────────────────── */
static std::vector<Poly> homogenize_sequential(int n) {
    int n_vars = 2 + n;
    std::vector<Poly> result(n);
    for (int i = 0; i < n; ++i)
        result[i] = homogenize_one(i, n_vars);
    return result;
}


/* ─────────────────────────────────────────────────────────────────────────
 * Parallel homogenization with p threads (OpenMP)
 * ───────────────────────────────────────────────────────────────────────── */
static std::vector<Poly> homogenize_parallel(int n, int p) {
    int n_vars = 2 + n;
    std::vector<Poly> result(n, Poly(n_vars));
    omp_set_num_threads(p);
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < n; ++i)
        result[i] = homogenize_one(i, n_vars);
    return result;
}


/* ─────────────────────────────────────────────────────────────────────────
 * Measure wall time in seconds
 * ───────────────────────────────────────────────────────────────────────── */
using Clock = std::chrono::high_resolution_clock;
static double elapsed(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}


/* ─────────────────────────────────────────────────────────────────────────
 * Main
 * ───────────────────────────────────────────────────────────────────────── */
int main() {
    // n_list: number of generators
    std::vector<int> n_list  = {10, 20, 40, 80, 160, 320};
    // p_list: processor counts to test (cap at available cores)
    int max_threads = omp_get_max_threads();
    std::vector<int> p_list;
    for (int p : {1, 2, 4, 8, 16}) {
        if (p <= max_threads) p_list.push_back(p);
    }
    int n_repeats = 5;

    std::cout << "Experiment 3.1 — Homogenization Speedup (C++/OpenMP)\n";
    std::cout << "Max threads available: " << max_threads << "\n";
    std::cout << std::string(70, '-') << "\n";
    std::cout << std::left
              << std::setw(8)  << "n"
              << std::setw(6)  << "p"
              << std::setw(14) << "avg_time_ms"
              << std::setw(12) << "speedup"
              << std::setw(12) << "efficiency"
              << "\n";
    std::cout << std::string(70, '-') << "\n";

    // Store results for CSV
    struct Row {
        int n, p;
        double avg_ms, speedup, efficiency;
    };
    std::vector<Row> rows;
    std::map<int, double> baseline; // n -> T(1) in ms

    for (int n : n_list) {
        for (int p : p_list) {
            std::vector<double> times;
            for (int r = 0; r < n_repeats; ++r) {
                auto t0 = Clock::now();
                if (p == 1)
                    homogenize_sequential(n);
                else
                    homogenize_parallel(n, p);
                times.push_back(elapsed(t0) * 1000.0);  // ms
            }
            double avg = std::accumulate(times.begin(), times.end(), 0.0)
                         / times.size();
            if (p == 1) baseline[n] = avg;

            double sp  = baseline.count(n) ? baseline[n] / avg : 1.0;
            double eff = sp / p;

            rows.push_back({n, p, avg, sp, eff});

            std::cout << std::left << std::fixed << std::setprecision(3)
                      << std::setw(8)  << n
                      << std::setw(6)  << p
                      << std::setw(14) << avg
                      << std::setw(12) << sp
                      << std::setw(12) << eff
                      << "\n";
        }
        std::cout << "\n";
    }

    // Write CSV
    std::filesystem::create_directories("../../results");
    std::ofstream csv("../../results/exp3_1_cpp.csv");
    csv << "n_generators,n_processors,avg_time_ms,speedup,efficiency\n";
    for (const auto& row : rows)
        csv << row.n << "," << row.p << ","
            << row.avg_ms << "," << row.speedup << "," << row.efficiency << "\n";
    csv.close();
    std::cout << "\nSaved → ../../results/exp3_1_cpp.csv\n";
    return 0;
}
