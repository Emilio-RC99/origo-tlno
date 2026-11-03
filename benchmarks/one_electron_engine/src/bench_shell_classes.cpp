// Per-shell-class kernel microbenchmark.
//
// Times TLNO's specialized contracted shell-block kernels for the diagonal
// angular-momentum classes (s|s), (p|p), (d|d), (f|f), decoupled from molecule
// composition and screening.  All classes use the SAME primitive set and the
// same group separation, so only L (and the Cartesian block size) changes.
// Output: nanoseconds per shell-block call and per Cartesian integral, for the
// fused overlap+kinetic kernel and for the nuclear kernel.  A matched Libcint
// timing is produced separately by bench_shell_classes_libcint.py.
#include "integrals/one_electron/onee_integrals.hpp"
#include "core/shell/shell_contract.hpp"
#include "core/basis/types.hpp"

#include <array>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace tlno::integrals::onee;
using tlno::core::basis::RawShell;
using tlno::core::basis::Shell;
using tlno::core::shell::make_shell;

namespace {

using Clock = std::chrono::steady_clock;

std::vector<std::array<int, 3>> cart_components(int L) {
    std::vector<std::array<int, 3>> out;
    for (int l = L; l >= 0; --l)
        for (int m = L - l; m >= 0; --m)
            out.push_back({l, m, L - l - m});
    return out;
}

// Synthetic shell group of momentum L with a shared primitive set, normalized
// per Cartesian component exactly as the production basis builder does.
ShellGroup make_group(int L, std::array<double, 3> center,
                      const std::vector<double>& exps,
                      const std::vector<double>& raw_coeffs, int& idx) {
    ShellGroup g;
    g.center = center;
    g.L = L;
    g.exponents = exps;
    for (const auto& a : cart_components(L)) {
        RawShell rs;
        rs.label = "x";
        rs.ang = a;
        rs.exponents = exps;
        rs.coefficients = raw_coeffs;
        Shell sh = make_shell(rs, center);
        std::vector<double> cc;
        for (const auto& p : sh.primitives) cc.push_back(p.coefficient);
        g.ang.push_back(a);
        g.coeffs.push_back(std::move(cc));
        g.basis_indices.push_back(idx++);
    }
    return g;
}

template <typename Body>
double time_loop(Body&& body, double budget_s) {
    // Warm up, then run until the time budget is met; return ns per call.
    for (int i = 0; i < 200; ++i) body();
    long reps = 0;
    auto t0 = Clock::now();
    double elapsed = 0.0;
    while (elapsed < budget_s) {
        for (int i = 0; i < 500; ++i) body();
        reps += 500;
        elapsed = std::chrono::duration<double>(Clock::now() - t0).count();
    }
    return elapsed * 1e9 / static_cast<double>(reps);
}

} // namespace

int main () {
    // Fixed three-primitive contraction and geometry shared by every class.
    const std::vector<double> exps{5.0, 1.2, 0.35};
    const std::vector<double> raw{0.15, 0.50, 0.40};
    // A 12-carbon line cluster (spacing 2.5 bohr); the timed shell pair sits on
    // the first two centres, and the nuclear kernel sums over all 12 nuclei so
    // the setup matches the Libcint cluster in bench_shell_classes_libcint.py.
    std::vector<int> Z;
    std::vector<std::array<double, 3>> nuc;
    for (int i = 0; i < 12; ++i) { nuc.push_back({2.5 * i, 0.0, 0.0}); Z.push_back(6); }
    const std::array<double, 3> A = nuc[0];
    const std::array<double, 3> B = nuc[1];

    std::printf("class  Ncart  st_ns_block  v_ns_block  st_ns_int  v_ns_int\n");
    const char* names[] = {"s|s", "p|p", "d|d", "f|f"};
    for (int L = 0; L <= 3; ++L) {
        int idx = 0;
        ShellGroup ga = make_group(L, A, exps, raw, idx);
        ShellGroup gb = make_group(L, B, exps, raw, idx);
        ShellPairPrim spp = build_shell_pair_prim(ga, gb);
        const int na = static_cast<int>(ga.ang.size());
        const int nb = static_cast<int>(gb.ang.size());
        const int nint = na * nb;
        std::vector<double> ov(nint), ki(nint), nuclear(nint);

        double st_ns = time_loop([&] {
            contracted_shell_block_overlap_kinetic(ga, gb, spp, ov.data(),
                                                   ki.data(), 0.0);
        }, 0.3);
        double v_ns = time_loop([&] {
            contracted_shell_block_nuclear(ga, gb, spp, nuclear.data(), Z, nuc);
        }, 0.3);

        std::printf("%-5s  %5d  %11.1f  %10.1f  %9.3f  %8.3f\n",
                    names[L], nint, st_ns, v_ns,
                    st_ns / nint, v_ns / nint);
    }
    return 0;
}
