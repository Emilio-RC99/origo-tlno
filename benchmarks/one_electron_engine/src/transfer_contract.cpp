// Numerical contracts for SIMD width, mixed Boys domains, charge tails and
// the compact nuclear recurrence. Run with/without sanitizers and for each ISA.
#include "integrals/one_electron/onee_integrals.hpp"
#include "core/math/general_math.hpp"
#include "core/shell/shell_contract.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tlno::integrals::onee;
using tlno::core::basis::RawShell;
using tlno::core::basis::Shell;
using tlno::core::shell::make_shell;
using tlno::core::math::boys_r000_all;
using tlno::core::math::boys_r000_all_batch;

static void close(double got, double ref, double tolerance, const char* what) {
    if (!std::isfinite(got) || std::abs(got - ref) > tolerance * (1.0 + std::abs(ref))) {
        std::fprintf(stderr, "%s: %.17g != %.17g\n", what, got, ref);
        std::exit(1);
    }
}

template<int W> static void seeds() {
    const double xs[] = {0.0, 1e-16, 0.009999999, 0.01, 0.010000001,
                        0.5, 3.0, 11.999999999, 12.0, 12.000000001,
                        39.99, 40.0, 100.0, 1e4};
    for (int n : {0,1,2,3,4,6,8,16,32,33}) {
        for (double p : {0.01, 0.5, 3.0, 80.0}) {
            for (int pattern = 0; pattern < 2; ++pattern) for (int base = 0; base < 14; ++base) {
                double x[W];
                for (int l = 0; l < W; ++l) x[l] = xs[(base + pattern * 3 * l) % 14];
                std::vector<double> out((n+1) * W + 2, -999.0), ref(n + 1);
                boys_r000_all_batch<W>(n, x, p, out.data() + 1);
                close(out.front(), -999.0,0.0, "leading sentinel");
                close(out.back(), -999.0, 0.0, "trailing sentinel");
                for (int l = 0; l < W; ++l) {
                    boys_r000_all(n, x[l], p, ref.data());
                    for (int k = 0; k <= n; ++k)
                        close(out[1 + k * W + l], ref[k], 2e-11, "packet Boys");
                }
            }
        }
    }
}

static ShellGroup group(int L, std::array<double, 3> center, int depth) {
    ShellGroup g;
    g.L = L; g.center = center;
    for (int i = 0; i < depth; ++i) g.exponents.push_back(0.18 + 0.7 * i);
    for (int x = L; x >= 0; --x) for (int y = L - x; y >= 0; --y) {
        RawShell raw;
        raw.ang={x, y, L - x - y}; raw.exponents = g.exponents;
        for (int i = 0; i < depth; ++i) raw.coefficients.push_back(i % 2 ? -0.2 : 0.7);
        const auto sh = make_shell(raw, center);
        g.ang.push_back(raw.ang);
        g.basis_indices.push_back(static_cast<int>(g.basis_indices.size()));
        g.coeffs.emplace_back();
        for (const auto& p : sh.primitives) g.coeffs.back().push_back(p.coefficient);
    }
    return g;
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--dump") {
        set_onee_nuclear_screen_eps(0.0);
        std::vector<int> charges;
        std::vector<std::array<double,3>> centers;
        for (int k = 0; k < 17; ++k) {
            charges.push_back(k % 5 == 1 ? 0 : (k % 3 ? 1 : -2));
            centers.push_back({0.1 * k, 0.7 * k, -0.2 * k});
        }
        for (int la = 0; la <= 4; ++la) for (int lb = 0; lb <= 4; ++lb) {
            auto a = group(la, {0.0, 0.0, 0.0}, 2), b = group(lb, {0.3, -0.8, 1.1}, 3);
            auto ab = build_shell_pair_prim(a, b);
            if ((la + lb) % 2 == 0) cache_shell_pair_nucleus_hermite(a, b, ab);
            const int na = a.ang.size(), nb = b.ang.size();
            std::vector<double> s(na*nb), t(na*nb), v(na*nb);
            contracted_shell_block_overlap_kinetic(a, b, ab, s.data(), t.data(), 0.0);
            contracted_shell_block_nuclear(a, b, ab, v.data(), charges, centers);
            for (int i = 0; i < na; ++i) for (int j = 0; j < nb; ++j) {
                const int k = i * nb + j;
                std::printf("%d %d %d %d %.17g %.17g %.17g\n", la, lb, i, j, s[k], t[k], v[k]);
            }
        }
        return 0;
    }
    seeds<1>(); seeds<2>(); seeds<4>(); seeds<8>();
    set_onee_nuclear_screen_eps(0.0);
    // All orientations through g exercise both specialized and generic paths.
    for (int la = 0; la <= 4; ++la) for (int lb = 0; lb <= 4; ++lb) {
        auto a = group(la, {0.0,0.0,0.0}, 2), b = group(lb, {0.3,-0.8,1.1}, 3);
        auto ab = build_shell_pair_prim(a,b), ba = build_shell_pair_prim(b,a);
        auto prepared = build_shell_pair_prim(a,b);
        cache_shell_pair_nucleus_hermite(a, b, prepared);
        auto cached = std::move(prepared);
        const int na = a.ang.size(), nb = b.ang.size(), count = na*nb;
        for (int nk : {0,1,2,3,4,5,7,8,9,15,16,17}) {
            std::vector<int> charges;
            std::vector<std::array<double,3>> centers;
            std::vector<double> sum(count), batch(count), reverse(count), reused(count);
            for (int k = 0; k < nk; ++k) {
                charges.push_back(k % 5 == 1 ? 0 : (k % 3 ? 1 : -2));
                centers.push_back({0.1 * k, 0.7 * k, -0.2 * k});
                contracted_shell_block_nuclear(a, b, ab, sum.data(), {charges.back()}, {centers.back()});
            }
            contracted_shell_block_nuclear(a, b, ab, batch.data(), charges, centers);
            contracted_shell_block_nuclear(b, a, ba, reverse.data(), charges, centers);
            contracted_shell_block_nuclear(a, b, cached, reused.data(), charges, centers);
            for (int i = 0; i < na; ++i) for (int j = 0; j < nb; ++j) {
                close(batch[i*nb+j], sum[i*nb+j], 2e-11, "charge additivity/tail");
                close(batch[i*nb+j], reverse[j*na+i], 2e-11,"bra/ket symmetry");
                close(batch[i*nb+j], reused[i*nb+j], 2e-13, "cached projectors/move");
            }
            // The block API accumulates into its output; it must not clear it.
            contracted_shell_block_nuclear(a, b, ab, batch.data(), charges, centers);
            for (int i = 0; i < count; ++i) close(batch[i], 2 * sum[i], 3e-11, "additive output");
        }
    }
    std::puts("PASS: widths 1/2/4/8, Boys domains, s-through-g, charge tails, symmetry, accumulation, cached projectors/move");
}
