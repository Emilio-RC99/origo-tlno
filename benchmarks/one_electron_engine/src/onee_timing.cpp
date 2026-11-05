#include "core/basis/basis_general.hpp"
#include "benchmarks/common/molecule_io.hpp"
#include "integrals/one_electron/onee_matrices.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

double median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    const double hi = values[mid];
    if (values.size() % 2 == 1) {
        return hi;
    }
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1), values.end());
    return 0.5 * (values[mid - 1] + hi);
}

double mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    const double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

double trace_of(const tlno::core::basis::DenseMatrix& m) {
    double out = 0.0;
    for (int i = 0; i < m.size; ++i) {
        out += m(i, i);
    }
    return out;
}

double frob_norm(const tlno::core::basis::DenseMatrix& m) {
    double out = 0.0;
    for (int i = 0; i < m.size; ++i) {
        for (int j = 0; j < m.size; ++j) {
            const double v = m(i, j);
            out += v * v;
        }
    }
    return std::sqrt(out);
}

void write_matrix_tsv(
    const std::string& path,
    const tlno::core::basis::DenseMatrix& matrix) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write matrix file: " + path);
    }
    out << std::setprecision(17);
    for (int row = 0; row < matrix.size; ++row) {
        for (int col = 0; col < matrix.size; ++col) {
            if (col != 0) {
                out << '\t';
            }
            out << matrix(row, col);
        }
        out << '\n';
    }
}

} //namespace

int main(int argc, char** argv) {
    try {
        std::string inp_path;
        std::string xyz_path;
        std::string basis_name = "631gd";
        int charge = 0;
        bool charge_explicit = false;
        int repeats = 3;
        int warmup = 1;
        std::string timing_scope = "full";
        bool quiet = false;
        std::string dump_prefix;

        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (a == "--inp" && i + 1 < argc) {
                inp_path = argv[++i];
            } else if (a == "--xyz" && i + 1 < argc) {
                xyz_path = argv[++i];
            } else if (a == "--charge" && i + 1 < argc) {
                charge = std::stoi(argv[++i]);
                charge_explicit = true;
            } else if (a == "--basis" && i + 1 < argc) {
                basis_name = argv[++i];
            } else if (a == "--repeats" && i + 1 < argc) {
                repeats = std::max(1, std::stoi(argv[++i]));
            } else if (a == "--warmup" && i + 1 < argc) {
                warmup = std::max(0, std::stoi(argv[++i]));
            } else if (a == "--timing-scope" && i + 1 < argc) {
                timing_scope = argv[++i];
            } else if (a == "--quiet") {
                quiet = true;
            } else if (a == "--dump-prefix" && i + 1 < argc) {
                dump_prefix = argv[++i];
            } else if (a == "--help" || a == "-h") {
                std::cout
                    << "Usage: onee_timing (--inp file.inp | --xyz file.xyz) "
                    << "[--charge q] "
                    << "[--basis sto3g|431g|631g|631gd|631gdp|def2svp|def2tzvp] "
                    << "[--repeats n] [--warmup n] "
                    << "[--timing-scope full|cached] [--quiet] "
                    << "[--dump-prefix path]\n"
                    << "Coordinates are interpreted as bohr.\n";
                return 0;
            } else {
                throw std::runtime_error("Unknown or incomplete argument: " + a);
            }
        }

        if (inp_path.empty() == xyz_path.empty()) {
            throw std::runtime_error("use exactly one of --inp <file.inp> or --xyz <file.xyz>");
        }
        if (timing_scope != "full" && timing_scope != "cached") {
            throw std::runtime_error("--timing-scope must be full or cached");
        }

        tlno::benchmarks::onee::MoleculeInput mol =
            !inp_path.empty()
                ? tlno::benchmarks::onee::read_inp_file(inp_path)
                : tlno::benchmarks::onee::read_xyz_file(xyz_path, charge);
        if (charge_explicit) {
            mol.charge = charge;
        }

        tlno::core::molecule::MoleculeSpec spec;
        spec.symbols = mol.symbols;
        spec.coords = mol.coords_bohr;
        spec.charge = mol.charge;

        const tlno::core::basis::BasisKind bk = tlno::core::basis::parse_basis_kind(basis_name);
        const tlno::core::basis::BasisBuildResult br = tlno::core::basis::build_basis_general(spec, bk);

        if (!quiet) {
            std::cout << "input=" << (!inp_path.empty() ? inp_path : xyz_path) << "\n";
        }
        std::cout << "basis=" << basis_name
                  << "  charge=" << mol.charge
                  << "  multiplicity=" << mol.multiplicity
                  << "  atoms=" << mol.symbols.size()
                  << "  basis_functions=" << br.basis.size()
                  << "  electrons=" << br.electron_count
                  << "  impl=shell_block"
                  << "  timing_scope=" << timing_scope << "\n";
#ifdef _OPENMP
        if (!quiet) {
            std::cout << "omp_max_threads = " << omp_get_max_threads() << "\n";
        }
#endif
        if (!quiet) {
            std::cout << "warmup = " << warmup << " repeats = " << repeats << "\n";
        }

        tlno::integrals::onee::OneElectronPlan plan;
        if (timing_scope == "cached") {
            plan = tlno::integrals::onee::build_one_electron_plan(br);
        }
        const auto run_once = [&]() {
            if (timing_scope == "cached") {
                return tlno::integrals::onee::evaluate_one_electron_matrices(br, plan);
            }
            return tlno::integrals::onee::build_one_electron_matrices(br);
        };

        for (int w = 0; w < warmup; ++w) {
            const auto tw0 = std::chrono::steady_clock::now();
            const auto mats = run_once();
            const auto tw1 = std::chrono::steady_clock::now();
            if (!quiet) {
                const double sec = std::chrono::duration<double>(tw1 - tw0).count();
                const double trace_h = mats.kinetic(0, 0) + mats.nuclear(0, 0);
                std::cout << "warmup_run=" << (w + 1)
                          << "  onee_s=" << std::fixed << std::setprecision(8) << sec
                          << "  sample_h00=" << trace_h << "\n";
            }
        }

        std::vector<double> times;
        times.reserve(static_cast<std::size_t>(repeats));
        for (int r = 0; r < repeats; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            const auto mats = run_once();
            const auto t1 = std::chrono::steady_clock::now();
            const double sec = std::chrono::duration<double>(t1 - t0).count();
            times.push_back(sec);
            if (!quiet) {
                const double sample_h00 = mats.kinetic(0, 0) + mats.nuclear(0, 0);
                std::cout << "run=" << (r + 1)
                          << "  onee_s=" << std::fixed << std::setprecision(8) << sec
                          << "  sample_h00=" << sample_h00 << "\n";
            }

            if (r == 0 && !quiet) {
                std::cout << "fingerprint"
                          << "  trS=" << std::setprecision(12) << trace_of(mats.overlap)
                          << "  trT=" << trace_of(mats.kinetic)
                          << "  trV=" << trace_of(mats.nuclear)
                          << "  fnS=" << frob_norm(mats.overlap)
                          << "  fnT=" << frob_norm(mats.kinetic)
                          << "  fnV=" << frob_norm(mats.nuclear)
                          << "\n";
                if (!dump_prefix.empty()) {
                    write_matrix_tsv(dump_prefix + "_S.tsv", mats.overlap);
                    write_matrix_tsv(dump_prefix + "_T.tsv", mats.kinetic);
                    write_matrix_tsv(dump_prefix + "_V.tsv", mats.nuclear);
                }
            }
        }

        std::cout << "median  onee_s=" << std::fixed << std::setprecision(8) << median(times) << "\n";
        if (!quiet) {
            std::cout << "mean    onee_s=" << std::fixed << std::setprecision(8) << mean(times) << "\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
