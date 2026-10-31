#include "cart2sph.hpp"
#include "cart2sph_tables.hpp"

#include <cstddef>

namespace tlno::core::shell {

void cart2sph_block(int La, int Lb, const double* cart, int ldb, double* sph, double* scratch) {
    using cart2sph_detail::kC2SBlocks;
    const int nca = cart_count(La);
    const int nsa = sph_count(La);
    const int nsb = sph_count(Lb);
    const auto& blk_a = kC2SBlocks[La];
    const auto& blk_b = kC2SBlocks[Lb];

    // Pass 1 (right transform): tmp[ca, sb] = sum_cb cart[ca,cb] Csc_b[cb,sb].
    double* tmp = scratch;  // nca x nsb
    for (int i = 0; i < nca * nsb; ++i) tmp[i] = 0.0;
    for (int ca = 0; ca < nca; ++ca) {
        const double* crow = cart + static_cast<std::ptrdiff_t>(ca) * ldb;
        double* trow = tmp + static_cast<std::ptrdiff_t>(ca) * nsb;
        for (int e = 0; e < blk_b.count; ++e) {
            trow[blk_b.data[e].sph] += crow[blk_b.data[e].cart] * blk_b.data[e].coeff;
        }
    }

    // Pass 2 (left transform): sph[sa, sb] = sum_ca Csc_a[ca,sa] tmp[ca, sb].
    for (int i = 0; i < nsa * nsb; ++i) sph[i] = 0.0;
    for (int e = 0; e < blk_a.count; ++e) {
        const double* trow = tmp + static_cast<std::ptrdiff_t>(blk_a.data[e].cart) * nsb;
        double* srow = sph + static_cast<std::ptrdiff_t>(blk_a.data[e].sph) * nsb;
        const double c = blk_a.data[e].coeff;
        for (int sb = 0; sb < nsb; ++sb) {
            srow[sb] += c * trow[sb];
        }
    }
}

} // namespace tlno::core::shell
