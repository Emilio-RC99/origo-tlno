#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_RADIAL_PROJECTOR_CACHE_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_RADIAL_PROJECTOR_CACHE_IPP

// Schedule-owned radial bra projectors for the exact V4 producer. This file is
// included in rys_engine's private scope after BasisDataHGP is complete.
// No recurrence or density-dependent data is cached here.
struct NativeRadialProjectorPrefix {
    std::vector<int> offsets, columns;
};

template<int W>
struct NativeRadialProjectorPacketCache {
    using V = tlno::core::simd::Packet<W>;
    std::vector<V> projector;
    // Same-bin triangular packets can have any valid prefix. Preserve the
    // original prefix-specific sparse column union and its ascending order.
    // Equal unions share one view rather than duplicating W metadata arrays.
    std::array<unsigned char, W> prefix_index{};
    std::vector<NativeRadialProjectorPrefix> prefixes;
};

template<int W>
struct NativeRadialPairProjectorCache {
    using V = tlno::core::simd::Packet<W>;
    using Packet = NativeRadialProjectorPacketCache<W>;
    int rows = 0, br = 0;
    std::vector<Packet> packets;

    NativeRadialPairProjectorCache(const BasisDataHGP& bd,
        const std::vector<std::uint32_t>& ids, int la, int lb)
        : rows(get_final_sph_pair_projector(la, lb).sph_rows),
          br(n_cart_cum(la + lb) - offset_for_L(la)) {
        static_assert(W == 2 || W == 4 || W == 8);
        packets.resize((ids.size() + W - 1) / W);
        const std::size_t extent = std::size_t(rows) * br;
        std::vector<double> coeff(extent * W);
        std::vector<unsigned> nonzero(extent);
        for (std::size_t pi = 0; pi < packets.size(); ++pi) {
            auto& packet = packets[pi];
            const std::size_t start = pi * W;
            const int valid = int(std::min<std::size_t>(W, ids.size() - start));
            std::fill(coeff.begin(), coeff.end(), 0.0);
            for (int lane = 0; lane < W; ++lane) {
                const auto id = ids[start + (lane < valid ? lane : 0)];
                const auto& source = bd.rys_radial_pair_projectors[id].bra_rows;
                for (int row = 0; row < rows; ++row) for (const auto& term : source[row])
                    coeff[(std::size_t(row) * br + term.index) * W + lane] = term.coeff;
            }
            packet.projector.resize(extent);
            for (std::size_t i = 0; i < extent; ++i) {
                const double* c = coeff.data() + i * W;
                packet.projector[i] = V::load(c);
                unsigned bits = 0;
                for (int lane = 0; lane < valid; ++lane)
                    if (c[lane] != 0.0) bits |= 1u << lane;
                nonzero[i] = bits;
            }
            for (int prefix = 1; prefix <= valid; ++prefix) {
                NativeRadialProjectorPrefix view;
                view.offsets.resize(rows + 1);
                view.offsets[0] = 0;
                const unsigned mask = (1u << prefix) - 1u;
                for (int row = 0; row < rows; ++row) {
                    for (int a = 0; a < br; ++a)
                        if (nonzero[std::size_t(row) * br + a] & mask)
                            view.columns.push_back(a);
                    view.offsets[row + 1] = int(view.columns.size());
                }
                if (packet.prefixes.empty() ||
                   view.offsets != packet.prefixes.back().offsets ||
                   view.columns != packet.prefixes.back().columns)
                    packet.prefixes.push_back(std::move(view));
                packet.prefix_index[prefix - 1] = static_cast<unsigned char>(packet.prefixes.size() - 1);
            }
        }
    }

    // Requested vector payload; excludes allocator/control-block overhead.
    std::size_t payload_bytes() const {
        std::size_t bytes = 0;
        for (const auto& p : packets) {
            bytes += p.projector.size() * sizeof(V);
            for (const auto& v : p.prefixes)
                bytes += (v.offsets.size() + v.columns.size()) * sizeof(int);
        }
        return bytes;
    }
    // Owned object and vector capacities; allocator headers/control blocks are
    // not observable here and are deliberately not claimed in this estimate.
    std::size_t capacity_bytes() const {
        std::size_t bytes = sizeof(*this) + packets.capacity() * sizeof(Packet);
        for (const auto& p : packets) {
            bytes += p.projector.capacity() * sizeof(V);
            bytes += p.prefixes.capacity() * sizeof(NativeRadialProjectorPrefix);
            for (const auto& v : p.prefixes)
                bytes += (v.offsets.capacity() + v.columns.capacity()) * sizeof(int);
        }
        return bytes;
    }
};
#endif
