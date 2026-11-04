#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_QUARTETS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_QUARTETS_IPP

// Shared SSE2/AVX2/AVX-512 contracted-quartet producer. Included in the same
// private scope as the geometry plan, Rys tables and matrix consumers.
// Scheduling metadata is O(shell_pairs); no molecular quartet list or N^4
// tensor is stored. Scratch holds at most one native-width shell block.

// Partition whole outer rows, preserving every SIMD packet and primitive
// rejection decision of the serial producer. Direct J uses deterministic
// weighted ranges; the cyclic option remains available for timing ablations.
#ifndef TLNO_NATIVE_J_WEIGHTED_ROWS
#define TLNO_NATIVE_J_WEIGHTED_ROWS 1
#endif
#if TLNO_NATIVE_J_WEIGHTED_ROWS != 0 && TLNO_NATIVE_J_WEIGHTED_ROWS != 1
#error "TLNO_NATIVE_J_WEIGHTED_ROWS must be 0 or 1"
#endif
struct NativeRowPartition { std::size_t rank = 0, count = 1; bool weighted = false; };
thread_local NativeRowPartition g_native_row_partition;
class NativeRowPartitionGuard {
    NativeRowPartition previous_;
public:
    NativeRowPartitionGuard(std::size_t rank, std::size_t count, bool weighted = false)
        : previous_(g_native_row_partition) { g_native_row_partition = {rank, count, weighted}; }
    ~NativeRowPartitionGuard() { g_native_row_partition = previous_; }
    NativeRowPartitionGuard(const NativeRowPartitionGuard&) = delete;
    NativeRowPartitionGuard& operator = (const NativeRowPartitionGuard&) = delete;
};

// Keep the two predicates independent. Folding per-pair Schwarz and decay
// values into one metric changes which quartets survive. The scoped context
// reaches every native packet layout without changing its primitive ordering.
struct NativeSecondaryShellScreen {
    const std::vector<double>* metric = nullptr;
    double threshold = 0.0;
};
thread_local const NativeSecondaryShellScreen* g_native_secondary_shell_screen = nullptr;

class NativeSecondaryShellScreenGuard {
    const NativeSecondaryShellScreen* previous_;
public:
    explicit NativeSecondaryShellScreenGuard(const NativeSecondaryShellScreen* screen)
        : previous_(g_native_secondary_shell_screen) { g_native_secondary_shell_screen = screen; }
    ~NativeSecondaryShellScreenGuard() { g_native_secondary_shell_screen = previous_; }
    NativeSecondaryShellScreenGuard(const NativeSecondaryShellScreenGuard&) = delete;
    NativeSecondaryShellScreenGuard& operator = (const NativeSecondaryShellScreenGuard&) = delete;
};

inline bool native_secondary_shell_live(const NativeSecondaryShellScreen* screen,
                                       std::uint32_t ab, std::uint32_t cd) noexcept {
    return screen == nullptr ||
        (*screen->metric)[ab] * (*screen->metric)[cd] >= screen->threshold;
}

// Retained ss|ss producer: native 1x16 improves all nine molecular GMs on
// Xeon 8370C (GM 1.014985x; 32/36 paired wins). 1x32 stays opt-in because
// deeper heterogeneous contractions regress. Zero restores the old producer.
// Tile sizes count ket pairs, not SIMD lanes.
#ifndef TLNO_NATIVE_SSSS_KET_TILE
#if defined(__AVX512F__) && defined(__FMA__)
#define TLNO_NATIVE_SSSS_KET_TILE 16
#else
#define TLNO_NATIVE_SSSS_KET_TILE 0
#endif
#endif
#if TLNO_NATIVE_SSSS_KET_TILE != 0 && TLNO_NATIVE_SSSS_KET_TILE != 16 && TLNO_NATIVE_SSSS_KET_TILE != 32
#error "TLNO_NATIVE_SSSS_KET_TILE must be 0, 16 or 32"
#endif

#ifndef TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
// Enable the validated native W8 cache; retain the original W4/SSE2 producer.
#if defined(__AVX512F__) && defined(__FMA__)
#define TLNO_NATIVE_RADIAL_PROJECTOR_CACHE 1
#else
#define TLNO_NATIVE_RADIAL_PROJECTOR_CACHE 0
#endif
#endif
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE != 0 && TLNO_NATIVE_RADIAL_PROJECTOR_CACHE != 1
#error "TLNO_NATIVE_RADIAL_PROJECTOR_CACHE must be 0 or 1"
#endif
#ifndef TLNO_NATIVE_RADIAL_PROJECTOR_CACHE_REPORT
#define TLNO_NATIVE_RADIAL_PROJECTOR_CACHE_REPORT 0
#endif
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
#include "native_radial_projector_cache.ipp"
#endif

struct NativePairBin {
    int la = 0, lb = 0;
    std::size_t primitives = 0;
    std::vector<std::uint32_t> ids;
    std::size_t stride = 0;
    std::vector<double> exponent, inverse, weight, px, py, pz, schwarz;
    std::vector<double> ax, ay, az, abx, aby, abz;
#if TLNO_NATIVE_SSSS_KET_TILE && defined(__AVX512F__) && defined(__FMA__)
    // Proved once at plan construction, never inferred from contraction depth.
    bool ssss_shared_exponents = false;
#endif
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
    // Immutable after schedule construction; basis identity owns the geometry.
    std::shared_ptr<const NativeRadialPairProjectorCache<tlno::core::simd::lanes>> radial_projector_cache;
#endif

    void pack(const BasisDataHGP& bd) {
        stride = (ids.size() + 7u) & ~std::size_t(7u);
        for (auto* v : {&exponent, &inverse, &weight, &px, &py, &pz, &schwarz})
            v->resize(primitives * stride);
        for (auto* v : {&ax, &ay, &az, &abx, &aby, &abz}) v->resize(stride);
        for (std::size_t i = 0; i < stride; ++i) {
            const auto id = ids[i < ids.size() ? i : 0];
            const auto [a, b] = bd.pair_groups[id];
            const auto& A = bd.groups[a].center; const auto& B = bd.groups[b].center;
            ax[i] = A[0]; ay[i] = A[1]; az[i] = A[2];
            abx[i] = A[0] - B[0]; aby[i] = A[1] - B[1]; abz[i] = A[2] - B[2];
            for (std::size_t k = 0; k < primitives; ++k) {
                const auto& q = bd.rys_pair_primitives[id][k]; const auto o = k * stride + i;
                exponent[o] = q.s; inverse[o] = q.invs; weight[o] = q.w;
                px[o] = q.Wx; py[o] = q.Wy; pz[o] = q.Wz; schwarz[o] = q.fast_schwarz;
            }
        }
#if TLNO_NATIVE_SSSS_KET_TILE && defined(__AVX512F__) && defined(__FMA__)
        ssss_shared_exponents = la == 0 && lb == 0 && !ids.empty();
        for (std::size_t k = 0; ssss_shared_exponents && k < primitives; ++k)
            for (std::size_t i = 1; i < ids.size(); ++i)
                if (exponent[k * stride + i] != exponent[k * stride]) {
                    ssss_shared_exponents = false; break;
                }
#endif
    }
};

#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
inline bool native_radial_cache_pair_is_used(const NativePairBin& a, const NativePairBin& b) {
    const int la = a.la, lb = a.lb, lc = b.la, ld = b.lb;
    if (la > 3 || lb > 3 || lc > 3 || ld > 3) return false;
    constexpr int W = tlno::core::simd::lanes;
    // Exact V4 route predicate: avoid materializing caches for bins whose
    // current consumers all bypass the radial projector (for example W8 ds).
    const bool fragment = W == 8 && la == 2 && lb == 1 &&
        ((lc == 2 && ld == 0) || (lc == 1 && ld == 1)) && a.ids.size() * b.ids.size() >= 256;
    const bool direct = W == 8 && ((la == 2 && lb == 0 && lc == 2 && ld == 0) ||
        (la == 2 && lb == 0 && lc == 1 && ld == 1) ||
        (la == 2 && lb <= 1 && lc <= 1 && ld == 0) || fragment);
    const int key = 10 * (la * (la + 1) / 2 + lb) + (lc * (lc + 1) / 2 + ld);
    return !direct && (la >= 2 || lc >= 2) && key != 53 && key != 63 &&
        key != 64 && key != 66 && key != 73;
}
#endif

struct NativeQuartetShape {
    int la, lb, lc, ld, lab, lcd, gk, roots, axis_size, lhs_cart, rhs_cart;
    std::vector<int> ix, iy, iz;
    NativeQuartetShape(int a, int b, int c, int d)
        : la(a), lb(b), lc(c), ld(d), lab(a + b), lcd(c + d), gk(c + d + 1),
          roots((a + b + c + d) / 2 + 1), axis_size((a + 1) * (b + 1) * (c + d + 1) * (d + 1)),
          lhs_cart(n_cart(a) * n_cart(b)), rhs_cart(n_cart(c) * n_cart(d)) {
        ix.reserve(lhs_cart * rhs_cart); iy.reserve(lhs_cart * rhs_cart); iz.reserve(lhs_cart * rhs_cart);
        for (int ai = 0; ai < n_cart(a); ++ai) for (int bi = 0; bi < n_cart(b); ++bi)
          for (int ci = 0; ci < n_cart(c); ++ci) for (int di = 0; di < n_cart(d); ++di) {
            const auto& av = g_cart.at(offset_for_L(a) + ai);
            const auto& bv = g_cart.at(offset_for_L(b) + bi);
            const auto& cv = g_cart.at(offset_for_L(c) + ci);
            const auto& dv = g_cart.at(offset_for_L(d) + di);
            ix.push_back(((av[0] * (b + 1) + bv[0]) * gk + cv[0]) * (d + 1) + dv[0]);
            iy.push_back(((av[1] * (b + 1) + bv[1]) * gk + cv[1]) * (d + 1) + dv[1]);
            iz.push_back(((av[2] * (b + 1) + bv[2]) * gk + cv[2]) * (d + 1) + dv[2]);
          }
    }
};

struct NativeQuartetSchedule {
    std::vector<NativePairBin> bins;
    std::map<int, NativeQuartetShape> shapes;
};

// In a sparse direct-J density, every quartet with a zero ket block needs
// only bra pairs whose density block is nonzero. Pack those pairs into full
// SIMD lanes once per P, without changing the immutable geometry schedule.
// The original position map preserves triangular enumeration within one bin.
struct NativeJCompactPairIndex {
    const NativePairBin* bins = nullptr;
    std::vector<NativePairBin> compact;
    std::vector<std::vector<std::size_t>> original_positions;
    std::vector<bool> compact_ready;
    bool enabled = false;

    void build(const BasisDataHGP& bd, const NativeQuartetSchedule& schedule,
               const double* rho) {
        bins = schedule.bins.data();
        compact.resize(schedule.bins.size());
        original_positions.resize(schedule.bins.size());
        compact_ready.resize(schedule.bins.size(), false);
        for (std::size_t b = 0; b < schedule.bins.size(); ++b) {
            const auto& source = schedule.bins[b];
            const auto& ids = source.ids;
            const bool has_zero = std::any_of(
                ids.begin(), ids.end(),
                [&](std::uint32_t id) { return rho[id] == 0.0; });
            if (!has_zero) continue;
            enabled = true;
            compact_ready[b] = true;
            auto& packed = compact[b];
            packed.la = source.la;
            packed.lb = source.lb;
            packed.primitives = source.primitives;
            auto& positions = original_positions[b];
            for (std::size_t i = 0; i < ids.size(); ++i) {
                if (rho[ids[i]] != 0.0) {
                    packed.ids.push_back(ids[i]);
                    positions.push_back(i);
                }
            }
            if (packed.ids.empty()) continue;
            packed.pack(bd);
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
            if (source.radial_projector_cache) {
                packed.radial_projector_cache =
                    std::make_shared<NativeRadialPairProjectorCache<
                        tlno::core::simd::lanes>>(
                        bd, packed.ids, packed.la, packed.lb);
            }
#endif
        }
    }

    const NativePairBin* compact_bin(const NativePairBin& bin) const {
        // Both the normal and equal-shape reversed producers pass bins from
        // this schedule; no compact bin is recursively dispatched.
        const std::size_t index = static_cast<std::size_t>(&bin - bins);
        return compact_ready[index] ? &compact[index] : nullptr;
    }
    const std::vector<std::size_t>* positions(const NativePairBin& bin) const {
        const std::size_t index = static_cast<std::size_t>(&bin - bins);
        return compact_ready[index] ? &original_positions[index] : nullptr;
    }
};

thread_local const NativeJCompactPairIndex* g_native_j_compact_pairs = nullptr;
class NativeJCompactPairGuard {
    const NativeJCompactPairIndex* previous_;
public:
    explicit NativeJCompactPairGuard(const NativeJCompactPairIndex* index)
        : previous_(g_native_j_compact_pairs) {
        g_native_j_compact_pairs = index;
    }
    ~NativeJCompactPairGuard() { g_native_j_compact_pairs = previous_; }
    NativeJCompactPairGuard(const NativeJCompactPairGuard&) = delete;
    NativeJCompactPairGuard& operator = (const NativeJCompactPairGuard&) = delete;
};

inline const NativePairBin* native_j_compact_bin(const NativePairBin& bin) {
    return g_native_j_compact_pairs
        ? g_native_j_compact_pairs->compact_bin(bin) : nullptr;
}
inline const std::vector<std::size_t>* native_j_compact_positions(
    const NativePairBin& bin) {
    return g_native_j_compact_pairs
        ? g_native_j_compact_pairs->positions(bin) : nullptr;
}

#ifndef TLNO_NATIVE_SHELL_PACKET_CULL
#define TLNO_NATIVE_SHELL_PACKET_CULL 1
#endif
#if TLNO_NATIVE_SHELL_PACKET_CULL != 0 && TLNO_NATIVE_SHELL_PACKET_CULL != 1
#error "TLNO_NATIVE_SHELL_PACKET_CULL must be 0 or 1"
#endif

#include "native_shell_packet_bounds.ipp"

// Call-local, immutable summaries are shared by all J workers. Keeping them
// outside the geometry plan avoids mutating cached schedule data when callers
// use different screening options or densities concurrently.
struct NativeShellPacketScreenIndex {
    const NativePairBin* bins = nullptr;
    std::vector<NativeShellPacketBounds> full, compact;
    bool enabled = false;

    void build(const NativeQuartetSchedule& schedule,
               const NativeJCompactPairIndex& compact_pairs,
               const std::vector<double>* metric, double threshold,
               const NativeSecondaryShellScreen* secondary) {
#if TLNO_NATIVE_SHELL_PACKET_CULL
        const auto* primary_metric = metric && threshold > 0.0 ? metric : nullptr;
        const auto* secondary_metric = secondary && secondary->threshold > 0.0
            ? secondary->metric : nullptr;
        if (!primary_metric && !secondary_metric) return;
        enabled = true;
        bins = schedule.bins.data();
        full.resize(schedule.bins.size());
        compact.resize(schedule.bins.size());
        for (std::size_t i = 0; i < schedule.bins.size(); ++i) {
            full[i].build(schedule.bins[i].ids, primary_metric,
                          secondary_metric, tlno::core::simd::lanes);
            if (compact_pairs.enabled && compact_pairs.compact_ready[i])
                compact[i].build(compact_pairs.compact[i].ids, primary_metric,
                                 secondary_metric, tlno::core::simd::lanes);
        }
#else
        (void)schedule; (void)compact_pairs; (void)metric;
        (void)threshold; (void)secondary;
#endif
    }

    const NativeShellPacketBounds* bounds(const NativePairBin& bin,
                                         bool use_compact) const {
        const std::size_t index = static_cast<std::size_t>(&bin - bins);
        return use_compact ? &compact[index] : &full[index];
    }
};

thread_local const NativeShellPacketScreenIndex* g_native_shell_packet_screen = nullptr;
class NativeShellPacketScreenGuard {
    const NativeShellPacketScreenIndex* previous_;
public:
    explicit NativeShellPacketScreenGuard(const NativeShellPacketScreenIndex* index)
        : previous_(g_native_shell_packet_screen) { g_native_shell_packet_screen = index; }
    ~NativeShellPacketScreenGuard() { g_native_shell_packet_screen = previous_; }
};

inline const NativeShellPacketBounds* native_shell_packet_bounds(
    const NativePairBin& bin, bool compact = false) {
    return g_native_shell_packet_screen
        ? g_native_shell_packet_screen->bounds(bin, compact) : nullptr;
}

struct NativeRowRange { std::size_t begin = 0, end = 0, step = 1; };

inline NativeRowRange native_uniform_row_range(std::size_t rows) {
    const auto& part = g_native_row_partition;
    if (part.count <= 1) return {0, rows, 1};
    if (!part.weighted) return {part.rank, rows, part.count};
    // Avoid multiplying the full row count by the worker rank.
    const auto boundary = [&](std::size_t rank) {
        return (rows / part.count) * rank + (rows % part.count) * rank / part.count;
    };
    return {boundary(part.rank), boundary(part.rank + 1), 1};
}

// Estimate emitted SIMD packets, including the exact-zero density shortcut.
// Each bin has fixed primitive depth and angular shape, so their common cost
// factors cancel when dividing its rows. Sparse triangular prefix sums need
// O(nonzero pairs) temporary storage; no quartet list is constructed. The
// ascending range and fixed rank reduction make every repeated call at a
// fixed thread count follow the same floating-point accumulation order.
inline NativeRowRange native_weighted_row_range(const NativePairBin& bra,
    const NativePairBin& ket, bool same_bin, std::size_t width,
    bool uses_compact_bra = true) {
    const auto& part = g_native_row_partition;
    const auto rows = ket.ids.size();
    if (part.count <= 1 || !part.weighted) return native_uniform_row_range(rows);
    const auto packets = [&](std::size_t n) {
        return n / width + (n % width != 0);
    };
    const auto* density = g_rys_density_screen;
    const auto* compact = uses_compact_bra && density && density->j_only
        ? native_j_compact_bin(bra) : nullptr;
    const auto* active = compact ? native_j_compact_positions(ket) : nullptr;
    // No compact ket means that all ket rows have nonzero density. For a
    // triangular bin the compact bra/ket and position map are identical.
    const auto full = packets(bra.ids.size());
    const auto sparse = compact ? packets(compact->ids.size()) : full;
    std::vector<long double> prefix;
    if (same_bin && active) {
        prefix.resize(active->size() + 1, 0.0L);
        for (std::size_t i = 0; i < active->size(); ++i) {
            const auto start = i ? (*active)[i - 1] + 1 : 0;
            const auto gap = (*active)[i] - start;
            prefix[i + 1] = prefix[i] + static_cast<long double>(gap) * (1 + packets(i)) +
                1 + packets((*active)[i] + 1);
        }
    }
    const auto work_before = [&](std::size_t end)->long double {
        if (!same_bin) {
            const auto live = active ? static_cast<std::size_t>(std::lower_bound(
                active->begin(), active->end(), end) - active->begin()) : end;
            return static_cast<long double>(end) * (1 + sparse) +
                static_cast<long double>(live) * (full - sparse);
        }
        if (active) {
            const auto live = static_cast<std::size_t>(std::lower_bound(
                active->begin(), active->end(), end) - active->begin());
            const auto start = live ? (*active)[live - 1] + 1 : 0;
            return prefix[live] + static_cast<long double>(end - start) * (1 + packets(live));
        }
        const auto groups = end / width, tail = end % width;
        return static_cast<long double>(end) +
            static_cast<long double>(width) * groups * (groups + 1) / 2 +
            static_cast<long double>(tail) * (groups + 1);
    };
    const auto total = work_before(rows);
    const auto boundary = [&](std::size_t rank) {
        if (rank == 0) return std::size_t(0);
        if (rank == part.count) return rows;
        const auto target = total * rank / part.count;
        std::size_t low = 0, high = rows;
        while (low < high) {
            const auto mid = low + (high - low) / 2;
            if (work_before(mid) < target) low = mid + 1; else high = mid;
        }
        // Select the closest whole-row boundary. Adjacent ranks evaluate
        // exactly the same boundary, including when a heavy row spans ranks.
        if (low && target - work_before(low - 1) <= work_before(low) - target) --low;
        return low;
    };
    return {boundary(part.rank), boundary(part.rank + 1), 1};
}

inline int native_shape_key(int a, int b, int c, int d) { return ((a * 5 + b) * 5 + c) * 5 + d; }

const NativeQuartetSchedule& get_native_quartet_schedule(const BasisDataHGP& bd) {
    std::lock_guard<std::mutex> guard(bd.native_schedule_mutex);
    if (bd.native_schedule) return *bd.native_schedule;
    auto result = std::make_shared<NativeQuartetSchedule>();
    std::map<std::tuple<int, int, std::size_t>, std::vector<std::uint32_t>> bins;
    for (std::size_t p = 0; p < bd.pair_groups.size(); ++p) {
        if (bd.packed_spherical_pair_components[p].empty()) continue;
        auto [a, b] = bd.pair_groups[p];
        bins[{bd.groups[a].L, bd.groups[b].L, bd.rys_pair_primitives[p].size()}].push_back(p);
    }
    for (auto& entry : bins) {
        auto [a, b, np] = entry.first;
        result->bins.push_back({a, b, np, std::move(entry.second)});
    }
    for (auto& bin : result->bins) bin.pack(bd);
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
    const auto cache_radial_bin = [&](std::size_t index) {
        auto& bin = result->bins[index];
        if (!bin.radial_projector_cache)
            bin.radial_projector_cache = std::make_shared<NativeRadialPairProjectorCache<tlno::core::simd::lanes>>(
                bd, bin.ids, bin.la, bin.lb);
    };
#endif
    for (std::size_t i = 0; i < result->bins.size(); ++i) for (std::size_t j = 0; j <= i; ++j) {
        const auto& a = result->bins[i]; const auto& b = result->bins[j];
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
        if (native_radial_cache_pair_is_used(a, b)) {
            cache_radial_bin(i);
            // V4 may reverse equal-angular cross bins. Both possible bra bins
            // therefore need caches; the retained orientation rule is unchanged.
            if (a.la == b.la && a.lb == b.lb) cache_radial_bin(j);
        }
#endif
        result->shapes.try_emplace(native_shape_key(a.la, a.lb, b.la, b.lb), a.la, a.lb, b.la, b.lb);
        (void)get_final_sph_pair_projector(a.la, a.lb);
        (void)get_final_sph_pair_projector(b.la, b.lb);
    }
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE && TLNO_NATIVE_RADIAL_PROJECTOR_CACHE_REPORT
    std::size_t cache_bins = 0, cache_packets = 0, payload = 0, capacity = 0;
    for (const auto& bin : result->bins) if (bin.radial_projector_cache) {
        ++cache_bins;
        cache_packets += bin.radial_projector_cache->packets.size();
        payload += bin.radial_projector_cache->payload_bytes();
        capacity += bin.radial_projector_cache->capacity_bytes();
    }
    std::fprintf(stderr, "NATIVE_RADIAL_PROJECTOR_CACHE width=%d bins=%zu packets=%zu payload_bytes=%zu capacity_bytes_excluding_allocator=%zu\n",
        tlno::core::simd::lanes, cache_bins, cache_packets, payload, capacity);
#endif
    bd.native_schedule = std::move(result);
    return *bd.native_schedule;
}

template<class V, class Shape>
inline void native_axis(const Shape& s, V c, V d, V b00,
                        V b10, V b01, V ab, V cd, V seed, V* out, V* g, V* bra) {
    g[0] = seed;
    for (int i = 0; i < s.lab; ++i) {
        V v = c * g[i * s.gk];
        if (i) v = V::fma(V(double(i)) * b10, g[(i - 1) * s.gk], v);
        g[(i + 1) * s.gk] = v;
    }
    for (int k = 0; k < s.lcd; ++k) for (int i = 0; i <= s.lab; ++i) {
        V v = d * g[i * s.gk + k];
        if (i) v = V::fma(V(double(i)) * b00, g[(i - 1) * s.gk + k], v);
        if (k) v = V::fma(V(double(k)) * b01, g[i * s.gk + k - 1], v);
        g[i * s.gk + k + 1] = v;
    }
    for (int i = 0; i <= s.lab; ++i) for (int k = 0; k <= s.lcd; ++k)
        bra[(i * (s.lb + 1)) * s.gk + k] = g[i * s.gk + k];
    for (int j = 1; j <= s.lb; ++j) for (int i = 0; i <= s.lab - j; ++i) for (int k = 0; k <= s.lcd; ++k)
        bra[(i * (s.lb + 1) + j) * s.gk + k] = V::fma(ab, bra[(i * (s.lb + 1) + j - 1) * s.gk + k],
                                                        bra[((i + 1) * (s.lb + 1) + j - 1) * s.gk + k]);
    for (int i = 0; i <= s.la; ++i) for (int j = 0; j <= s.lb; ++j) for (int k = 0; k <= s.lcd; ++k)
        out[((i * (s.lb + 1) + j) * s.gk + k) * (s.ld + 1)] = bra[(i * (s.lb + 1) + j) * s.gk + k];
    for (int l = 1; l <= s.ld; ++l) for (int i = 0; i <= s.la; ++i) for (int j = 0; j <= s.lb; ++j)
      for (int k = 0; k <= s.lcd - l; ++k) {
        const int base = ((i * (s.lb + 1) + j) * s.gk + k) * (s.ld + 1);
        out[base + l] = V::fma(cd, out[base + l - 1], out[base + s.ld + 1 + l - 1]);
      }
}

// Lane gathering happens at the plan/primitive boundary; all recurrence and
// projection arithmetic uses the selected native register width.
template<int W, class F>
inline tlno::core::simd::Packet<W> native_gather(F&& f) {
    double x[W]; for (int lane = 0; lane < W; ++lane) x[lane] = f(lane);
    return tlno::core::simd::Packet<W>::load(x);
}

inline bool native_density_live(const RysDensityScreenContext* ctx,
                               std::uint32_t ab, std::uint32_t cd, double q) {
    if (!ctx || !ctx->shell_rho) return true;
    // Both density blocks must be zero: either block alone can still receive
    // a nonzero Coulomb update from the opposite pair.
    if (ctx->j_only && ctx->shell_rho[ab] == 0.0 && ctx->shell_rho[cd] == 0.0)
        return false;
    if (ctx->density_break <= 0) return true;
    const double coul = 4.0 * std::max(ctx->shell_rho[ab], ctx->shell_rho[cd]);
    if (ctx->j_only) return q * coul >= ctx->density_break;
    const auto a = ctx->pair_group_left[ab], b = ctx->pair_group_right[ab];
    const auto c = ctx->pair_group_left[cd], d = ctx->pair_group_right[cd];
    const std::size_t n = ctx->ng;
    const double* den = ctx->group_pair_density_max;
    const double exchange = std::max(std::max(den[c * n + a], den[d * n + a]), std::max(den[c * n + b], den[d * n + b]));
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    if (ctx->k_only) return q * exchange >= ctx->density_break;
#endif
    return q * std::max(coul, exchange) >= ctx->density_break;
}

template<int W, class Emit>
void native_quartet_bin_pair(const BasisDataHGP& bd, const NativePairBin& lhs,
                             const NativePairBin& rhs, const NativeQuartetShape& s,
                             bool same_bin, const std::vector<double>* metric,
                             double threshold, double primitive_squared,
                             std::vector<double>& values) {
    using V = tlno::core::simd::Packet<W>;
    struct Scratch { std::vector<V> gx, gy, gz, g, bra, cart, mid, sph; };
    thread_local Scratch scratch;
    auto& t = scratch;
    t.gx.resize(s.axis_size); t.gy.resize(s.axis_size); t.gz.resize(s.axis_size);
    t.g.resize((s.lab + 1) * s.gk); t.bra.resize((s.lab + 1) * (s.lb + 1) * s.gk);
    t.cart.resize(s.lhs_cart * s.rhs_cart);
    const auto& lp = get_final_sph_pair_projector(s.la, s.lb);
    const auto& rp = get_final_sph_pair_projector(s.lc, s.ld);
    t.mid.resize(lp.sph_rows * s.rhs_cart); t.sph.resize(lp.sph_rows * rp.sph_rows);
    const auto* density = g_rys_density_screen;
    const auto* secondary = g_native_secondary_shell_screen;
    const auto* compact_bra = native_j_compact_bin(lhs);
    const auto* compact_positions = native_j_compact_positions(lhs);
    const auto* full_bounds = native_shell_packet_bounds(lhs);
    const auto* compact_bounds = compact_bra ? native_shell_packet_bounds(lhs, true) : nullptr;
    const double secondary_threshold = secondary ? secondary->threshold : 0.0;
    const auto rows = native_weighted_row_range(lhs, rhs, same_bin, W);
    for (std::size_t ki = rows.begin; ki < rows.end; ki += rows.step) {
      const auto ket_id = rhs.ids[ki];
      const bool ket_zero = density && density->j_only &&
          density->shell_rho[ket_id] == 0.0;
      const auto& bra = ket_zero && compact_bra ? *compact_bra : lhs;
      const auto* bounds = ket_zero && compact_bra ? compact_bounds : full_bounds;
      const double primary_other = metric ? (*metric)[ket_id] : 0.0;
      const double secondary_other = secondary ? (*secondary->metric)[ket_id] : 0.0;
      if (bounds && !bounds->row_live(primary_other, threshold,
                                    secondary_other, secondary_threshold)) continue;
      const auto [gc, gd] = bd.pair_groups[ket_id];
      const auto& C = bd.groups[gc].center; const auto& D = bd.groups[gd].center;
      const auto& kets = bd.rys_pair_primitives[ket_id];
      const std::size_t end = same_bin
          ? (ket_zero && compact_positions
                 ? static_cast<std::size_t>(std::upper_bound(
                       compact_positions->begin(), compact_positions->end(), ki)
                       -compact_positions->begin())
                 : ki + 1)
          : bra.ids.size();
      for (std::size_t start = 0; start < end; start += W) {
        if (bounds && !bounds->packet_live(start, primary_other, threshold,
                                         secondary_other, secondary_threshold)) continue;
        const int valid = static_cast<int>(std::min<std::size_t>(W, end - start));
        std::uint32_t ids[W]; bool active[W]; bool any = false;
        for (int lane = 0; lane < W; ++lane) {
            ids[lane] = bra.ids[start + (lane < valid ? lane : 0)];
            const double q = metric ? (*metric)[ids[lane]] * (*metric)[ket_id] : 0.0;
            active[lane] = lane < valid && (!metric || (q >= threshold &&
                native_secondary_shell_live(secondary, ids[lane], ket_id))) &&
                (!density || native_density_live(density, ids[lane], ket_id, q));
            any = any || active[lane];
        }
        if (!any) continue;
        V av[3], ab[3], cd[3];
        for (int axis = 0; axis < 3; ++axis) {
            av[axis] = native_gather<W>([&](int l) { return bd.groups[bd.pair_groups[ids[l]].first].center[axis]; });
            ab[axis] = av[axis] - native_gather<W>([&](int l) { return bd.groups[bd.pair_groups[ids[l]].second].center[axis]; });
            cd[axis] = V(C[axis] - D[axis]);
        }
        std::fill(t.cart.begin(), t.cart.end(), V());
        for (std::size_t ip = 0; ip < bra.primitives; ++ip) {
          const RysPrimitivePair* bps[W];
          for (int l = 0; l < W; ++l) bps[l] = &bd.rys_pair_primitives[ids[l]][ip];
          const V p = native_gather<W>([&](int l) { return bps[l]->s; });
          const V inv2p = native_gather<W>([&](int l) { return .5 * bps[l]->invs; });
          V pv[3] = {native_gather<W>([&](int l) { return bps[l]->Wx; }),
                   native_gather<W>([&](int l) { return bps[l]->Wy; }),
                   native_gather<W>([&](int l) { return bps[l]->Wz; })};
          const V bw = native_gather<W>([&](int l) { return active[l] ? bps[l]->w : 0.0; });
          for (const auto& kp : kets) {
            if (primitive_squared > 0) {
                bool primitive_live = false;
                for (int l = 0; l < valid; ++l) primitive_live = primitive_live ||
                    (active[l] && bps[l]->fast_schwarz * kp.fast_schwarz >= primitive_squared);
                if (!primitive_live) continue;
            }
            const V q(kp.s), sum = p + q, invsum = V(1.0) / sum;
            const V qo = q * invsum, po = p * invsum, inv2pq = V(.5) * invsum, inv2q(.5 * kp.invs);
            const V qv[3] = {V(kp.Wx), V(kp.Wy), V(kp.Wz)};
            const V pq[3] = {pv[0] - qv[0], pv[1] - qv[1], pv[2] - qv[2]};
            const V targ = p * q * invsum * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
            const V pref = V(kTwoPi52 * kp.w) * bw / V::sqrt(sum);
            double ts[W], roots[9][W], weights[9][W]; targ.store(ts);
            for (int lane = 0; lane < W; ++lane) {
                double rr[9], ww[9]; rys_roots_weights(s.roots, ts[lane], rr, ww);
                for (int r = 0; r < s.roots; ++r) { roots[r][lane] = rr[r]; weights[r][lane] = ww[r]; }
            }
            for (int r = 0; r < s.roots; ++r) {
                const V root = V::load(roots[r]);
                const V b00 = inv2pq * root, b10 = inv2p * (V(1.0) - qo * root), b01 = inv2q * (V(1.0) - po * root);
                V* axes[3] = {t.gx.data(), t.gy.data(), t.gz.data()};
                for (int axis = 0; axis < 3; ++axis)
                    native_axis(s, pv[axis] - av[axis] - qo * pq[axis] * root,
                        qv[axis] - V(C[axis]) + po * pq[axis] * root, b00, b10, b01, ab[axis], cd[axis],
                        axis == 2 ? pref * V::load(weights[r]) : V(1.0), axes[axis], t.g.data(), t.bra.data());
                for (std::size_t o = 0; o < t.cart.size(); ++o)
                    t.cart[o] = V::fma(t.gx[s.ix[o]] * t.gy[s.iy[o]], t.gz[s.iz[o]], t.cart[o]);
            }
          }
        }
        for (int i = 0; i < lp.sph_rows; ++i) {
            for (int j = 0; j < s.rhs_cart; ++j) {
                V v;
                for (const auto& term : lp.rows[i]) v = V::fma(V(term.coeff), t.cart[term.index * s.rhs_cart + j], v);
                t.mid[i * s.rhs_cart + j] = v;
            }
            for (int j = 0; j < rp.sph_rows; ++j) {
                V v;
                for (const auto& term : rp.rows[j]) v = V::fma(V(term.coeff), t.mid[i * s.rhs_cart + term.index], v);
                t.sph[i * rp.sph_rows + j] = v;
            }
        }
        // Explicit stores avoid aliasing a SIMD wrapper as a scalar array.
        thread_local std::vector<double> scalar;
        scalar.resize(t.sph.size() * W);
        for (std::size_t i = 0; i < t.sph.size(); ++i) t.sph[i].store(scalar.data() + i * W);
        for (int lane = 0; lane < valid; ++lane) {
            if (!active[lane]) continue;
            rys_emit_spherical_lane_block<Emit>(values,
                bd.packed_spherical_pair_components[ids[lane]],
                bd.packed_spherical_pair_components[ket_id], ids[lane] == ket_id,
                [&](const PackedPairComponentH& l, const PackedPairComponentH& r) {
                    return scalar[(std::size_t(l.component_index) * rp.sph_rows + r.component_index) * W + lane];
                });
        }
      }
    }
}

#ifndef TLNO_NATIVE_SPECIALIZED
#define TLNO_NATIVE_SPECIALIZED 1
#endif
#if TLNO_NATIVE_SPECIALIZED
#include "native_specialized.ipp"
#endif

#include "native_parallel_j.ipp"

template<class Emit>
EriPackedGData build_native_quartet_stream(const BasisDataHGP& bd,
    const EriPackedBuildOptions& options, bool materialize) {
#ifdef _OPENMP
    if (omp_get_max_threads() != 1 &&
       (!std::is_same_v<Emit, RysDirectJCoulombEmit> || materialize))
        throw std::runtime_error("native multithreading supports the direct Coulomb-J endpoint only");
#endif
    EriPackedGData result;
    EriStreamAccumulator accumulator;
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    EriMaterializedSphericalAudit audit;
    if (materialize) { const auto n = spherical_packed_value_count(bd); result.values.assign(n, 0); audit.reset(n); }
    EriMaterializedSphericalAuditGuard audit_guard(materialize ? &audit : nullptr);
#else
    if (materialize) throw std::runtime_error("materialized output requires validation build");
#endif
    EriStreamModeGuard stream_guard(!materialize, &bd.packed_spherical_pair_components, &accumulator);
    const auto& schedule = get_native_quartet_schedule(bd);
    NativeJCompactPairIndex compact_pairs;
    if constexpr (std::is_same_v<Emit, RysDirectJCoulombEmit>) {
        const auto* density = g_rys_density_screen;
        if (density && density->j_only && density->shell_rho)
            compact_pairs.build(bd, schedule, density->shell_rho);
    }
    const NativeJCompactPairGuard compact_guard(
        compact_pairs.enabled ? &compact_pairs : nullptr);
    const bool density_active = g_rys_density_screen && g_rys_density_screen->density_break > 0;
    const std::vector<double>* metric = nullptr;
    double threshold = options.schwarz_threshold;
    if (options.decay_threshold > 0 && density_active)
        throw std::runtime_error("density screening requires Schwarz, not a decay metric");
    const bool combined = threshold > 0 && options.decay_threshold > 0;
    if (threshold > 0 || density_active)
        metric = &get_rys_spherical_pair_schwarz_bounds(bd).pair_group_max;
    else if (options.decay_threshold > 0) {
        metric = &bd.pair_decay_bounds; threshold = options.decay_threshold;
    }
    const NativeSecondaryShellScreen secondary{&bd.pair_decay_bounds, options.decay_threshold};
    const NativeSecondaryShellScreenGuard secondary_guard(combined ? &secondary : nullptr);
    NativeShellPacketScreenIndex packet_screen;
    packet_screen.build(schedule, compact_pairs, metric, threshold,
                        combined ? &secondary : nullptr);
    const NativeShellPacketScreenGuard packet_screen_guard(
        packet_screen.enabled ? &packet_screen : nullptr);
    bool parallel_j = false;
#ifdef _OPENMP
    if constexpr (std::is_same_v<Emit, RysDirectJCoulombEmit>) {
        if (omp_get_max_threads() > 1) {
            accumulate_native_parallel_j(bd, schedule, metric, threshold,
                options.primitive_fastschwarz_threshold * options.primitive_fastschwarz_threshold,
                combined ? &secondary : nullptr);
            parallel_j = true;
        }
    }
#endif
    if (!parallel_j) for (std::size_t i = 0; i < schedule.bins.size(); ++i) for (std::size_t j = 0; j <= i; ++j) {
        const auto& lhs = schedule.bins[i]; const auto& rhs = schedule.bins[j];
        const auto& shape = schedule.shapes.at(native_shape_key(lhs.la, lhs.lb, rhs.la, rhs.lb));
#if TLNO_NATIVE_SPECIALIZED
        if (dispatch_native_specialized<tlno::core::simd::lanes, Emit>(bd, lhs, rhs, shape, i == j, metric, threshold,
            options.primitive_fastschwarz_threshold * options.primitive_fastschwarz_threshold, result.values)) continue;
#endif
        native_quartet_bin_pair<tlno::core::simd::lanes, Emit>(bd, lhs, rhs, shape, i == j, metric, threshold,
            options.primitive_fastschwarz_threshold * options.primitive_fastschwarz_threshold, result.values);
    }
    if (rys_spherical_screening_diagnostics_requested()) {
        std::uint64_t total = 0, primary_rejected = 0, secondary_rejected = 0;
        for (std::size_t ab = 0; ab < bd.pair_groups.size(); ++ab)
          for (std::size_t cd = 0; cd <= ab; ++cd) {
            ++total;
            if (metric && (*metric)[ab] * (*metric)[cd] < threshold) ++primary_rejected;
            else if (combined && !native_secondary_shell_live(&secondary, ab, cd)) ++secondary_rejected;
          }
        std::fprintf(stderr,
            "SPHSCREEN_DIAG metric=%s route=native lanes=%d "
            "schwarz_threshold=%.17g decay_threshold=%.17g primitive_fastschwarz_threshold=%.17g "
            "exact_total=%llu exact_primary_rejected=%llu exact_decay_rejected_after_primary=%llu\n",
            combined ? "schwarz+decay" : (options.decay_threshold > 0 ? "decay" : (options.schwarz_threshold > 0 ? "schwarz" : "none")),
            tlno::core::simd::lanes, options.schwarz_threshold, options.decay_threshold, options.primitive_fastschwarz_threshold,
            static_cast<unsigned long long>(total), static_cast<unsigned long long>(primary_rejected),
            static_cast<unsigned long long>(secondary_rejected));
    }
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    if (materialize) {
        const bool require_complete = options.schwarz_threshold == 0 && options.decay_threshold == 0 &&
                                      options.primitive_fastschwarz_threshold == 0 && !density_active;
        audit.validate(require_complete);
        std::fprintf(stderr,
            "MATERIALIZED_SPHERICAL_AUDIT expected=%zu written=%zu "
            "implicit_zeros=%zu require_complete=%d\n",
            audit.expected_count, audit.unique_writes,
            audit.expected_count - audit.unique_writes, int(require_complete));
        return result;
    }
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    const auto& fp = accumulator.fingerprint;
    result.values = {accumulator.checksum, double(fp.indexed_moment_1), double(fp.indexed_moment_2),
                   double(fp.indexed_moment_3), double(fp.canonical_count)};
#else
    result.values.assign(1, accumulator.checksum);
#endif
    return result;
}
#endif
