#ifndef TLNO_CORE_SIMD_PACKET_HPP
#define TLNO_CORE_SIMD_PACKET_HPP

#include <array>
#include <cmath>
#if defined(__SSE2__)
#include <immintrin.h>
#endif

// Shared arithmetic for W independent doubles. Engines assign meaning to lanes.
// Loads and stores accept buffers without SIMD alignment guarantees.
// The generic template supplies scalar fallbacks for widths 1, 2, 4, and 8.
namespace tlno::core::simd {

template<int W> struct Packet {
    static_assert(W == 1 || W == 2 || W == 4 || W == 8, "Unsupported packet width.");
    std::array<double, W> value{};

    Packet() = default;
    explicit Packet(double x) { value.fill(x); }

    static Packet load(const double* p) {
        Packet r;
        for (int i = 0; i < W; ++i) r.value[i] = p[i];
        return r;
    }

    static unsigned less_mask(const double* p, double threshold) {
        unsigned mask = 0;
        for (int i = 0; i < W; ++i) {
            mask |= static_cast<unsigned>(p[i] < threshold) << i;
        }
        return mask;
    }

    void store(double* p) const {
        for (int i = 0; i < W; ++i) p[i] = value[i];
    }

    friend Packet operator+(Packet a, Packet b) {
        for (int i = 0; i < W; ++i) a.value[i] += b.value[i];
        return a;
    }

    friend Packet operator-(Packet a, Packet b) {
        for (int i = 0; i < W; ++i) a.value[i] -= b.value[i];
        return a;
    }

    friend Packet operator*(Packet a, Packet b) {
        for (int i = 0; i < W; ++i) a.value[i] *= b.value[i];
        return a;
    }

    friend Packet operator/(Packet a, Packet b) {
        for (int i = 0; i < W; ++i) a.value[i] /= b.value[i];
        return a;
    }

    static Packet sqrt(Packet a) {
        for (int i = 0; i < W; ++i) a.value[i] = std::sqrt(a.value[i]);
        return a;
    }

    // Preserve the original scalar behavior; fused rounding is not guaranteed.
    static Packet fma(Packet a, Packet b, Packet c) { return a * b + c; }
};

#if defined(__SSE2__)
template<> struct alignas(16) Packet<2> {
    __m128d value;

    Packet() : value(_mm_setzero_pd()) {}
    explicit Packet(double x) : value(_mm_set1_pd(x)) {}
    explicit Packet(__m128d x) : value(x) {}

    static Packet load(const double* p) { return Packet(_mm_loadu_pd(p)); }
    static unsigned less_mask(const double* p, double threshold) {
        return static_cast<unsigned>(
            _mm_movemask_pd(_mm_cmplt_pd(_mm_loadu_pd(p), _mm_set1_pd(threshold))));
    }
    void store(double* p) const { _mm_storeu_pd(p, value); }

    friend Packet operator+(Packet a, Packet b) { return Packet(_mm_add_pd(a.value, b.value)); }
    friend Packet operator-(Packet a, Packet b) { return Packet(_mm_sub_pd(a.value, b.value)); }
    friend Packet operator*(Packet a, Packet b) { return Packet(_mm_mul_pd(a.value, b.value)); }
    friend Packet operator/(Packet a, Packet b) { return Packet(_mm_div_pd(a.value, b.value)); }
    static Packet sqrt(Packet a) { return Packet(_mm_sqrt_pd(a.value)); }
    static Packet fma(Packet a, Packet b, Packet c) { return a * b + c; }
};
#endif

#if defined(__AVX2__) && defined(__FMA__)
template<> struct alignas(32) Packet<4> {
    __m256d value;

    Packet() : value(_mm256_setzero_pd()) {}
    explicit Packet(double x) : value(_mm256_set1_pd(x)) {}
    explicit Packet(__m256d x) : value(x) {}

    static Packet load(const double* p) { return Packet(_mm256_loadu_pd(p)); }
    static unsigned less_mask(const double* p, double threshold) {
        return static_cast<unsigned>(_mm256_movemask_pd(
            _mm256_cmp_pd(_mm256_loadu_pd(p), _mm256_set1_pd(threshold), _CMP_LT_OQ)));
    }
    void store(double* p) const { _mm256_storeu_pd(p, value); }

    friend Packet operator+(Packet a, Packet b) { return Packet(_mm256_add_pd(a.value, b.value)); }
    friend Packet operator-(Packet a, Packet b) { return Packet(_mm256_sub_pd(a.value, b.value)); }
    friend Packet operator*(Packet a, Packet b) { return Packet(_mm256_mul_pd(a.value, b.value)); }
    friend Packet operator/(Packet a, Packet b) { return Packet(_mm256_div_pd(a.value, b.value)); }
    static Packet sqrt(Packet a) { return Packet(_mm256_sqrt_pd(a.value)); }
    static Packet fma(Packet a, Packet b, Packet c) {
        return Packet(_mm256_fmadd_pd(a.value, b.value, c.value));
    }
};
#endif

#if defined(__AVX512F__) && defined(__FMA__)
template<> struct alignas(64) Packet<8> {
    __m512d value;

    Packet() : value(_mm512_setzero_pd()) {}
    explicit Packet(double x) : value(_mm512_set1_pd(x)) {}
    explicit Packet(__m512d x) : value(x) {}

    static Packet load(const double* p) { return Packet(_mm512_loadu_pd(p)); }
    static unsigned less_mask(const double* p, double threshold) {
        return static_cast<unsigned>(
            _mm512_cmp_pd_mask(_mm512_loadu_pd(p), _mm512_set1_pd(threshold), _CMP_LT_OQ));
    }
    void store(double* p) const { _mm512_storeu_pd(p, value); }

    friend Packet operator+(Packet a, Packet b) { return Packet(_mm512_add_pd(a.value, b.value)); }
    friend Packet operator-(Packet a, Packet b) { return Packet(_mm512_sub_pd(a.value, b.value)); }
    friend Packet operator*(Packet a, Packet b) { return Packet(_mm512_mul_pd(a.value, b.value)); }
    friend Packet operator/(Packet a, Packet b) { return Packet(_mm512_div_pd(a.value, b.value)); }
    static Packet sqrt(Packet a) { return Packet(_mm512_sqrt_pd(a.value)); }
    static Packet fma(Packet a, Packet b, Packet c) {
        return Packet(_mm512_fmadd_pd(a.value, b.value, c.value));
    }
};
#endif

#if defined(__AVX512F__) && defined(__FMA__)
inline constexpr int lanes = 8;
inline constexpr const char* isa_name = "avx512";
#elif defined(__AVX2__) && defined(__FMA__)
inline constexpr int lanes = 4;
inline constexpr const char* isa_name = "avx2";
#elif defined(__SSE2__)
inline constexpr int lanes = 2;
inline constexpr const char* isa_name = "sse2";
#else
inline constexpr int lanes = 1;
inline constexpr const char* isa_name = "scalar";
#endif

using NativePacket = Packet<lanes>;

} // namespace tlno::core::simd

#endif
