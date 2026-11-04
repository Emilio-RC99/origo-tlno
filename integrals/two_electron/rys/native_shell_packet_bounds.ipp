#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SHELL_PACKET_BOUNDS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SHELL_PACKET_BOUNDS_IPP

// Conservative shell-screen envelopes over the existing packet membership.
// No pair is sorted, moved, or removed from a packet that reaches the kernel.
// Primary Schwarz and secondary decay are bounded independently: combining
// their values would change the original conjunction of two predicates.
struct NativeShellPacketBounds {
    std::vector<double> primary, secondary;
    double primary_max = 0.0, secondary_max = 0.0;
    std::size_t width = 1;

    void build(const std::vector<std::uint32_t>& ids,
               const std::vector<double>* primary_metric,
               const std::vector<double>* secondary_metric,
               std::size_t packet_width) {
        primary.clear(); secondary.clear();
        primary_max = 0.0; secondary_max = 0.0;
        width = packet_width;
        const std::size_t packets = (ids.size() + width - 1) / width;
        if (primary_metric) primary.assign(packets, 0.0);
        if (secondary_metric) secondary.assign(packets, 0.0);
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const std::size_t packet = i / width;
            if (primary_metric) {
                const double value = (*primary_metric)[ids[i]];
                primary[packet] = std::max(primary[packet], value);
                primary_max = std::max(primary_max, value);
            }
            if (secondary_metric) {
                const double value = (*secondary_metric)[ids[i]];
                secondary[packet] = std::max(secondary[packet], value);
                secondary_max = std::max(secondary_max, value);
            }
        }
    }

    bool row_live(double primary_other, double primary_threshold,
                  double secondary_other, double secondary_threshold) const {
        return (primary.empty() || primary_max * primary_other >= primary_threshold) &&
               (secondary.empty() || secondary_max * secondary_other >= secondary_threshold);
    }

    bool packet_live(std::size_t start, double primary_other,
                     double primary_threshold, double secondary_other,
                     double secondary_threshold) const {
        const std::size_t packet = start / width;
        return (primary.empty() || primary[packet] * primary_other >= primary_threshold) &&
               (secondary.empty() || secondary[packet] * secondary_other >= secondary_threshold);
    }
};
#endif
