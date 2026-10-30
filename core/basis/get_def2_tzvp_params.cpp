#include "get_def2_tzvp_params.hpp"
#include "cartesian_shells.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace tlno::core::basis {
namespace {

std::string uppercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

using detail::add_cartesian_shells;

} // namespace

// BSE def2-TZVP version 1; tabulated parameters for H-Ar.
// https://www.basissetexchange.org/api/basis/def2-tzvp/format/orca/?elements=1-18&version=1
// Retain Cartesian 6d/10f components and labels; order contractions by L,
// then decreasing leading exponent (preserving the original sulfur d order).
RawAtomBasis get_def2_tzvp_params(const std::string& symbol) {
    RawAtomBasis atom;
    const std::string upper = uppercase(symbol);

    if (upper == "H") {
        atom.charge = 1;
        atom.shells.reserve(6);
        add_cartesian_shells(atom, 0, 1, {34.0613410, 5.1235746, 1.1646626}, 
                            {0.0060251978, 0.045021094, 0.20189726});
        add_cartesian_shells(atom, 0, 2, {0.32723041}, 
                            {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.10307241}, 
                            {1.0000000});
        add_cartesian_shells(atom, 1, 1, {0.8000000}, 
                            {1.0000000});
        return atom;
    }
    if (upper == "HE") {
        atom.charge = 2;
        atom.shells.reserve(6);
        add_cartesian_shells(atom, 0, 1, {98.078321616, 14.764404247, 3.3185831473}, 
                            {0.0075803064967, 0.054848620937, 0.22074382186});
        add_cartesian_shells(atom, 0, 2, {0.87413869551}, 
                            {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.24459897208}, 
                            {1.0000000});
        add_cartesian_shells(atom, 1, 1, {1.0000000}, 
                            {1.0000000});
        return atom;
    }
    if (upper == "LI") {
        atom.charge = 3;
        atom.shells.reserve(14);
        add_cartesian_shells(atom, 0, 1, {6269.2628010, 940.31612431, 214.22107528, 60.759840184, 19.915152032, 7.3171509797}, 
                            {0.00020540968826, 0.0015916554089, 0.0082869829707, 0.033856374249, 0.11103225876, 0.27449383329});
        add_cartesian_shells(atom, 0, 2, {2.9724674216, 1.2639852314}, 
                            {0.23792456411, 0.30765411924});
        add_cartesian_shells(atom, 0, 3, {0.51427489953}, 
                            {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.077030885901}, 
                            {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.028938896433}, 
                            {1.0000000});
        add_cartesian_shells(atom, 1, 1, {3.3270000}, 
                            {1.0000000});
        add_cartesian_shells(atom, 1, 2, {0.4000000}, 
                            {1.0000000});
        add_cartesian_shells(atom, 1, 3, {0.0600000}, 
                            {1.0000000});
        return atom;
    }
    if (upper == "BE") {
        atom.charge = 4;
        atom.shells.reserve(20);
        add_cartesian_shells(atom, 0, 1, {4700.2365626, 704.82845622, 160.43110478, 45.425347336, 14.798334125, 5.3512452537},
                             {0.00023584389316, 0.0018243791019, 0.0093966148224, 0.036908924159, 0.10897561281, 0.21694284551});
        add_cartesian_shells(atom, 0, 2, {2.1542044819, 0.93363744400},
                             {0.44695408857, 0.20866985771});
        add_cartesian_shells(atom, 0, 3, {0.18791432989},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.074648267947},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.032650484598},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {3.6316917145},
                             {-0.029033998305});
        add_cartesian_shells(atom, 1, 2, {0.71695694366, 0.19541932860},
                             {-0.16877854032, -0.51403419628});
        add_cartesian_shells(atom, 1, 3, {0.060515465890},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.1800000},
                             {1.0000000});
        return atom;
    }
    if (upper == "B") {
        atom.charge = 5;
        atom.shells.reserve(36);
        add_cartesian_shells(atom, 0, 1, {8564.8660687, 1284.1516263, 292.27871604, 82.775469176, 27.017939269, 9.8149619660},
                             {0.00022837198155, 0.0017682576447, 0.0091407080516, 0.036342638989, 0.11063458441, 0.23367344321});
        add_cartesian_shells(atom, 0, 2, {3.9318559059, 1.6595599712},
                             {0.41818777978, 0.22325473798});
        add_cartesian_shells(atom, 0, 3, {0.35762965239},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.14246277496},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.060560594768},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {22.453875803, 5.1045058330, 1.4986081344, 0.50927831315},
                             {0.0050265575179, 0.032801738965, 0.13151230768, 0.33197167769});
        add_cartesian_shells(atom, 1, 2, {0.18147077798},
                             {0.47314319570});
        add_cartesian_shells(atom, 1, 3, {0.064621893904},
                             {0.25802783943});
        add_cartesian_shells(atom, 2, 1, {0.66100000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 2, {0.19900000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {0.49000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "C") {
        atom.charge = 6;
        atom.shells.reserve(36);
        add_cartesian_shells(atom, 0, 1, {13575.3496820, 2035.2333680, 463.22562359, 131.20019598, 42.853015891, 15.584185766},
                             {0.00022245814352, 0.0017232738252, 0.0089255715314, 0.035727984502, 0.11076259931, 0.24295627626});
        add_cartesian_shells(atom, 0, 2, {6.2067138508, 2.5764896527},
                             {0.41440263448, 0.23744968655});
        add_cartesian_shells(atom, 0, 3, {0.57696339419},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.22972831358},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.095164440028},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {34.697232244, 7.9582622826, 2.3780826883, 0.81433208183},
                             {0.0053333657805, 0.035864109092, 0.14215873329, 0.34270471845});
        add_cartesian_shells(atom, 1, 2, {0.28887547253},
                             {0.46445822433});
        add_cartesian_shells(atom, 1, 3, {0.10056823671},
                             {0.24955789874});
        add_cartesian_shells(atom, 2, 1, {1.09700000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 2, {0.31800000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {0.76100000},
                             {1.0000000});
        return atom;
    }
    if (upper == "N") {
        atom.charge = 7;
        atom.shells.reserve(36);
        add_cartesian_shells(atom, 0, 1, {19730.8006470, 2957.8958745, 673.22133595, 190.68249494, 62.295441898, 22.654161182},
                             {0.00021887984991, 0.0016960708803, 0.0087954603538, 0.035359382605, 0.11095789217, 0.24982972552});
        add_cartesian_shells(atom, 0, 2, {8.9791477428, 3.6863002370},
                             {0.40623896148, 0.24338217176});
        add_cartesian_shells(atom, 0, 3, {0.84660076805},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.33647133771},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.13647653675},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {49.200380510, 11.346790537, 3.4273972411, 1.1785525134},
                             {0.0055552416751, 0.038052379723, 0.14953671029, 0.34949305230});
        add_cartesian_shells(atom, 1, 2, {0.41642204972},
                             {0.45843153697});
        add_cartesian_shells(atom, 1, 3, {0.14260826011},
                             {0.24428771672});
        add_cartesian_shells(atom, 2, 1, {1.65400000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 2, {0.46900000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {1.09300000},
                             {1.0000000});
        return atom;
    }
    if (upper == "O") {
        atom.charge = 8;
        atom.shells.reserve(36);
        add_cartesian_shells(atom, 0, 1, {27032.3826310, 4052.3871392, 922.32722710, 261.24070989, 85.354641351, 31.035035245},
                             {0.00021726302465, 0.0016838662199, 0.0087395616265, 0.035239968808, 0.11153519115, 0.25588953961});
        add_cartesian_shells(atom, 0, 2, {12.260860728, 4.9987076005},
                             {0.39768730901, 0.24627849430});
        add_cartesian_shells(atom, 0, 3, {1.1703108158},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.46474740994},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.18504536357},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {63.274954801, 14.627049379, 4.4501223456, 1.5275799647},
                             {0.0060685103418, 0.041912575824, 0.16153841088, 0.35706951311});
        add_cartesian_shells(atom, 1, 2, {0.52935117943},
                             {0.44794207502});
        add_cartesian_shells(atom, 1, 3, {0.17478421270},
                             {0.24446069663});
        add_cartesian_shells(atom, 2, 1, {2.31400000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 2, {0.64500000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {1.42800000},
                             {1.0000000});
        return atom;
    }
    if (upper == "F") {
        atom.charge = 9;
        atom.shells.reserve(36);
        add_cartesian_shells(atom, 0, 1, {35479.1004410, 5318.4728983, 1210.4810975, 342.85518140, 112.01943181, 40.714740248},
                             {0.00021545014888, 0.0016700686527, 0.0086733211476, 0.035049933175, 0.11165320133, 0.25988506647});
        add_cartesian_shells(atom, 0, 2, {16.039678111, 6.5038186740},
                             {0.39422966880, 0.24998238551});
        add_cartesian_shells(atom, 0, 3, {1.5440477509},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.61223452862},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.24027979698},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {80.233900483, 18.594010743, 5.6867902653, 1.9511006294},
                             {0.0063685999134, 0.044303143530, 0.16867248708, 0.36166346255});
        add_cartesian_shells(atom, 1, 2, {0.66970211298},
                             {0.44202901491});
        add_cartesian_shells(atom, 1, 3, {0.21651300410},
                             {0.24319875730});
        add_cartesian_shells(atom, 2, 1, {3.10700000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 2, {0.85500000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {1.91700000},
                             {1.0000000});
        return atom;
    }
    if (upper == "NE") {
        atom.charge = 10;
        atom.shells.reserve(36);
        add_cartesian_shells(atom, 0, 1, {45069.4640220, 6755.9768656, 1537.6502864, 435.51697667, 142.28655638, 51.692153804},
                             {0.00021687155182, 0.0016812736757, 0.0087356062782, 0.035361266922, 0.11321521454, 0.26654653104});
        add_cartesian_shells(atom, 0, 2, {20.315870490, 8.2021942646},
                             {0.39631959951, 0.25582811251});
        add_cartesian_shells(atom, 0, 3, {1.9681276278},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.77904756001},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.30229502043},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {99.782996032, 23.176124101, 7.1163945872, 2.4418711435},
                             {0.0065569234163, 0.045888009138, 0.17331287812, 0.36475267512});
        add_cartesian_shells(atom, 1, 2, {0.83389605766},
                             {0.43831075171});
        add_cartesian_shells(atom, 1, 3, {0.26607311301},
                             {0.24160029835});
        add_cartesian_shells(atom, 2, 1, {4.01400000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 2, {1.09600000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {2.54400000},
                             {1.0000000});
        return atom;
    }
    if (upper == "NA") {
        atom.charge = 11;
        atom.shells.reserve(35);
        add_cartesian_shells(atom, 0, 1, {26041.1099270, 3906.1268548, 888.97454993, 251.45497961, 81.650143512, 28.904158401, 10.625782932},
                             {0.00061806342811, 0.0047748604414, 0.024471684829, 0.094755394977, 0.26867496920, 0.47925475440, 0.33248591469});
        add_cartesian_shells(atom, 0, 2, {53.769410179, 16.308243025, 2.3730384125},
                             {0.019527731872, 0.092648010794, -0.39938670172});
        add_cartesian_shells(atom, 0, 3, {0.95730772603, 0.40806460959},
                             {1.6428595391, 0.55692596966});
        add_cartesian_shells(atom, 0, 4, {0.049967582329},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.019268616250},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {138.07979989, 32.232700393, 9.9816075360, 3.4822033928, 1.2299134620},
                             {0.0057951891929, 0.041620846251, 0.16281916885, 0.36011784647, 0.44858979889});
        add_cartesian_shells(atom, 1, 2, {0.41743959423},
                             {0.23040759559});
        add_cartesian_shells(atom, 1, 3, {0.0910000},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 4, {0.0300000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {2.6090000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 2, {0.4300000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 3, {0.1000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "MG") {
        atom.charge = 12;
        atom.shells.reserve(35);
        add_cartesian_shells(atom, 0, 1, {31438.3495550, 4715.5153354, 1073.1629247, 303.57238768, 98.626251042, 34.943808417, 12.859785199},
                             {0.00060912311326, 0.0047066196465, 0.024135820657, 0.093628959834, 0.26646742093, 0.47890929917, 0.33698490286});
        add_cartesian_shells(atom, 0, 2, {64.876913004, 19.725520777, 2.8951804339},
                             {0.019180889307, 0.090913704392, -0.39563756125});
        add_cartesian_shells(atom, 0, 3, {1.1960454710, 0.54329451156},
                             {1.6827603373, 0.52141091954});
        add_cartesian_shells(atom, 0, 4, {0.10099104092},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.036865728085},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {179.87189612, 42.120069376, 13.120503032, 4.6257503609, 1.6695211016},
                             {0.0053799549018, 0.039318014098, 0.15740129476, 0.35919094128, 0.45533379310});
        add_cartesian_shells(atom, 1, 2, {0.58551012105},
                             {0.21986432910});
        add_cartesian_shells(atom, 1, 3, {0.18914796195},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 4, {0.053768755187},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {3.4440000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 2, {0.2900000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 3, {0.0700000},
                             {1.0000000});
        return atom;
    }
    if (upper == "AL") {
        atom.charge = 13;
        atom.shells.reserve(42);
        add_cartesian_shells(atom, 0, 1, {37792.5507720, 5668.0682165, 1289.8582841, 364.86596028, 118.57631515, 42.024867605, 15.499501629},
                             {0.00057047888709, 0.0044093016538, 0.022630967411, 0.088025644295, 0.25223701612, 0.45960547169, 0.33277886014});
        add_cartesian_shells(atom, 0, 2, {75.208026598, 23.031408972, 3.6348797649},
                             {0.019250560190, 0.087906743952, -0.34246704535});
        add_cartesian_shells(atom, 0, 3, {1.6065049957, 0.76103394581},
                             {1.5106266058, 0.58071016470});
        add_cartesian_shells(atom, 0, 4, {0.16556708849},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.060041577113},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {452.52303192, 107.08195049, 34.131021255, 12.587037428, 4.9811919704},
                             {0.0023110812466, 0.018568641823, 0.087216237035, 0.26902101523, 0.52128324272});
        add_cartesian_shells(atom, 1, 2, {2.0070350900},
                             {0.60271687494});
        add_cartesian_shells(atom, 1, 3, {0.80083714374},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 4, {0.20178927472},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 5, {0.057895550392},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {1.57000000, 0.333000000},
                             {0.20000000, 1.00000000});
        add_cartesian_shells(atom, 2, 2, {0.111000000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {0.244000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "SI") {
        atom.charge = 14;
        atom.shells.reserve(42);
        add_cartesian_shells(atom, 0, 1, {44773.3580780, 6717.1992104, 1528.8960325, 432.54746585, 140.61505226, 49.857636724, 18.434974885},
                             {0.00055914765868, 0.0043206040189, 0.022187096460, 0.086489249116, 0.24939889716, 0.46017197366, 0.34250236575});
        add_cartesian_shells(atom, 0, 2, {86.533886111, 26.624606846, 4.4953057159},
                             {0.021300063007, 0.094676139318, -0.32616264859});
        add_cartesian_shells(atom, 0, 3, {2.1035045710, 1.0106094922},
                             {1.3980803850, 0.63865786699});
        add_cartesian_shells(atom, 0, 4, {0.23701751489},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.085703405362},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {394.47503628, 93.137683104, 29.519608742, 10.781663791, 4.1626574778},
                             {0.0026285693959, 0.020556257749, 0.092070262801, 0.25565889739, 0.42111707185});
        add_cartesian_shells(atom, 1, 2, {1.6247972989},
                             {0.34401746318});
        add_cartesian_shells(atom, 1, 3, {0.54306660493},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 4, {0.20582073956},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 5, {0.070053487306},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {2.30300000, 0.476000000},
                             {0.20000000, 1.00000000});
        add_cartesian_shells(atom, 2, 2, {0.160000000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {0.336000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "P") {
        atom.charge = 15;
        atom.shells.reserve(42);
        add_cartesian_shells(atom, 0, 1, {52426.9992330, 7863.2660552, 1789.5227333, 506.27300165, 164.60698546, 58.391918722, 21.643663201},
                             {0.00055207164100, 0.0042678595308, 0.021931529186, 0.085667168373, 0.24840686605, 0.46336753971, 0.35350558156});
        add_cartesian_shells(atom, 0, 2, {99.013837620, 30.550439817, 5.4537087661},
                             {0.021895679958, 0.095650470295, -0.29454270186});
        add_cartesian_shells(atom, 0, 3, {2.6503362563, 1.2726688867},
                             {1.3294381200, 0.66109396473});
        add_cartesian_shells(atom, 0, 4, {0.31645005203},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.11417466938},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {472.27219248, 111.58882756, 35.445936418, 12.990776875, 5.0486221658},
                             {0.0025710623052, 0.020250297999, 0.091580716787, 0.25749454014, 0.42862899758});
        add_cartesian_shells(atom, 1, 2, {1.9934049566},
                             {0.34359817849});
        add_cartesian_shells(atom, 1, 3, {0.66527284430},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 4, {0.25516832128},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 5, {0.090357762251},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {3.12000000, 0.648000000},
                             {0.20000000, 1.00000000});
        add_cartesian_shells(atom, 2, 2, {0.218000000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {0.452000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "S") {
        atom.charge = 16;
        atom.shells.reserve(42);
        add_cartesian_shells(atom, 0, 1, {60700.9281040, 9102.6106854, 2071.4166009, 586.02476821, 190.55395021, 67.630384260, 25.127306905},
                             {0.00054695944225, 0.0042297224557, 0.021747824159, 0.085100053589, 0.24799128459, 0.46703640406, 0.36434587550});
        add_cartesian_shells(atom, 0, 2, {112.57463010, 34.795554217, 6.5115556215},
                             {0.021670040240, 0.093602301760, -0.26068001422});
        add_cartesian_shells(atom, 0, 3, {3.2399032261, 1.5477160881},
                             {1.2842089435, 0.66036416584});
        add_cartesian_shells(atom, 0, 4, {0.40541030112},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.14550651059},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {564.36716027, 133.42624379, 42.468271189, 15.616527580, 6.1093988469},
                             {0.0024796796317, 0.019677930250, 0.089980008258, 0.25705880575, 0.43515167292});
        add_cartesian_shells(atom, 1, 2, {2.4404160198},
                             {0.34883240595});
        add_cartesian_shells(atom, 1, 3, {0.83882201296},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 4, {0.31288746900},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 5, {0.10770109004},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {3.75600000, 0.812000000},
                             {0.20000000, 1.00000000});
        add_cartesian_shells(atom, 2, 2, {0.273000000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {0.557000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "CL") {
        atom.charge = 17;
        atom.shells.reserve(42);
        add_cartesian_shells(atom, 0, 1, {69507.9909450, 10426.1568800, 2373.2334061, 671.56420071, 218.41999790, 77.572249714, 28.888815277},
                             {0.00054314897497, 0.0041990463961, 0.021592141679, 0.084598850094, 0.24757249724, 0.47016930228, 0.37436370716});
        add_cartesian_shells(atom, 0, 2, {127.10527185, 39.339582961, 7.6740679989},
                             {0.025182166603, 0.10786112456, -0.27408821574});
        add_cartesian_shells(atom, 0, 3, {3.8745627630, 1.8385832573},
                             {1.3213875014, 0.68636955368});
        add_cartesian_shells(atom, 0, 4, {0.50229057542},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.17962723420},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {666.50423284, 157.64241690, 50.262520978, 18.536078105, 7.2940532777},
                             {0.0023632663836, 0.018879300374, 0.087206341273, 0.25285612970, 0.43507154820});
        add_cartesian_shells(atom, 1, 2, {2.9433248995},
                             {0.35026513165});
        add_cartesian_shells(atom, 1, 3, {1.0404970818},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 4, {0.38456415080},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 5, {0.13069642732},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {4.61000000, 1.01100000},
                             {0.20000000, 1.00000000});
        add_cartesian_shells(atom, 2, 2, {0.339000000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {0.706000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "AR") {
        atom.charge = 18;
        atom.shells.reserve(42);
        add_cartesian_shells(atom, 0, 1, {79111.4229980, 11864.7460090, 2700.1642973, 763.95943485, 248.45150561, 88.283581000, 32.948607069},
                             {0.00051029325002, 0.0039463036981, 0.020307073910, 0.079691825214, 0.23420623836, 0.44833849481, 0.36408167400});
        add_cartesian_shells(atom, 0, 2, {142.55358000, 44.163688009, 8.9524995000},
                             {0.026387407001, 0.11226433999, -0.26178922001});
        add_cartesian_shells(atom, 0, 3, {4.5546920941, 2.1444079001},
                             {1.3002484998, 0.67197237009});
        add_cartesian_shells(atom, 0, 4, {0.60708777004},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 5, {0.21651431999},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {776.77541998, 183.80107018, 58.694003175, 21.701591695, 8.5821489635},
                             {0.0022028005003, 0.017694180008, 0.082431293717, 0.24207278863, 0.42263558251});
        add_cartesian_shells(atom, 1, 2, {3.4922679161},
                             {0.34151806086});
        add_cartesian_shells(atom, 1, 3, {1.2637426998},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 4, {0.46607870005},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 5, {0.15766003000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {5.55100000, 1.23500000},
                             {0.20000000, 1.00000000});
        add_cartesian_shells(atom, 2, 2, {0.412000000},
                             {1.0000000});
        add_cartesian_shells(atom, 3, 1, {0.890000000},
                             {1.0000000});
        return atom;
    }
    throw std::runtime_error("get_def2_tzvp_params: Unsupported element \"" + symbol + "\".");
}

} // namespace tlno::core::basis
