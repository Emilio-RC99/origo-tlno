#include "get_def2_svp_params.hpp"
#include "cartesian_shells.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace tlno::core::basis {
namespace {

std::string uppercase (std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

using detail::add_cartesian_shells;

} // namespace

// BSE def2-SVP version 1; tabulated parameters for H-Ar.
// https://www.basissetexchange.org/api/basis/def2-svp/format/orca/?elements=1-18&version=1
// Preserve the engine's Cartesian angular components and shell labels.
RawAtomBasis get_def2_svp_params(const std::string& symbol) {
    RawAtomBasis atom;
    const std::string upper = uppercase(symbol);

    if (upper == "H") {
        atom.charge = 1;
        atom.shells.reserve(5);
        add_cartesian_shells(atom, 0, 1, {13.0107010, 1.9622572, 0.44453796},
                             {0.019682158, 0.13796524, 0.47831935});
        add_cartesian_shells(atom, 0, 2, {0.12194962},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {0.8000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "HE") {
        atom.charge = 2;
        atom.shells.reserve(5);
        add_cartesian_shells(atom, 0, 1, {38.354936737, 5.7689081479, 1.2399407035},
                             {0.023814288905, 0.15490906777, 0.46998096633});
        add_cartesian_shells(atom, 0, 2, {0.29757815953},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {1.0000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "LI") {
        atom.charge = 3;
        atom.shells.reserve(9);
        add_cartesian_shells(atom, 0, 1, {266.27785516, 40.069783447, 9.0559944389, 2.4503009051, 0.72209571855},
                             {0.0064920150325, 0.047747863215, 0.20268796111, 0.48606574817, 0.43626977955});
        add_cartesian_shells(atom, 0, 2, {0.052810884721},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.020960948798},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {1.4500000, 0.3000000},
                             {0.2586000, 1.0000000});
        add_cartesian_shells(atom, 1, 2, {0.0820000},
                             {1.0000000});
        return atom;
    }
    if (upper == "BE") {
        atom.charge = 4;
        atom.shells.reserve(9);
        add_cartesian_shells(atom, 0, 1, {515.18616125, 77.511037595, 17.552481693, 4.8028940596, 1.4516214316},
                             {0.0055615307983, 0.041190068062, 0.17913378108, 0.44736716455, 0.42009581920});
        add_cartesian_shells(atom, 0, 2, {0.13281633620},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.045837372213},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {3.6316917145, 0.71695694366, 0.19541932860},
                             {-0.029033998305, -0.16877854032, -0.51403419628});
        add_cartesian_shells(atom, 1, 2, {0.060515465890},
                             {1.0000000});
        return atom;
    }
    if (upper == "B") {
        atom.charge = 5;
        atom.shells.reserve(15);
        add_cartesian_shells(atom, 0, 1, {839.31830086, 126.26464843, 28.620600763, 7.8793722710, 2.4088857172},
                             {-0.0055929201074, -0.041565520771, -0.18299816983, -0.46540391866, -0.44173884791});
        add_cartesian_shells(atom, 0, 2, {0.25105109036},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.083648866069},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {6.0332223619, 1.2499157866, 0.33871676350},
                             {-0.035603672456, -0.19895775769, -0.50850202618});
        add_cartesian_shells(atom, 1, 2, {0.096415632351},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.5000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "C") {
        atom.charge = 6;
        atom.shells.reserve(15);
        add_cartesian_shells(atom, 0, 1, {1238.4016938, 186.29004992, 42.251176346, 11.676557932, 3.5930506482},
                             {0.0054568832082, 0.040638409211, 0.18025593888, 0.46315121755, 0.44087173314});
        add_cartesian_shells(atom, 0, 2, {0.40245147363},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.13090182668},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {9.4680970621, 2.0103545142, 0.54771004707},
                             {0.038387871728, 0.21117025112, 0.51328172114});
        add_cartesian_shells(atom, 1, 2, {0.15268613795},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.8000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "N") {
        atom.charge = 7;
        atom.shells.reserve(15);
        add_cartesian_shells(atom, 0, 1, {1712.8415853, 257.64812677, 58.458245853, 16.198367905, 5.0052600809},
                             {-0.0053934125305, -0.040221581118, -0.17931144990, -0.46376317823, -0.44171422662});
        add_cartesian_shells(atom, 0, 2, {0.58731856571},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.18764592253},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {13.571470233, 2.9257372874, 0.79927750754},
                             {-0.040072398852, -0.21807045028, -0.51294466049});
        add_cartesian_shells(atom, 1, 2, {0.21954348034},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {1.0000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "O") {
        atom.charge = 8;
        atom.shells.reserve(15);
        add_cartesian_shells(atom, 0, 1, {2266.1767785, 340.87010191, 77.363135167, 21.479644940, 6.6589433124},
                             {-0.0053431809926, -0.039890039230, -0.17853911985, -0.46427684959, -0.44309745172});
        add_cartesian_shells(atom, 0, 2, {0.80975975668},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.25530772234},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {17.721504317, 3.8635505440, 1.0480920883},
                             {0.043394573193, 0.23094120765, 0.51375311064});
        add_cartesian_shells(atom, 1, 2, {0.27641544411},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {1.2000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "F") {
        atom.charge = 9;
        atom.shells.reserve(15);
        add_cartesian_shells(atom, 0, 1, {2894.8325990, 435.41939120, 98.843328866, 27.485198001, 8.5405498171},
                             {-0.0053408255515, -0.039904258866, -0.17912768038, -0.46758090825, -0.44653131020});
        add_cartesian_shells(atom, 0, 2, {1.0654578038},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.33247346748},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {22.696633924, 4.9872339257, 1.3491613954},
                             {-0.045212874436, -0.23754317067, -0.51287353587});
        add_cartesian_shells(atom, 1, 2, {0.34829881977},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {1.4000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "NE") {
        atom.charge = 10;
        atom.shells.reserve(15);
        add_cartesian_shells(atom, 0, 1, {3598.9736625, 541.32073112, 122.90450062, 34.216617022, 10.650584124},
                             {-0.0053259297003, -0.039817417969, -0.17914358188, -0.46893582977, -0.44782537577});
        add_cartesian_shells(atom, 0, 2, {1.3545953960},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 3, {0.41919362639},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {28.424053785, 6.2822510953, 1.6978715079},
                             {-0.046031944795, -0.23993183041, -0.50871724964});
        add_cartesian_shells(atom, 1, 2, {0.43300700172},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {1.8880000},
                             {1.0000000});
        return atom;
    }
    if (upper == "NA") {
        atom.charge = 11;
        atom.shells.reserve(16);
        add_cartesian_shells(atom, 0, 1, {4098.2003908, 616.49374031, 139.96644001, 39.073441051, 11.929847205},
                             {-0.0058535911879, -0.043647161872, -0.19431465884, -0.48685065731, -0.41881705137});
        add_cartesian_shells(atom, 0, 2, {20.659966030, 1.9838860978, 0.64836323942},
                             {0.085949689854, -0.56359144041, -0.51954009048});
        add_cartesian_shells(atom, 0, 3, {0.052443967404},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.028048160742},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {75.401862017, 17.274818978, 5.1842347425, 1.6601211973, 0.51232528958},
                             {0.0154353625324, 0.0997382931840, 0.312095939659, 0.492956748074, 0.324203983180});
        add_cartesian_shells(atom, 1, 2, {0.0520000},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.1310000},
                             {1.0000000});
        return atom;
    }
    if (upper == "MG") {
        atom.charge = 12;
        atom.shells.reserve(19);
        add_cartesian_shells(atom, 0, 1, {4953.8339196, 745.18044154, 169.21604972, 47.300672019, 14.461336973},
                             {-0.0057778967498, -0.043124761082, -0.19268216987, -0.48641439116, -0.42550894077});
        add_cartesian_shells(atom, 0, 2, {24.768174789, 2.4940945349, 0.87807584533},
                             {0.087956969984, -0.55165058128, -0.53443294833});
        add_cartesian_shells(atom, 0, 3, {0.087212782497},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.033599293780},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {98.053010494, 22.586932277, 6.8391509842, 2.2332843818, 0.71606599387},
                             {-0.014480564601, -0.095495750787, -0.30787672651, -0.49936292886, -0.31503476213});
        add_cartesian_shells(atom, 1, 2, {0.18914796195},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 3, {0.053768755187},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.1010000},
                             {1.0000000});
        return atom;
    }
    if (upper == "AL") {
        atom.charge = 13;
        atom.shells.reserve(19);
        add_cartesian_shells(atom, 0, 1, {5887.5727030, 885.61225996, 201.13604899, 56.284974674, 17.229551243},
                             {0.0013483347987, 0.010071576809, 0.045132454056, 0.11461268043, 0.10159608943});
        add_cartesian_shells(atom, 0, 2, {29.340249922, 3.0439630420, 1.1285539518},
                             {0.069347454208, -0.42528117679, -0.41449832210});
        add_cartesian_shells(atom, 0, 3, {0.14234175160},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.054400192313},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {145.11918809, 33.717894833, 10.369863083, 3.5135616036, 1.1980050273},
                             {0.0063963373134, 0.044189359965, 0.15581575993, 0.28635286951, 0.22921423248});
        add_cartesian_shells(atom, 1, 2, {0.26583005913},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 3, {0.071003361994},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.3000000},
                             {1.0000000});
        return atom;
    }
    if (upper == "SI") {
        atom.charge = 14;
        atom.shells.reserve(19);
        add_cartesian_shells(atom, 0, 1, {6903.7118686, 1038.4346419, 235.87581480, 66.069385169, 20.247945761},
                             {0.0013373962995, 0.0099966546241, 0.044910165101, 0.11463638540, 0.10280063858});
        add_cartesian_shells(atom, 0, 2, {34.353481730, 3.6370788192, 1.4002048599},
                             {0.070837285010, -0.43028836252, -0.41382774969});
        add_cartesian_shells(atom, 0, 3, {0.20484414805},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.077994095468},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {179.83907373, 41.907258846, 12.955294367, 4.4383267393, 1.5462247904},
                             {0.0061916656462, 0.043399431982, 0.15632019351, 0.29419996982, 0.23536823814});
        add_cartesian_shells(atom, 1, 2, {0.35607612302},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 3, {0.10008513762},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.3500000},
                             {1.0000000});
        return atom;
    }
    if (upper == "P") {
        atom.charge = 15;
        atom.shells.reserve(19);
        add_cartesian_shells(atom, 0, 1, {8002.4795106, 1203.6813590, 273.44227031, 76.655541517, 23.516927435},
                             {0.0057503489302, 0.043007628959, 0.19363986252, 0.49651693399, 0.44983262479});
        add_cartesian_shells(atom, 0, 2, {39.791683439, 4.2770343323, 1.6940256888},
                             {0.095188129789, -0.57649840368, -0.54239583865});
        add_cartesian_shells(atom, 0, 3, {0.27567674644},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.10495590099},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {219.50755823, 51.274155030, 15.921595892, 5.5069913481, 1.9537719426},
                             {0.0092100565257, 0.065409765745, 0.24033730279, 0.46318321788, 0.37392563382});
        add_cartesian_shells(atom, 1, 2, {0.47803397932},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 3, {0.13657952621},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.4500000},
                             {1.0000000});
        return atom;
    }
    if (upper == "S") {
        atom.charge = 16;
        atom.shells.reserve(19);
        add_cartesian_shells(atom, 0, 1, {9184.9303010, 1381.5105503, 313.87147580, 88.053870623, 27.039914905},
                             {-0.0022294387756, -0.016683029937, -0.075262436116, -0.19376827038, -0.17718020803});
        add_cartesian_shells(atom, 0, 2, {45.648731303, 4.9664522326, 2.0116242047},
                             {-0.10736062573, 0.65066293018, 0.59712155354});
        add_cartesian_shells(atom, 0, 3, {0.35661077013},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.13507221477},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {261.98233439, 61.306894736, 19.103729887, 6.6567720378, 2.3959635161},
                             {-0.0092729929822, -0.066547669241, -0.24828595903, -0.48703847402, -0.39337850312});
        add_cartesian_shells(atom, 1, 2, {0.61776161679},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 3, {0.16993376871},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.5500000},
                             {1.0000000});
        return atom;
    }
    if (upper == "CL") {
        atom.charge = 17;
        atom.shells.reserve(19);
        add_cartesian_shells(atom, 0, 1, {10449.8275660, 1571.7365221, 357.12065523, 100.25185935, 30.812727554},
                             {0.0019708362484, 0.014754727977, 0.066679112875, 0.17228924084, 0.15883786100});
        add_cartesian_shells(atom, 0, 2, {51.923789434, 5.7045760975, 2.3508376809},
                             {-0.10009298909, 0.60841752753, 0.54352153355});
        add_cartesian_shells(atom, 0, 3, {0.44605124672},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.16848856190},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {307.66790569, 72.102015515, 22.532680262, 7.8991765444, 2.8767268321},
                             {-0.0087801484118, -0.063563355471, -0.24016428276, -0.47798866557, -0.38515850005});
        add_cartesian_shells(atom, 1, 2, {0.77459363955},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 3, {0.21037699698},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.6500000},
                             {1.0000000});
        return atom;
    }
    if (upper == "AR") {
        atom.charge = 18;
        atom.shells.reserve(19);
        add_cartesian_shells(atom, 0, 1, {11797.1197650, 1774.3522753, 403.18875733, 113.24933999, 34.835298193},
                             {0.0020214479973, 0.015139852688, 0.068525400873, 0.17762929207, 0.16496495001});
        add_cartesian_shells(atom, 0, 2, {58.614775042, 6.4922045078, 2.7117014403},
                             {-0.10343394000, 0.63133365899, 0.54887572301});
        add_cartesian_shells(atom, 0, 3, {0.54412974538},
                             {1.0000000});
        add_cartesian_shells(atom, 0, 4, {0.20517411136},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 1, {356.28707256, 83.593132866, 26.186704029, 9.2257275925, 3.3922754954},
                             {-0.0087321783271, -0.063680317444, -0.24311906886, -0.48956069653, -0.39229190040});
        add_cartesian_shells(atom, 1, 2, {0.94740534010},
                             {1.0000000});
        add_cartesian_shells(atom, 1, 3, {0.25659135062},
                             {1.0000000});
        add_cartesian_shells(atom, 2, 1, {0.6960000},
                             {1.0000000});
        return atom;
    }
    throw std::runtime_error("get_def2_svp_params: Unsupported_element \"" + symbol + "\".");
}

} // namespace tlno::core::basis
