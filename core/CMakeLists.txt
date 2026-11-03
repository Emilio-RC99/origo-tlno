add_library(tlno_molecule molecule/molecule_reader.cpp)
add_library(tlno::molecule ALIAS tlno_molecule)
target_link_libraries(tlno_molecule PUBLIC tlno_build_options)

add_library(tlno_shell
    shell/shell_math.cpp
    shell/shell_contract.cpp
    shell/cart2sph.cpp)
add_library(tlno::shell ALIAS tlno_shell)
target_link_libraries(tlno_shell PUBLIC tlno_build_options)

add_library(tlno_math math/general_math.cpp)
add_library(tlno::math ALIAS tlno_math)
target_link_libraries(tlno_math PUBLIC tlno_build_options)

add_library(tlno_basis
    basis/basis_general.cpp
    basis/get_sto3g_params.cpp
    basis/get_4_31g_params.cpp
    basis/get_6_31g_params.cpp
    basis/get_6_31g_d_params.cpp
    basis/get_6_31g_dp_params.cpp
    basis/get_def2_svp_params.cpp
    basis/get_def2_tzvp_params.cpp)
add_library(tlno::basis ALIAS tlno_basis)
target_link_libraries(tlno_basis PUBLIC tlno_shell)
