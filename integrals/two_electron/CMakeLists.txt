add_library(tlno_twoe_engine twoe_integrals.cpp fock/jk_direct.cpp)

add_library(tlno::twoe_engine ALIAS tlno_twoe_engine)

target_link_libraries(tlno_twoe_engine PUBLIC tlno_build_options)