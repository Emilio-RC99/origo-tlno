add_library(tlno_onee_engine
    onee_integrals.cpp
    onee_matrices.cpp)
add_library(tlno::onee ALIAS tlno_onee_engine)
target_link_libraries(tlno_onee_engine PUBLIC tlno_shell tlno_math)