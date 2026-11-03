add_library(onee_molecule_io STATIC src/molecule_io.cpp)
target_link_libraries(onee_molecule_io PUBLIC tlno_build_options)

add_library(tlno_onee_capi SHARED src/onee_capi.cpp src/basis_export.cpp)
set_target_properties(tlno_onee_capi PROPERTIES OUTPUT_NAME tlno_onee)
target_link_libraries(tlno_onee_capi PRIVATE tlno_onee_engine tlno_basis onee_molecule_io)
configure_file(build_info.txt.in "${CMAKE_BINARY_DIR}/onee_build_info.txt" @ONLY)

foreach(tool onee_timing onee_class_profile plan_contract)
    add_executable(${tool} src/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE tlno_onee_engine tlno_basis onee_molecule_io)
endforeach()
foreach(tool bench_shell_classes transfer_contract)
    add_executable(${tool} src/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE tlno_onee_engine)
endforeach()
foreach(tool boys_probe boys_packet_probe boys_mask_contract)
    add_executable(${tool} src/${tool}.cpp)
    target_link_libraries(${tool} PRIVATE tlno_math)
endforeach()

if(NOT BUILD_TESTING)
    return()
endif()

set(onee_scripts "${CMAKE_CURRENT_SOURCE_DIR}/scripts")
set(onee_results "${CMAKE_CURRENT_BINARY_DIR}/validation")
set(onee_tests)

add_test(NAME onee_shell_contract COMMAND transfer_contract)
add_test(NAME onee_boys_masks COMMAND boys_mask_contract)
list(APPEND onee_tests onee_shell_contract onee_boys_masks)
foreach(basis sto3g 431g 631g 631gd 631gdp def2svp def2tzvp)
    add_test(NAME onee_plan_${basis} COMMAND plan_contract
        "${PROJECT_SOURCE_DIR}/test_molecules/h2o.inp" "${basis}")
    list(APPEND onee_tests onee_plan_${basis})
endforeach()
foreach(fixture separated water10)
    add_test(NAME onee_plan_${fixture} COMMAND plan_contract
        "${PROJECT_SOURCE_DIR}/test_molecules/${fixture}.inp" def2tzvp)
    list(APPEND onee_tests onee_plan_${fixture})
endforeach()

if(TLNO_ENABLE_PYTHON_VALIDATION)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy"
        RESULT_VARIABLE onee_numpy_available OUTPUT_QUIET ERROR_QUIET)
    if(onee_numpy_available EQUAL 0)
        add_test(NAME onee_validation_guards
            COMMAND "${Python3_EXECUTABLE}" -B "${onee_scripts}/check_validation_guards.py")
        list(APPEND onee_tests onee_validation_guards)
    endif()
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy, mpmath"
        RESULT_VARIABLE onee_high_precision_available OUTPUT_QUIET ERROR_QUIET)
    if(onee_high_precision_available EQUAL 0)
        add_test(NAME onee_boys_high_precision
            COMMAND "${Python3_EXECUTABLE}" -B "${onee_scripts}/validate_boys_packet.py"
                --probe $<TARGET_FILE:boys_packet_probe>
                --out "${onee_results}/boys_packets.json")
        list(APPEND onee_tests onee_boys_high_precision)
    else()
        message(STATUS "NumPy/mpmath unavailable: high-precision Boys validation is not registered")
    endif()
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import numpy, pyscf"
        RESULT_VARIABLE onee_pyscf_available OUTPUT_QUIET ERROR_QUIET)
    if(onee_pyscf_available EQUAL 0)
        add_test(NAME onee_pyscf_shells
            COMMAND "${Python3_EXECUTABLE}" -B "${onee_scripts}/validate_transfer_fixtures.py"
                --contract $<TARGET_FILE:transfer_contract>
                --out "${onee_results}/pyscf_shells.json")
        list(APPEND onee_tests onee_pyscf_shells)
        foreach(threads 1 2)
            add_test(NAME onee_matrix_api_t${threads}
                COMMAND "${Python3_EXECUTABLE}" -B "${onee_scripts}/validate_matrix.py"
                    --lib $<TARGET_FILE:tlno_onee_capi> --threads ${threads}
                    --out "${onee_results}/matrix_api_t${threads}.json")
            list(APPEND onee_tests onee_matrix_api_t${threads})
        endforeach()
        foreach(scope full cached)
            foreach(representation cartesian spherical)
                set(gate_test onee_benchmark_gate_${scope}_${representation})
                add_test(NAME ${gate_test}
                    COMMAND "${Python3_EXECUTABLE}" -B "${onee_scripts}/benchmark_onee.py"
                        --validate-only --lib $<TARGET_FILE:tlno_onee_capi>
                        --molecules h2o,h2s
                        --basis sto3g,431g,631g,631gd,631gdp,def2svp,def2tzvp
                        --screening both --scope ${scope} --representation ${representation}
                        --out "${onee_results}/benchmark_${scope}_${representation}")
                list(APPEND onee_tests ${gate_test})
            endforeach()
        endforeach()
        execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import scipy"
            RESULT_VARIABLE onee_scipy_available OUTPUT_QUIET ERROR_QUIET)
        if(onee_scipy_available EQUAL 0)
            add_test(NAME onee_pyscf_molecules
                COMMAND "${Python3_EXECUTABLE}" -B "${onee_scripts}/validate.py"
                    --lib $<TARGET_FILE:tlno_onee_capi>
                    --boys-probe $<TARGET_FILE:boys_probe>
                    --molecules h2o,heh,oh,h2s --basis sto3g,431g,631g,631gd,631gdp,def2svp,def2tzvp
                    --out-dir "${onee_results}/molecules")
            list(APPEND onee_tests onee_pyscf_molecules)
        else()
            message(STATUS "SciPy unavailable: scalar Boys/molecule validation is not registered")
        endif()
    else()
        message(STATUS "NumPy/PySCF unavailable: independent molecule/shell/API validation is not registered")
    endif()
endif()

set_tests_properties(${onee_tests} PROPERTIES
    LABELS onee TIMEOUT 240
    ENVIRONMENT "OMP_NUM_THREADS=1;OPENBLAS_NUM_THREADS=1;MKL_NUM_THREADS=1;OMP_DYNAMIC=FALSE;PYTHONDONTWRITEBYTECODE=1")
