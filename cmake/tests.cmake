#
# Define a target named `TEST_NAME` for a single extension.
# Optional arguments:
#
# LIBRARIES <libraries>      - Extra link libraries.
# INCLUDE_DIRECTORIES <dirs> - Extra include directories.
# SOURCES <sources>           - Extra source files.
#
function(add_test_executable TEST_NAME)
    cmake_parse_arguments(TEST "" "LIBRARIES;INCLUDE_DIRECTORIES;SOURCES" "" ${ARGN})
    add_executable(${TEST_NAME} ${TEST_NAME}.cpp ${TEST_SOURCES})
    target_include_directories(${TEST_NAME} PRIVATE
        ${TEST_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(${TEST_NAME}
        ${TEST_LIBRARIES}
    )
    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
    )
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()
