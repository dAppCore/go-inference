# SPDX-Licence-Identifier: EUPL-1.2
#
# Builds the GENERATED lem C SDK (task sdk -> build/sdk/c, then
# ./postgen-fix.sh — see README.md#friction) as a subdirectory, then links
# this example against it. The library's CMake target name is
# "lethean_lem_api" — the "c" generator derives it from the OpenAPI spec's
# info.title, NOT from sdk-config/c.yaml's projectName (see Friction).
cmake_minimum_required(VERSION 3.16)
project(lem_sdk_c_example C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_subdirectory(
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/sdk/c
    ${CMAKE_CURRENT_BINARY_DIR}/lem-sdk-c
)

add_executable(lem_c_example main.c)
target_link_libraries(lem_c_example PRIVATE lethean_lem_api)
