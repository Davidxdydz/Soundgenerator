cmake_minimum_required(VERSION 3.10.2)
project(functionGenerator VERSION 1.0 DESCRIPTION "A graph based function generator.")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE}
    CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo"
    FORCE)

set(CMAKE_CXX_FLAGS_DEBUG
    "-fno-omit-frame-pointer -g -Wall "
    CACHE STRING "Flags used by the C++ compiler during AddressSanitizer builds."
    FORCE)

set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -Wall "
    CACHE STRING "Flags used by the C++ compiler during Release builds."
    FORCE)

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "-O3 -Wall  -g"
    CACHE STRING "Flags used by the C++ compiler during RelWithDebInfo builds."
    FORCE)

set(cc_sources
    function_generator/src/sine_generator.cc
    function_generator/src/uniform_sampler.cc
    function_generator/src/saw_generator.cc
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "..")
find_package(pybind11 REQUIRED)
pybind11_add_module(function_generator SHARED "${cc_sources}" function_generator/src/bind.cc)
target_include_directories(function_generator PRIVATE function_generator/include)
target_compile_definitions(function_generator PUBLIC _USE_MATH_DEFINES)

