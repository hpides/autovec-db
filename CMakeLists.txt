cmake_minimum_required(VERSION 3.18)
project(autovec_db)

set(CMAKE_CXX_STANDARD 20)

add_library(autovec INTERFACE)
target_include_directories(autovec INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

include(FetchContent)

##################### Google Benchmark ####################
FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.6.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing google benchmark's tests" FORCE)
FetchContent_MakeAvailable(google_benchmark)

##################### Compiler Options ####################
if (CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "BUILDING BENCHMARKS IN RELEASE")
  target_compile_options(autovec INTERFACE -O3)
else()
  message(WARNING "Building benchmarks in ${CMAKE_BUILD_TYPE}. This may impact performance!")
endif()

##################### Our Benchmarks ####################
function(add_benchmark bm_name)
  add_executable(${bm_name} ${bm_name}.cpp)
  target_link_libraries(${bm_name} PRIVATE autovec benchmark)
endfunction()


add_benchmark(hash_bucket)
add_benchmark(hashing)
