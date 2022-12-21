cmake_minimum_required(VERSION 3.18)
project(autovec_db)

set(CMAKE_CXX_STANDARD 20)

##### BUILD TYPES #####
set(ALLOWED_BUILD_TYPES Asan Tsan Debug Release RelWithDebInfo MinSizeRel)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${ALLOWED_BUILD_TYPES}")

if(CMAKE_BUILD_TYPE AND NOT CMAKE_BUILD_TYPE IN_LIST ALLOWED_BUILD_TYPES)
message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}. Allowed types: ${ALLOWED_BUILD_TYPES}")
endif()

set(CMAKE_CXX_FLAGS_ASAN "-Og -g -fno-omit-frame-pointer -fsanitize=address -fsanitize=undefined" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -fsanitize=address -fsanitize=undefined" CACHE STRING "" FORCE)

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -D_GLIBCXX_ASSERTIONS" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -DNDEBUG" CACHE STRING "" FORCE)

if (NOT CMAKE_BUILD_TYPE MATCHES Release)
  message(WARNING "Building benchmarks in ${CMAKE_BUILD_TYPE}. This may impact performance!")
endif()

set(OVERRIDE_ARCHITECTURE_FLAGS "" CACHE STRING "If not empty, the value is passed to the compiler instead of march/mcpu=native.")

if(OVERRIDE_ARCHITECTURE_FLAGS STREQUAL "")
  EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE CPU_ARCHITECTURE)
  if (${CPU_ARCHITECTURE} STREQUAL "x86_64")
    add_compile_options("-march=native")
  elseif (${CPU_ARCHITECTURE} STREQUAL "aarch64")
    add_compile_options("-mcpu=native")
  else ()
    message(STATUS "Unknown platform ${CPU_ARCHITECTURE}. Not using any native flags.")
  endif ()
else()
  add_compile_options("${OVERRIDE_ARCHITECTURE_FLAGS}")
endif()


##### COMPILE COMMANDS #####
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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


##################### Our Benchmarks ####################
function(add_benchmark bm_name)
  add_executable(${bm_name} ${bm_name}.cpp)
  target_link_libraries(${bm_name} PRIVATE autovec benchmark)
endfunction()

add_benchmark(compressed_scan)
add_benchmark(hash_bucket)
add_benchmark(hashing)
