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

set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3 -D_GLIBCXX_ASSERTIONS -DENABLE_DEBUG_DO" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g3 -gdwarf-4 -DNDEBUG" CACHE STRING "" FORCE)

set(OVERRIDE_ARCHITECTURE_FLAGS "" CACHE STRING "If not empty, the value is passed to the compiler instead of march/mcpu=native.")

if(OVERRIDE_ARCHITECTURE_FLAGS STREQUAL "")
  EXECUTE_PROCESS(COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE CPU_ARCHITECTURE)
  if (${CPU_ARCHITECTURE} STREQUAL "x86_64")
    add_compile_options("-march=native")
  elseif (${CPU_ARCHITECTURE} STREQUAL "aarch64" OR ${CPU_ARCHITECTURE} STREQUAL "arm64")
    add_compile_options("-mcpu=native")
  else ()
    message(WARNING "Unknown platform ${CPU_ARCHITECTURE}. Not using any native flags.")
  endif ()
  if(EXISTS "/sys/devices/cpu/caps/pmu_name")
    FILE(READ "/sys/devices/cpu/caps/pmu_name" CPU_MICROARCHITECTURE)
    string(REGEX REPLACE "\n$" "" CPU_MICROARCHITECTURE "${CPU_MICROARCHITECTURE}")
    if (${CPU_MICROARCHITECTURE} STREQUAL "icelake")
      # LLVM uses only 256bit vectors due to downclocking policies. With icelake, this has been fixed, but LLVM doesn't
      # yet know about this -- see https://travisdowns.github.io/blog/2020/08/19/icl-avx512-freq.html
      # Clang's min_vector_width attribute doesn't work as documented (https://github.com/llvm/llvm-project/issues/60946)
      # so we need to set compile flags for the whole compilation unit.
      set(USE_512_VECTORS_FLAGS "-mprefer-vector-width=512")
    endif()
  endif()
else()
  add_compile_options("${OVERRIDE_ARCHITECTURE_FLAGS}")
endif()

# We need this to to reinterpret_casts of GCC vector types.
add_compile_options("-flax-vector-conversions")

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
  add_compile_options("-fconstexpr-ops-limit=90000000")
else()
  add_compile_options("-fconstexpr-steps=30000000")
endif()

##################### Google Benchmark ####################
include(FetchContent)

FetchContent_Declare(
    google_benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.7.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing google benchmark's tests" FORCE)
FetchContent_MakeAvailable(google_benchmark)
# Mark google benchmark headers as system headers to avoid applying our stricter compile options to them.
get_property(_google_benchmark_include_dirs TARGET benchmark PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(benchmark SYSTEM INTERFACE ${_google_benchmark_include_dirs})
target_compile_options(benchmark PRIVATE "-Wno-error")


##### COMPILER WARNINGS #####
list(APPEND WARNING_OPTIONS "-Wall" "-Wextra" "-Werror" "-Wpedantic" "-Wextra-semi" "-Wnon-virtual-dtor" "-Wunused" "-Wzero-as-null-pointer-constant")
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  list(APPEND WARNING_OPTIONS "-Wno-maybe-uninitialized")
endif()

# disabled for clang because it triggers too often: https://github.com/llvm/llvm-project/issues/55309
# disabled for GCC because it triggers for 512bit GCC-vector helper functions.
# We build all files with the same compiler, so this is fine for us.
list(APPEND WARNING_OPTIONS "-Wno-psabi")

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Additional checks not supported by GCC -- some are supported on GCC, but not ignorable / not filterable
  list(APPEND WARNING_OPTIONS "-Wdocumentation" "-Wconditional-uninitialized" "-Wundef" "-Wunused-exception-parameter" "-Wunused-member-function")
  list(APPEND WARNING_OPTIONS "-Wno-gnu-zero-variadic-macro-arguments") # legal in c++20
  list(APPEND WARNING_OPTIONS "-Wshadow-all" "-Wno-shadow-field-in-constructor")
  list(APPEND WARNING_OPTIONS "-Wno-gnu-line-marker") # We don't care about this, but it pops up if we use "--save-temps" with clang.
endif()

##### COMPILE COMMANDS #####
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(AUTOVEC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/benchmarks)

##### INTERFACE TARGET FOR INCLUDE PATHS #####
add_library(autovec_includes INTERFACE)
target_include_directories(autovec_includes INTERFACE ${AUTOVEC_SOURCE_DIR})

##################### Our Benchmarks ####################
function(add_benchmark bm_name)
  add_executable(${bm_name} ${AUTOVEC_SOURCE_DIR}/${bm_name}.cpp)
  target_link_libraries(${bm_name} PRIVATE autovec_includes benchmark)
  target_compile_options(${bm_name} PRIVATE "${WARNING_OPTIONS}")

  # Keep the temporary files so we can inspect the generated assembly.
  target_compile_options(${bm_name} PRIVATE "-save-temps=obj" "-fverbose-asm")
endfunction()

add_benchmark(compare_to_bitmask)
add_benchmark(compressed_scan)
add_benchmark(dictionary_scan)
add_benchmark(hash_bucket)
add_benchmark(hashing)

target_compile_options(hashing PRIVATE ${USE_512_VECTORS_FLAGS})
