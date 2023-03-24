# Velox Results

The results in here were measured on DES01 (cascadelake), nvram-06 (icelake), and a MacBook Pro (M1).


### Docker/enroot
You can download a prepared Docker image `hpides/base_velox:v1` (from Dockerhub).


### Code Variants
Our modified code can be found at https://github.com/He3lixxx/velox/, there is one branch per variant.
These are the commits that produced the results in this repository:

- velox_xsimd: 7da8af105065ff72e92a0560e335b38e813d29a7
- velox_compiler: 154fcf25b638fc592ba9324f4571883b0fea393f
- velox_compiler_padding: 11a79a111c865a5b16c3338d3f3df58273417eec
- velox_autovec: 81b72b00b78c7278e9735039a47f00f2a7bbe528
- velox_novec: 480a06ced0af76d758e01cd1725e8e605932f6a0


### Native-Flags

In `velox/CMakeLists.txt` add the following line at the `#NATIVE_FLAGS_HERE_MARKER` comment, depending on the flag combination you want to have.
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} [-march=X] [-mtune=Y]")
```

#### XSIMD and march=native
Velox doesn't build with march=native on AVX512-platforms, since some function specializations are not implemented.
AVX512-usage can be explicitly disabled for code using xsimd by overriding `XSIMD_WITH_AVX512F` to `0` in `third_party/xsimd/include/xsimd/config/xsimd_config.hpp`.


### Running the benchmarks
All benchmarking was done using `numactl -N 0` to pin execution to a single NUMA-node.
For all benchmarks, we used the flags `--num_drivers=4 --minloglevel=5 --bm_min-iters=10 --bm_regex="q\d$|q1\d|q20|q22" --dynamic_cputhreadpoolexecutor=0 --dynamic_iothreadpoolexecutor=0 --cache_gb=10`.
This enforces at least 10 iterations per tpch-query and removes some log-spam.
Disabling the dynamic executors and using `cache_gb` gave us more consistent results across benchmark runs.

We used the script `scripts/run_velox_benchmarks.sh` to run all benchmarks on all code-variant/native-flags combinations.
Due to the submodule-nature of xsimd, the xsimd-march-native run has to be performed by hand.

We repeated each benchmark run 10 times. You can copy the output to a text file and use the `eval/scripts/velox_out_to_csv.py` script to post-process the output.
The diff-script at `eval/scripts/diff.py` can be used to compare runs.


### Other Issues

* To suppress the warning spam during compilation, you can add this in the root `CMakeLists.txt`:
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAS_UNCAUGHT_EXCEPTIONS -Wno-nullability-completeness -Wno-unqualified-std-cast-call ")
```
* A linking error related to missing symbols regarding atomics can be fixed by adding `atomic` to `target_link_libraries(velox_common_base ...)` in `velox/common/base/CMakeLists.txt`
