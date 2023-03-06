# Velox Eval

Results in here run on DES01 and nvram-06

### Docker/enroot
You can download a prepared Docker image `hpides/base_velox:v1` (from Dockerhub).

### Build Versions
- velox_compiler based on commit 9434385a3020dd83bb17a63cdab122a2a052c806
- velox_xsimd based on commit 20b2273551b180b2c648c9077f27f5c558d4f3f6


### General Changes
To suppress some a huge number of warnings, in the root `CMakeLists.txt` add:
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAS_UNCAUGHT_EXCEPTIONS -Wno-nullability-completeness -Wno-unqualified-std-cast-call ")
```

### Changes to Flags

In `velox/CMakeLists.txt` add the following line before(!) the line with the `# VELOX BASE` comment, depending on the flag combination you want to have.
```cmake
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} [-march=X] [-mtune=Y]")
```


### Changes to XSIMD for -march=native

In `third_party/xsimd/include/xsimd/config/xsimd_config.hpp` set all AVX512 to 0


### Other Issues

If you get an error for missing symbols for some atomics, add `atomic` to `target_link_libraries(velox_common_base ...)` in `velox/common/base/CMakeLists.txt` (I have no idea why this fails only sometimes.)
