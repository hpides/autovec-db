Results in here run on DES01.

build in hpides/base_velox:v1

- velox_compiler based on commit 9434385a3020dd83bb17a63cdab122a2a052c806
- velox_xsimd based on commit 20b2273551b180b2c648c9077f27f5c558d4f3f6


in root CMakeLists.txt
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D HAS_UNCAUGHT_EXCEPTIONS -Wno-nullability-completeness -Wno-unqualified-std-cast-call ")

#### Changes to Flags

in `velox/CMakeLists.txt` add
```
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} [-march=X] [-mtune=Y]")
```
before `# VELOX BASE`


#### Changes to XSIMD

in `third_party/xsimd/include/xsimd/config/xsimd_config.hpp` set all AVX512 to 0
