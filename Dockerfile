FROM ubuntu:22.10

RUN apt-get -q update
# git required for CMake FetchContent_MakeAvailable
RUN apt-get -q install -y clang-15 gcc-12 cmake git

CMD ./scripts/docker_entrypoint.sh
WORKDIR /autovec-db
