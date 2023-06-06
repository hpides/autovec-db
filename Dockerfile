# This is the basic docker image we use for our benchmarks.
# We have a x86 version and a aarch64 version.

# x86 image: Build with --target=x86
FROM amd64/ubuntu:22.10 as x86
RUN apt-get -q update && apt-get -q install -y clang-15 gcc-13 g++-13 cmake git
ENV AUTOVEC_DB_COMPILER=clang++-15
CMD ./scripts/docker_entrypoint.sh
WORKDIR /autovec-db

# AArch64 image: Build with --target=aarch64
FROM arm64v8/ubuntu:22.10 as aarch64
RUN apt-get -q update && apt-get -q install -y wget gnupg software-properties-common cmake git gcc-13 g++-13
RUN wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 17
ENV AUTOVEC_DB_COMPILER=clang++-17
CMD ./scripts/docker_entrypoint.sh
WORKDIR /autovec-db
