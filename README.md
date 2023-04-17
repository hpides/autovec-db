Microbenchmarks for our paper "Writing Less Platform-Specific SIMD Code in Databases"

This file focuses on the microbenchmarks. For reproducing the Velox measurements, we have [a separate README file](eval/results/velox/README.md).

Our measurement results can be found in [`eval/results/`](eval/results/)

## Running the benchmarks
The measurements were done using a docker image based on ubuntu 22.10 with clang 15 and gcc 12.

To repeat the measurements, run the following commands:
```bash
git clone https://github.com/hpides/autovec-db.git

# using enroot:
enroot import docker://hpides/autovec-db
enroot create hpides+autovec-db.sqsh
enroot start -m ./autovec-db/:/autovec-db --rw hpides+autovec-db

# alternatively, using docker:
docker pull hpides/autovec-db
# note: we're using --privileged as we observed up to 5x slower measurements without it (likely a seccomp problem)
docker run -it --privileged -v "$(pwd):/autovec-db" hpides/autovec-db
```

The commands that run inside the container can be found at [`scripts/docker_entrypoint.sh`](scripts/docker_entrypoint.sh).

The benchmarks log their results to `.csv` files that can be compared using our diff script at `eval/scripts/diff.py`.

The docker image was built using:
```bash
# x86
DOCKER_BUILDKIT=1 docker build -t hpides/autovec-db:x86 --target=x86 .
# AArch64
DOCKER_BUILDKIT=1 docker build -t hpides/autovec-db:aarch64 --target=aarch64 .
```

*Note:* To build the AArch64 image, you need QEMU on an x86 system.

To upload the images and automatically detect the architecture on download, run:
```bash
# Create
docker manifest create hpides/autovec-db:latest \
          --amend hpides/autovec-db:x86 \
          --amend hpides/autovec-db:aarch64

# Upload
docker manifest push hpides/autovec-db:latest
```

## Plotting the results

This is a short guide on how to work with the results from the paper and generate the plots.

### Setup

We use a basic Python + Pandas + Matplot setup. To create it, run the following commands in the `eval` directory:

```shell
$ python3 -m virtualenv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```

### Plots

Our scripts always generate a PNG and an SVG for each plot.
By default, it will generate a version optimized for PNG.
If you want to create the SVG plots used in the paper, set `AUTOVEC_DB_PLOTS=ON` in your environment variables.

To generate a plot for a benchmark, run the following command:
```shell
$ python3 scripts/<script_name>.py results/ /path/to/plot/dir
```

The script will output a command that you can use to view the file.
Generally, the plot is stored at `/path/to/plot/dir/<script_name>.[png|svg]`.

You can also create all plots with:
```shell
$ ./all_plots.sh results /path/to/plot/dir
```
