Microbenchmarks for our paper "Writing Less Platform-Specific SIMD Code in Databases"

This file focuses on the microbenchmarks. For reproducing the Velox measurements, we have [a separate README file](eval/results/velox/README.md).

# Running the benchmarks
Our measurements were done using a docker image based on ubuntu 22.10 with clang 15 and gcc 12.

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

The commands that run inside the container can be found at `scripts/docker_entrypoint.sh`

The docker image was built using
```bash
docker build -t hpides/autovec-db .
docker login
docker push hpides/autovec-db
```

# Plots and Results

This is a short guide on how to work with the results from the paper and generate the plots.

### Setup

We use a basic Python + Pandas + Matplot setup. To create it, run the following commands in the [`eval/`](./) directory:

```shell
$ python3 -m virtualenv venv
$ source venv/bin/activate
$ pip3 install -r requirements.txt
```

### Plots

To generate a plot for a benchmark, run the following command:
```shell
$ python3 scripts/<script_name>.py results/ /path/to/plot/dir
```

The script will output a command that you can use to view the file.
Generally, the plot is stored at `/path/to/plot/dir/<script_name>.[png|svg]`.
