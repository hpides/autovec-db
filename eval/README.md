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
