# Scripts

## Overview

This folder contains the executable Python scripts used to **run PaKS** and to **prepare benchmark instances** for the computational experiments.

The folder contains two scripts:
- `main.py` - runs PaKS on a single SSCFLP instance
- `load_instance.py` - converts benchmark instances from their original formats into the unified JSON format used throughout the repository

---

## Running PaKS

The script `main.py` runs PaKS on a single SSCFLP instance using a specified algorithm configuration and time limit.

The script is executed as follows:

```bash
python scripts/main.py <path_to_instance> <config> <timelimit>
```

The arguments are:
- `<path_to_instance>`- path to the input instance in the unified JSON format
- `<config>` - name of the algorithm configuration defined in `src/algs/configs.py`
- `<timelimit>` - total time limit for the algorithm, in seconds

For example:
```bash
python scripts/main.py data/i300_1.json default 3600
```

This runs PaKS on instance `i300_1.json` using the `default` configuration and a time limit of 3600 seconds.

The resulting output is stored in the `results/` folder using the following naming convention:

`<timelimit>s-<config>-<instance>.json`

For the example above, the output file is:

`results/3600s-default-i300_1.json`

During execution, logging information is written both to the console and to the `logs/` folder.

---

## Preparing Benchmark Instances

The script `load_instance.py` converts benchmark instances from their original file formats into the unified JSON format used throughout this repository.

Supported input formats are:
- `.txt` files from the OR-Library (OR4)
- `.plc` files from Avella & Boccia and Guastaroba & Speranza

The script is executed as follows:

```bash
python scripts/load_instance.py <folder_path> <name> <test_set> [<capacity>]
```

The arguments are:
- `<folder_path>` - path to the folder containing the original instance file
- `<name>` - name of the instance without the file extension
- `<test_set>` - benchmark test set identifier (use "OR4" for the OR-Library instances.; any other value is interpreted as an instance in the common `.plc` format used by the remaining benchmark sets)
- `<capacity>`- optional facility capacity used for OR4 instances

For example:
```bash
python scripts/load_instance.py data/raw_data i300_1 TB-1
```

This converts `i300_1.plc` from text_set=TB-1 into the unified JSON format and stores the resulting instance as:

`data/i300_1.json`

See `data/README.md` for information on the benchmark datasets and the unified JSON structure.
