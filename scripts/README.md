# Scripts

## Overview

This folder contains executable Python scripts for:

- Running experiments with the algorithms implemented in this repository
- Converting benchmark instances into a unified `.json` format

---

## Files

### `main.py`

Runs a specified SSCFLP instance using either the **original Kernel Search (KS14)** or the **Pattern-based Kernel Search (PaKS)** for a given time limit.

#### Usage:
```bash
python main.py <path_to_instance> <config> <timelimit>
```

#### Arguments:
- `path_to_instance` — Path to the input .json instance file
- `config` — Algorithm variant to use: KS14 or PaKS
- `timelimit` — Time limit in seconds

#### Example:
```bash
python main.py ../data/i300_1.json PaKS 300
```
This will create a result file in the format `../results/300s-PaKS-i300_1.json`.

### `load_instance.py`

Converts benchmark instances from various source formats into the unified JSON format used throughout this repository.

Supported Input Formats:
- `.txt` files from the OR-Library (OR4)
- `.plc` files from Avella & Boccia and Guastaroba & Speranza

#### Usage:
```bash
python load_instance.py <folder_path> <name> <test_set> [<capacity>]
```

#### Arguments:
- <folder_path> — Path to the folder containing the original instance file
- <name> — Base name of the instance file (without extension)
- <test_set> — Either "OR4" or a tag such as "B1" / "TB" / "TB-C"
- [<capacity>] — Optional: overrides the facility capacity for OR4 instances

#### Example:
```bash
python load_instance.py ../raw_data capa OR4 5000
```
This will convert `capa.txt` into `data/capa.json`.

---

## Output
Converted instance files are stored in the `data/` folder.

Result files are stored in the `results/` folder.