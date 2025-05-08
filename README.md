# Enhancing Kernel Search with Pattern Recognition: the Single-Source Capacitated Facility Location Problem

This repository provides the implementation of **Pattern-based Kernel Search (PaKS)**, a two-phase matheuristic for solving the **Single-Source Capacitated Facility Location Problem (SSCFLP)**. The algorithm integrates **pattern recognition techniques** into a traditional kernel search framework to improve solution quality and scalability.

The repository also includes a re-implementation of the Kernel Search presented in *Guastaroba, G., & Speranza, M. G. (2014). A heuristic for BILP problems: The Single Source Capacitated Facility Location Problem. European Journal of Operational Research, 238(2), 438-450.* for benchmarking and comparison purposes.

This code accompanies the manuscript:

> **Enhancing Kernel Search with Pattern Recognition: the Single-Source Capacitated Facility Location Problem**
> Hannah Bakker, Stefan Nickel, Gianfranco Guastaroba, and M. Grazia Speranza

---

## Repository Structure

```
PaKS-for-the-SSCFLP/
│
├── data/             # Benchmark instances (in unified JSON format)
├── results/          # Output files from algorithm runs
├── scripts/          # Executable scripts (run experiments, load instances)
├── src/              # Source code modules (kernel_search, models, utils)
├── requirements.txt  # Python dependencies 
├── LICENSE           # License file 
└── README.md         # This file
```

---

## Installation

To install the required dependencies, first create a virtual environment (optional but recommended), then install the packages:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

---

## Running an Instance

To execute a single SSCFLP instance using either the original Kernel Search (KS14) or the Pattern-based Kernel Search (PaKS), use the script `scripts/main.py`.

This script loads a `.json` instance file, applies the selected configuration, and writes the results to the `results/` folder.

### Usage

```bash
python scripts/main.py <path_to_instance> <config> <timelimit>
```

### Arguments

* `<path_to_instance>` — Path to the `.json` input file describing the instance (e.g., `data/i300_1.json`)
* `<config>` — Algorithm to use: `KS14` or `PaKS`
* `<timelimit>` — Time limit for the algorithm (in seconds)

### Example

```bash
python scripts/main.py data/i300_1.json PaKS 300
```

This will run the PaKS algorithm on instance `i300_1.json` for 300 seconds and store the result in:
`results/300s-PaKS-i300_1.json`

---

## Preparing Input Instances

All benchmark instances are converted to a unified JSON format. One example instance (`i300_1.json`) is provided in the `data/` folder.

To convert additional benchmark files (e.g., `.txt` or `.plc` formats), use:

```bash
python scripts/load_instance.py <folder_path> <name> <test_set> [<capacity>]
```

See `scripts/README.md` and `data/README.md` for details.

---

## Results and Output

Output files are saved to the `results/` folder and include:

* `.json` summaries of individual runs
* A summary Excel file (`PaKS_CompleteResults.xlsx`) consolidating main results reported in the manuscript

See `results/README.md` for more information.

---

## Contact

For questions or feedback, please contact:
**Hannah Bakker** — \[[hannah.bakker@kit.edu](mailto:hannah.bakker@kit.edu)]
