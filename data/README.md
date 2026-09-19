# Data

## Overview

The computational experiments use benchmark instances from three established SSCFLP test sets. To ensure a consistent input format across all datasets, the original benchmark instances are converted into a **unified JSON format** before being processed by the algorithms.

This folder contains one example instance, `i300_1.json`, illustrating the JSON structure used throughout the repository. The corresponding original benchmark file, prior to conversion into the unified JSON format, is provided in the `raw_data` folder.

Additional benchmark instances can be obtained from the original sources listed below and converted using `scripts/load_instance.py`.

---

## Data Sources

The computational experiments use instances from the following benchmark datasets.

1. **12 test instances** from the OR-Library (OR4).  
   [OR-Library CAPA-CAPC](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/capinfo.html) (last accessed: 21st March 2025).

2. **100 test instances** presented in:  
   Avella, P., Boccia, M., & Sforza, A. (2008). [Solving large scale capacitated facility location problems by genetic algorithms](https://doi.org/10.1016/j.ejor.2006.09.036).  
   *European Journal of Operational Research, 185(3), 1304–1325.*  
   Instances available at [OR@Brescia](https://or-brescia.unibs.it/instances/instances_sscflp) (last accessed: 21st March 2025).

3. **445 test instances** presented in:  
   Avella, P., & Boccia, M. (2009). [A cutting plane algorithm for the capacitated facility location problem](https://doi.org/10.1007/s10589-007-9117-5).  
   *Computational Optimization and Applications, 43(1), 39–65.*  
   and 
   Guastaroba, G., & Speranza, M. G. (2012). [Kernel search for the capacitated facility location problem](https://doi.org/10.1007/s10732-012-9212-8).  
   *Journal of Heuristics, 18(6), 877-917.* 
   - *TB-A* and *TB-B* instances available at [Unina](https://wpage.unina.it/sforza/test/) (last accessed: 21st March 2025).
   - *TB-C* instances available at [OR@Brescia](https://or-brescia.unibs.it/instances/instances_sscflp) (last accessed: 21st March 2025).

---

## Unified JSON Format

Each converted instance is stored as a JSON object containing two sections:

- `info` - basic instance information
- `params` - the numerical parameters defining the SSCFLP instance

The `params` object contains: 

- `I` - number of candidate facilities
- `J` - number of customers
- `D` - customer demands, stored as a list of length `J`
- `Q` - facility capacities, stored as a list of length `I`
- `F` - facility fixed opening costs, stored as a list of length `I`
- `c` - transportation costs, stored as an `I × J` matrix

The provided example,  `i300_1.json`, therefore has the following structure:

```json
{
  "info": {
    "name": "i300_1"
  },
  "params": {
    "I": 300,
    "J": 300,
    "D": [  
      ...
    ],       
    "Q": [  
      ...
    ],        
    "F": [  
      ...
    ],        
    "c": [             
      [...],
      [...],
      ...
    ]
  }
}
```

---

## Preparing Additional Instances

The repository does not include all original benchmark files. They can be downloaded from the sources listed above and converted to the unified JSON format using:

```
python scripts/load_instance.py <folder_path> <name> <test_set> [<capacity>]
```

The conversion procedure automatically reads the corresponding original benchmark format and generates the JSON representation required by the algorithms in this repository.

See `scripts/README.md` for details on the arguments, supported test sets, and instance-conversion procedure.