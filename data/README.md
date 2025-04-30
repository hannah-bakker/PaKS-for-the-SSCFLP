# Data

## Overview

All benchmark instances used in our computational experiments were **converted into a unified JSON format** for consistency and ease of use.

As an example, this folder provides the file `i300_1.json`, which demonstrates the structure used across all instances.

In addition, the folder `scripts` contains the Python script `load_instance.py`, which can be used to load instances from the different benchmark datasets and convert them into the unified JSON format.

---

## Data Sources

We used instances from three benchmark datasets, which can be accessed through the links below:

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

## Example File

The unified JSON structure contains:

- Basic instance metadata (e.g., name, subgroup, set)
- Number of facilities (`I`) and customers (`J`)
- Customer demands (`D_j`) as a list
- Facility capacities (`Q_i`) as a list
- Facility fixed costs (`F_i`) as a list
- Transportation costs (`c_ij`) as a **2D list** (matrix of size `I × J`)

The provided example file `i300_1.json` follows this format:

```json
{
  "info": {
    "name": "i300_1",
    "subgroup": "TB1",
    "set": "TBED1",
    "path": "../data/i300_1.json"
  },
  "params": {
    "I": 300,
    "J": 300,
    "D_j": [  ],       
    "Q_i": [  ],        
    "F_i": [  ],        
    "c_ij": [             
      [  ],
      [  ],
      [  ]
    ]
  }
}
```
