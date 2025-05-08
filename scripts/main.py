## -*- coding: utf-8 -*-
import os
import sys

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import json
from models.instance import Instance
from models.ss_cflp import SS_CFLP
from kernel_search.kernel_search import KernelSearch   
from kernel_search.configurations import KS14, PaKS 
import warnings


# Suppress the specific warning
warnings.filterwarnings("ignore", message="Boolean Series key will be reindexed to match DataFrame index.")
    
def run_instance(path_to_instance, config, timelimit):
    
        inst = Instance(path_to_instance)  
        timelimit = int(timelimit)
        name = os.path.basename(path_to_instance).rsplit('.', 1)[0]
        data = {
                "inst": name,
                "I": inst.data["params"]["I"],
                "J": inst.data["params"]["J"]
                }
        if config == "KS14":
            this_config = KS14
        elif config == "PaKS":
            this_config = PaKS
        this_config["total_timelimit"] = timelimit
        
        KS = KernelSearch(instance = inst, problem = SS_CFLP, configuration = this_config)
        KS.run_kernel_search(verbose = True)
        KS.get_timings()
        data.update(KS.get_KPIs())
        with open(f"../results/{str(timelimit)}s-{config}-{name}.json", "w") as json_file:
            json.dump(data, json_file, indent=4)
       

if __name__ == "__main__":
    
    if len(sys.argv) < 3:
        print("Usage: python main.py <path_to_instance> <config> <timelimit>")
        sys.exit(1)
    path_to_instance = sys.argv[1]
    config = sys.argv[2]
    timelimit = sys.argv[3]
    run_instance(path_to_instance, config, timelimit)
  