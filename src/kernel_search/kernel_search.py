# -*- coding: utf-8 -*-
"""
kernel_search.py 

Kernel Search Algorithm Implementation, 
both PaKS and KS2014 depending on the configuration

"""

import time
import itertools
import json
import numpy as np

from ..kernel_search.biclustering import Biclustering as BC
from ..utils.helpers import time_step

class KernelSearch():
    """
        Implementation of Kernel Search    
    """

    def __init__(self, instance, problem, configuration, data = None, file = None):
        """
        Initialize the Kernel Search algorithm.
    
        Parameters
        ----------
        instance : object
            Problem instance.
        problem : class
            Problem model class.
        configuration : dict
            Kernel Search configuration.
        data : dict, optional
            Data dictionary for intermediate reporting. The default is None.
        file : str, optional
            Path to file for saving progress. The default is None.
        """
        self.configuration = configuration
        self.instance = instance
        self.problem = problem
        self.data = {}
        self.file = file
        if data is not None: 
            self.data = data
            self.data["algorithm_terminated"] = False
            self.file = file
            
        # Global stopping criterion
        self.done = False       
        self.status = "Started"
        
        # Objects to be used and reused during the procedure
        self.model = None                           # Model object - generated once, then upper bounds 
                                                    # on variables are modified to model restricted MIPs.
        self.S: dict[int, object] = {}              # Set of solutions derived from the LP relaxation
        self.K: dict[str, list] = {}                # Kernel
        self.B: dict[int, dict[str, list]] = {}     # Sequence of buckets
        
        
         # Parameters determined throughout the procedure
        self.len_initial_K_y = 0                    # Number of "Y" variables in initial kernel        
        self.num_Br = 0                             # Number of buckets

        # Time reporters
        self.method_times = {}
        self.LP_times = {}
        self.time_per_MIP = []

        # Information on KS solution
        self.z_H = float('inf')                         # objective value kernel search
        self.I_KS = list()                              # index set of facilities operating in KS solution
        self.sol_KS = None
        
    @time_step
    def run_kernel_search(self,
                          verbose = False,
                          ):
        """
        Executes the kernel search algorithm.
        
        Parameters
        ----------
        verbose : bool, optional
            If True, prints output to console. Default is False.

        Raises
        ------
        StopIteration
            If the algorithm is stopped prematurely.

        Outputs
        -------
        None
            Updates internal attributes with the best solution found.
        """
        self.start_time_KS = time.time()
        self.verbose = verbose # whether or not to print output in console

        try:
            # Phase 1 & KS Initialization
            self.preprocessing()

            # Phase 2 - KS Optimization    
            self.check_time()
            self.solve_MIP_K() 

            self.check_time()
            self.improvement()
           
        except StopIteration as e:
            self.log(f"Kernel Search stopped: {e}")
        finally:
            self.log(f"\nKS solution: {self.z_H}.")          
        
    @time_step
    def preprocessing(self,
                   ):
        """
        Phase 1 & KS Initialization in case of PaKS; 
        Solving the LP relaxation & KS Initialization in case of KS2014.
        """
        self.generate_S()
             
        self.check_time()
        if self.configuration["name"] == "PaKS":
            self.generate_R()
            self.check_time()
        
        # Phase 2 - KS Initialization  
        self.check_time()            
        self.sort_variables()
        
        self.check_time()
        self.derive_kernel_K()
        
        self.check_time()
        self.derive_buckets_Bcal()       
            
        
    @time_step
    def generate_S(self,
                   ):
        """
        Generate the initial set of solutions S. 
        For PaKS a set S; 
        for KS2014 a S contains only the solution to LP(Y U X). 
        """
        self.log("\nInitializing model object.")
        self.model = self.problem(self.instance)  # Generate model object
        
        # Generate initial solutions based on the configuration
        if self.configuration["name"] == "KS2014":
             self._get_LP_solution()
        elif self.configuration["name"] == "PaKS":
             self._get_solution_set()
        else:
             raise ValueError(f"Unknown configuration name: {self.configuration['name']}")

        # Check feasibility of the LP(Y U X) solution s1.
        if self.S[0].data["obj"] == "infeasible":
            self.done = True
            self.check_done(message="Stop kernel search! The LP-relaxation is infeasible!")
        else:
            self.log("Generated initial set of solutions successfully.")
 
    def _get_LP_solution(self):
        """
        Get LP(Y U X) solution s1 for KS2014.
        """
        self.S = {
            0: self.model._get_LP_solution(
                timelimit=self.configuration["total_timelimit"]
            )
        }   
        
    def _get_solution_set(self):
        """
        Generate a set of solutions for PaKS, adapting the number of VIs based on 
        problem size.
        """
        self.S = self.model._get_S(
            timelimit= self.configuration["total_timelimit"]
            num_VI = self.configuration["num_VI"], 
            N = self.configuration["N"], 
            epsilon = self.configuration["epsilon"], 
        )
    
    @time_step
    def generate_R(self):
        """
            Generate the regions from S.
        """
        self.log("\nDeriving regions.")       
        
        biclustering = BC(
            S=self.S,
            instance=self.instance,
            theta =  self.configuration["theta"], 
        )
        
        self.R = biclustering.Rcal
        
        self.log(f"Identified {len(self.R)}regions with an avg. size and l_inter of {biclustering.l_inter}.")
    
    @time_step
    def sort_variables(self):
        """
            Sort y-variables and obtain dictionaries with sorting values for x-variables. 
        """
        self.log("\n Sorting variables...")
        
        # Prepare keys and weights
        S_keys = list(self.S.keys())
        I_keys = np.arange(self.instance.data["params"]["I"])
        J_keys = np.arange(self.instance.data["params"]["J"])

        num_solutions = len(S_keys)
        if num_solutions == 1:
            weights = np.array([1])  # Only one solution, so weight is 1
        elif self.configuration["lambda_s"] == "(1,N^-1,...,N^-1)":
            weights = np.array([1] + [1 / (num_solutions - 1)] * (num_solutions - 1))   
        else:
            weights = np.array([1] * (num_solutions))     
        

        # Calculate Weighted Demand for Open Facilities 
        x_data = np.array([
            [[self.S[s].data["dvars"]["x"][(i, j)] for j in J_keys] for i in I_keys]
            for s in S_keys
        ])
        demand = np.array(self.instance.data["params"]["D_j"])
        y_open = np.array([
            [self.S[s].data["dvars"]["y"][i] for i in I_keys] for s in S_keys
        ])
    
        weighted_demand = np.tensordot(weights, x_data * demand, axes=(0, 0))  # Weighted sum over solutions
        weighted_demand = weighted_demand.sum(axis=1)  # Sum over J_keys
        I_open_mask = y_open.sum(axis=0) > 0  # Mask for open facilities
        I_open = {i: weighted_demand[i] for i in I_keys[I_open_mask]}
        sorted_I_open = dict(sorted(I_open.items(), key=lambda item: item[1], reverse=True))

        # Add Closed Facilities Sorted by Reduced Costs 
        if self.configuration["name"] == "KS2014": # for PaKS facilities that do not operate in and solution in S are no longer considered
            reduced_costs_y = np.array([
                [self.S[s].data["reduced_costs"]["y"][i] for i in I_keys] for s in S_keys
            ])
            I_closed_mask = ~I_open_mask
            I_closed = {i: (weights @ reduced_costs_y[:, i]) for i in I_keys[I_closed_mask]}
            sorted_I_closed = dict(sorted(I_closed.items(), key=lambda item: item[1]))
        else:
            sorted_I_closed = {}        
        
        # Final sorted y-variable list
        self.sorted_I = {**sorted_I_open, **sorted_I_closed}

        # Identify x-Variables with Low Reduced Costs
        IJ_keys = itertools.product(range(self.instance.data["params"]["I"]), range(self.instance.data["params"]["J"]))
        threshold = np.percentile(list(self.S[0].data["reduced_costs"]["x"].values()), 50)
        self.sorted_red_costs = [
            ij for ij in IJ_keys
            if self.S[0].data["reduced_costs"]["x"][ij] < threshold
        ]
        self.sorted_red_costs = set(self.sorted_red_costs)
        
        
        # Identify x-Variables with Low Transportation Costs (PaKS only)
        if self.configuration["name"] == "PaKS":
            threshold = np.percentile(np.array(self.instance.data["params"]["c_ij"]), 50, axis=0)
            self.sorted_transport =  [
                        (facility_idx, customer_idx)
                        for facility_idx in range(self.instance.data["params"]["I"])
                        for customer_idx in range(self.instance.data["params"]["J"])
                        if self.instance.data["params"]["c_ij"][facility_idx][customer_idx] <  threshold[customer_idx]
                    ]
            self.sorted_transport = set(self.sorted_transport)
        
        
        self.log("Variable sorting completed.")
        self.log(f"Number of facilities that are sorted {len(self.sorted_I)}.")
        I_keys = list(self.sorted_I.keys())
        self.log(f"{len(I_keys)} list; {len(set(I_keys))} set.")
    
    def _get_x_variables_associated_with_y_variables(self, Y):
        """
        Derive relevant x_ij variables for a given set of open facilities Y.

        Depending on the configuration, this method selects x_ij variables using:
        - Region-based selection and transportation cost thresholds (PaKS)
        - Reduced-cost filtering (KS2014)

        Parameters
        ----------
        Y : set[int]
            Set of facility indices (y-variables) currently considered (e.g., in kernel or bucket).

        Returns
        -------
        K_x : list[tuple[int, int]]
            List of (i, j) indices corresponding to selected assignment variables x_ij.
        """
        K_x = [] # Initialize result list of relevant x_ij variables        
        
        if self.configuration["name"] == "PaKS":
           for r_index, R in self.R.items():
                I_R = [i for i in Y if i in R[0]]
                if I_R:
                    # Inner-region assignments with low reduced cost
                    inner_allocations = [
                        (i, j) for i in I_R for j in R[1]
                        if  (i, j) in self.sorted_red_costs
                    ]
                    K_x.extend(inner_allocations)
            
                    # Grey zone: Add additional assignments based on transport cost
                    I_R_set = set(I_R)
                    K_x.extend([ij for ij in self.sorted_red_costs if ij[0] in I_R_set 
                               and ij in self.sorted_transport
                                ])
          
        elif self.configuration["name"] == "KS2014":
            # Select all x_ij with reduced cost below threshold, where i is in the current Y  
            K_x = [ij for ij in self.sorted_red_costs if ij[0] in Y]
        
        K_x = list(set(K_x))
        return K_x        

    @time_step
    def derive_kernel_K(self):
        """
        Derive the initial kernel K = (K_y, K_x) of facility and assignment variables.
        """      
        self.log("\nDeriving the initial kernel.")
        
        # Initialize kernel structure
        self.K = {"y": [], "x": []}
        sorted_I = self.sorted_I.keys()
        
        # Get K_y based on configuration
        if self.configuration["name"] == "KS2014":
            # Add all facilities that had y_i > 0 in s1 (stored in S[0])
            self.K["y"] = [i for i in sorted_I if self.S[0].data["dvars"]["y"][i]>0]
                 
        elif self.configuration["name"] == "PaKS":
            # For each region r, count number m_R of facilities active in LP
            # Then take first m_R entries from the sorted region list
            for r_index, R in self.R.items():
                I_R = set(R[0])  # Facilities in the region
                m_R = sum(
                        1 for i in I_R if self.S[0].data["dvars"]["y"][i] > 0
                        )  # Number of operating facilities
                sorted_I_R = [i for i in sorted_I if i in I_R]
                self.K["y"].extend(sorted_I_R[:m_R])                            
        
        # Track kernel size
        self.len_initial_K_y = len(self.K["y"])
        
        self.log(f"Number of facilities in initial kernel {len(self.K['y'])}.")
        self.log(f"{len(self.K['y'])} list; {len(self.K['y'])} set.")
       
        # Track facilities not in kernel (to build buckets later)
        self.unassigned_Y_i = [i for i in sorted_I if i not in self.K["y"]]
    
        # Select corresponding x-variables
        self.K["x"] = self._get_x_variables_associated_with_y_variables(set(self.K["y"]))
    
        self.log(
            f"Derived initial kernel of size ({len(self.K['y'])},{len(self.K['x'])})."
        )
        
    @time_step        
    def derive_buckets_Bcal(self):
        """
        Derive buckets B for kernel search.
        """    
        self.log("\nDeriving buckets.")

        # Determine B_r_y      
        if self.configuration["name"] == "KS2014":
            self.length_bucket_y = self.len_initial_K_y 
            bucket_indices_y = [
                                    self.unassigned_Y_i[i:i + self.length_bucket_y]
                                    for i in range(0, len(self.unassigned_Y_i), self.length_bucket_y)
                                ]
        
        elif self.configuration["name"] == "PaKS":
            # One bucket per region using unassigned y-vars
            bucket_indices_y = []
            for r_index, R in self.R.items():
                # Get unassigned facilities in the region
                bucket = [i for i in R[0] if i in set(self.unassigned_Y_i)]
                if bucket:
                    bucket_indices_y.append(bucket)
        
        # Determine B_r_x for each bucket
        bucket_indices_x = [
            self._get_x_variables_associated_with_y_variables(set(I_h)) for I_h in bucket_indices_y
        ]
    
        # Store bucket structures
        self.B = {h: {"y": bucket_indices_y[h], "x": bucket_indices_x[h]} for h in range(len(bucket_indices_y))}
    
        # Validation for KS2014: all y-variables should be covered
        if self.configuration["name"] == "KS2014":
            assert len(self.K["y"]) + sum([len(B["y"]) for B in self.B.values()])==self.instance.data["params"]["I"], "Not all variables have been assigned to a bucket."
        # Determine the number of buckets
        self.num_Br = len(self.B)

        
        # Sort buckets by size (PaKS only)
        if self.configuration["name"] == "PaKS":
           # Sort all buckets in descending order
            self.B = dict(
                        sorted(self.B.items(), key=lambda item: len(item[1]["y"]), reverse=True)
                    )
        self.log(f"Derived {self.num_Br} buckets.")

    @time_step
    def solve_MIP_K(self,):
        """
        Solve the restricted MIP (MIP(K)) for the current kernel configuration.
        """ 
        self.log("\nStarting MIP(K) solve.")

        # Update the time limit for the restricted MIP
        self.update_time_for_restricted_MIPs(
            self.configuration["total_timelimit"] - (time.time() - self.start_time_KS),
            1 + min(self.configuration["NB_bar"], self.num_Br),
        )        
       
        # Restrict to current kernel   
        self.model._restrict_to_current_kernel_ubs({"y":list(self.model.dvars["y"].keys()), 
                                                    "x":list(self.model.dvars["x"].keys())}, 
                                                             self.K, 
                                                             {"y":[], "x":[]}, 
                                                             {"y":[], "x":[]}
                                                   )
        
        # Solve the MIP
        self.sol_initial_kernel = self.model._solve(timelimit = self.time_limit_restricted_MIP, 
                                                    conditional_timelimit = self.configuration["total_timelimit"] - (time.time() - self.start_time_KS)
                                                    )
        # Check feasibility
        if self.sol_initial_kernel.data["obj"]  == "infeasible":
            self.log("No feasible solution found for intial kernel.")
            self.z_H = float('inf')
        else:   
            
            # Store information on initial kernel solution. 
            self.MIPs_solved +=1
            self.z_H = self.sol_initial_kernel.data["obj"]
            self.sol_KS = self.sol_initial_kernel 
            self.I_KS = [i for i in self.I_K]
            
            self.log(f"Solved MIP(K) - z_H={self.z_H}.")
        
        # Check if optimal solution was found or timelimit was reached.
        self.solved_previous_MIP_to_optimality = bool(self.sol_initial_kernel.data["optimal"])

        

    @time_step
    def improvement(self,
                    ):
        """
        Improvement phase of kernel search.
        """
        self.log("\nStarting the improvement phase.")

        # Initialize unused kernel counter
        K_unused = {i: 0 for i in self.K["y"]}
        iteration = 0
        max_iterations = min(self.configuration["NB_bar"], self.num_Br)

        while iteration < max_iterations:
            self.log(f"\nIteration {iteration + 1}.")
            
            # Get current bucket
            B = self.B[iteration] # Get bucket 
            self.log(f"K_y={self.K['y']} -- B_h_y={B['y']} -- size ({len(B['y'])+len(self.K['y'])}, {len(B['x'])+len(self.K['x'])})")
                        
            # Update constraints from model. 
            self.model._remove_enforcing_of_previous_bucket()
            if iteration == 0:
                B_previous = {"y":[], "x":[]}
                K_plus = {"y":[], "x":[]}
                K_minus = {"y":[], "x":[]}
            elif self.z_H == float('inf'):  # if no feasible solution was found yet, then variables from the previous bucket should still be considered
                B_previous = {"y":[], "x":[]}
                K_minus = {"y":[], "x":[]}
            else:
                B_previous = self.B[iteration-1]
            
            self.model._restrict_to_current_kernel_ubs(B_previous, B, K_plus, K_minus)
            self.model._add_objective_upper_bound(self.z_H, iteration) # upper bound on objective
            
            if  self.solved_previous_MIP_to_optimality: # only enforce using the bucket variables if the previous MIP was solved to optimality
                self.model._enforce_using_the_bucket_variables(B_y = B["y"], iteration = iteration) # use variables from buckets
                       
            # Get time remaining for restricted MIP
            self.update_time_for_restricted_MIPs(self.configuration["total_timelimit"]-(time.time()-self.start_time_KS), 
                                                 min(self.configuration["NB_bar"], self.num_Br)-iteration)

            self.log(f"\nSolve in {self.time_limit_restricted_MIP}s.")

            # Solve restricted MIP (KUB)
            if self.z_H == float('inf'): # if no feasible solution was found yet, then try to find incumbent solution
                 current_sol = self.model._solve(timelimit = self.time_limit_restricted_MIP,
                                                 conditional_timelimit = self.configuration["total_timelimit"] - (time.time() - self.start_time_KS)
                                                 )
            else:
                 current_sol = self.model._solve(timelimit = self.time_limit_restricted_MIP)

            self.solved_previous_MIP_to_optimality = bool(current_sol.data["optimal"])

            # Update based on current solution
            if current_sol.data["obj"] == "infeasible":
                self.log("--> Infeasible.")
                iteration+=1
                continue # start next iteration
            else: 
                # Report improvements
                self.z_H = current_sol.data["obj"]
                
                # Update kernel
                I_open = [i for i, y in current_sol.data["dvars"]["y"].items() if y>0]
                I_closed = [i for i, y in current_sol.data["dvars"]["y"].items() if y<=0]
                
                # K_+
                K_plus = {"y":[i for i in I_open if i in B["y"]]}
                K_plus["x"] = [ij for ij in B["x"] if ij[0] in K_plus["y"]]
                
                if K_plus["y"]: self.iterations_with_change_in_I_KS.append(iteration+1)
                
                # K_-
                K_unused = {i:val+int(i in I_closed) for i, val in K_unused.items()}
                K_minus = {"y":[i for i, val in K_unused.items() if val >= self.configuration["p"]]}
                K_minus["x"] = [ij for ij in self.K["x"] if ij[0] in K_minus["y"]]

                if self.verbose: print(f"z_H = {self.z_H}\n K_+={K_plus['y']}\n  K_minus {K_minus['y']}")
                
                # Add K_+
                self.K["y"] = self.K["y"] + K_plus["y"]
                self.K["x"] = self.K["x"] + K_plus["x"]
                
                # Remove K_-
                self.K["y"] = list(set(self.K["y"]) - set(K_minus["y"]))
                self.K["x"] = list(set(self.K["x"]) - set(K_minus["x"]))
                
                K_unused = {i:val for i,val in K_unused.items() if i not in set(K_minus["x"])}
                
                # =============================================================================
                # Update currently best solution.
                # =============================================================================    
                self.sol_KS = current_sol
                iteration+=1
            if self.file is not None: self.update_status(f"Solved Iteration {iteration}")

        if  self.sol_KS is not None:
            self.I_KS = [i for i, y in self.sol_KS.data["dvars"]["y"].items() if y>0]
        self.log("Improvement phase completed.")
    
    
    # =============================================================================
    # Helper functions
    # =============================================================================
    def check_done(self, message=None):
        """
        Check if the global stopping criterion is met.
        
        Parameters
        ----------
        message : str, optional
            Custom message explaining why the algorithm is stopping.
        """
        if self.done:
            if self.verbose:
                if message:
                    print(f"Kernel search terminated: {message}")
                else:
                    print("Kernel search terminated: global stopping criterion met.")
            # Use a custom exception to stop execution
            raise StopIteration(message or "Global stopping criterion reached.")

    def check_time(self):
        """
        Check the remaining time and update `self.done` if the time limit is exceeded.
        """
        self.time_remaining = self.configuration["total_timelimit"] - (time.time() - self.start_time_KS)
        if self.time_remaining <= 0:
            self.done = True
            if self.verbose:
                print("Time limit exceeded. Terminating kernel search.")
            raise StopIteration("Time limit exceeded.")
    
    def log(self, message):
        """
        Log a message if verbose mode is enabled.
        
        Parameters
        ----------
        message : str
            The message to log.
        """
        if self.verbose:
            print(message)
    
    def update_time_for_restricted_MIPs(self, remaining_time, remaining_restricted_MIPs):
        """
            Update time for the next restricted MIP by taking the remaining time and dividing it 
            equally between the number of restricted MIPs still planned to solve.

        Parameters
        ----------
        remaining_time : float
            remaining time in seconds.
        remaining_restricted_MIPs : int
            number of problems still to be solved.
        """
        if remaining_time>0:
            self.time_limit_restricted_MIP = int(remaining_time/remaining_restricted_MIPs)
        else:
            self.time_limit_restricted_MIP = 0
    
    def get_timings(self):
        """
        Retrieve all recorded times.
        """
        if hasattr(self, "method_times"):
            total_time = sum(self.method_times.values())
            print("Execution Times Summary:")
            for method, elapsed_time in self.method_times.items():
                print(f"{method}: {elapsed_time:.4f} seconds")
            print(f"Total recorded time: {total_time:.4f} seconds")
        else:
            print("No timing data available.")
    
        
    def update_status(self, status_msg):
        """
        Update status message and save KPIs to a JSON file.
        """
        self.status = status_msg
        with open(self.file, "w") as json_file:
            json.dump(self.get_KPIs(), json_file, indent=4)
            json.close()
        
    def get_KPIs(self):
        """
            Retrieve KPIs to report on the performance of the heuristic.
        """
        KPIs = {
                "Configuration":                self.configuration["name"],
                "total_timelimit":              self.configuration["total_timelimit"],
                "status":                       self.status,
                "z_KS" :                        self.z_H, #objective value 
                "len(I_KS)":                    len(self.I_KS),             # number of facilities operating in KS final solution
                "m":                            self.len_initial_K_y,       # variables in initial kernel 
                "len(I_K)":                     len(self.I_K),              # number of facilities operating in initial kernel solution
                "num_Br":                           self.num_Br,                    # parameter NB - how many restricted MIPs were to be solved                
                "bucket_size_y":                0,
                "bucket_size_x":                0,
                "bucket_size_y_avg":            0,
                "bucket_size_x_avg":            0,        
                "r":                            self.model.r,        
                }

        # Retrieve number of variables (x and y) in kernel and buckets.       
        if "y" in self.K:
            KPIs["bucket_size_y"] = [len(B["y"]) for B in self.B.values()]
            KPIs["bucket_size_x"] = [len(B["x"]) for B in self.B.values()]
            KPIs["bucket_size_y_avg"]  = np.mean(KPIs["bucket_size_y"]) # Get average siye     
            KPIs["bucket_size_x_avg"] = np.mean(KPIs["bucket_size_x"] )
        KPIs.update(self.method_times)
        if self.data is not None:
            self.data["algorithm_terminated"] = True
            self.data.update(KPIs)
        return KPIs