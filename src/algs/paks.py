# -*- coding: utf-8 -*-
import time
import itertools
import numpy as np
import json
import logging

from models.cb_report import IntermediateReportingCallback as CB
from utils.biclustering import Biclustering as BC
from utils.helpers import time_step

class PaKS():
    """
       Wrapper class implementing both PaKS and KS14.
    """

    def __init__(self, instance, problem, configuration, data = None, file = None, logger = None):
        """
        Initialize the Kernel Search algorithm.
    
        Parameters
        ----------
        instance : object
            The problem instance.
        problem : class
            The problem model class.
        configuration : dict
            Configuration settings for the algorithm.
        """
        self.configuration = configuration
        self.instance = instance
        self.problem = problem
        self.file = file 
        self.data = data
        if self.data: 
            self.data["algorithm_terminated"] = False

        # Global stopping criterion
        self.done = False       
        self.status = "Started"
        
        # inherit main's logger if provided; else fall back to module logger
        if logger: self.logger = logger or logging.getLogger(__name__)

        # Objects to be used and reused during the procedure
        self.model = None                           # Model object - generated once, then upper bounds 
                                                    # are modified to model restricted MIPs
        self.S: dict[int, object] = {}              # Set of solutions derived from the LP relaxation
        self.K: dict[str, list] = {}                # Kernel
        self.Bcal: dict[int, dict[str, list]] = {}  # Sequence of buckets
        
        
         # Parameters determined throughout the procedure
        self.len_initial_kernel = 0                 # Number of "Y" variables in initial kernel        
        self.NB = 0                                 # Number of buckets
        self.gamma_reduced_costs_ij = 0             # Threshold on the reduced cost coefficient of x-variables

        # Info on procedure
        self.z_LP_0 = 0                             # Objective value of the initial LP-relaxation
        self.z_H_per_iteration = []                 # Development of the objective value
        self.I_K = list()
        self.num_iterations_with_improvement = 0    # Number of iterations with improvement
        self.iterations_with_improvements = []      # Iterations with improvement
        self.iterations_with_change_in_I_KS = []    # Iterations with change in the set of operating facilities
        
        # Reporters on MIP solve process
        self.MIPs_solved = 0                        # Number of restricted MIPs to be solved
        self.optimally_solved = []                  # Number of restricted MIPs solved to optimality before 
                                                    # the time limit was reached 
        self.mipgap_per_MIP = []                    # MIP gap for each MIP
        
        # Time reporters
        self.method_times = {}
        self.LP_times = {}
        self.time_generate_LP_0 = 0                 # Time to generate LP 0
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
        The total execution time is logged at the end.
        Args:
            verbose (bool, optional): If True, prints output to the console. Defaults to False.
        Raises:
            StopIteration: If the kernel search is stopped prematurely.
        Logs:
            The method logs the reason for stopping if a StopIteration exception is raised.
            It also logs the final solution and the total time taken for the kernel search.
        """
        self.start_time_KS = time.time()
        self.verbose = verbose # whether or not to print output in console

        try:
            # Initialization
            self.check_time()
            if self.logger: self.logger.info("Start Phase 1")
            self.phase1()
            
            if self.logger: self.logger.info("Start Solving MIP(K)")
            self.solve_MIP_K() 
            self.check_time()
            if self.file is not None: self.update_status("Solved MIP(K)")
            
            if self.logger: self.logger.info("Start Improvement iterations")
            self.improvement()
           
        except StopIteration as e:
            self.logger.error(f"Kernel Search stopped: {e}")
        finally:
            self.logger.info(f"\nKS solution: {self.z_H}.")          
        
    @time_step
    def phase1(self,
                   ):
        """
        Perform initialization steps for the Kernel Search algorithm.
        """
        self.generate_S()
        if self.logger: self.logger.info("Generated set of solutions.")   
        self.check_time()
        
        if self.configuration["name"] == "PaKS":
            self.generate_regions()
            self.check_time()
        
        self.sort_variables()
        self.check_time()
        
        self.derive_kernel_K()
        self.check_time()
        
        self.derive_buckets_Bcal()       
        self.check_time()
        
    @time_step
    def generate_S(self,
                   ):
        """
        Generate the initial set of solutions. Depending on the configuration, either 
        generate only the LP relaxation or derive a set of solutions from it.
        """
        # Generate model object
        self.model = self.problem(self.instance)  
        
        # Generate initial solutions based on the configuration
        if self.configuration["name"] == "KS14":
             self._generate_initial_LP_relaxation()
        elif self.configuration["name"] == "PaKS":
             self._generate_solution_set()
        else:
             raise ValueError(f"Unknown configuration name: {self.configuration['name']}")

        # Check feasibility of the LP relaxation solution
        self.time_generate_LP_0 = self.S[0].data["time"]
        
        if self.S[0].data["obj"] == "infeasible":
            self.done = True
            self.check_done(message="Stop kernel search! The LP-relaxation is infeasible!")
        else:
            self.logger.info("Generated initial set of solutions successfully.")
   
    def _generate_initial_LP_relaxation(self):
        """
        Generate the LP relaxation for KS14.
        """
        self.S = {
            0: self.model._get_LP_solution(
                timelimit=self.configuration["total_timelimit"],
                logger = self.logger,
            )
        }   
        
    def _generate_solution_set(self):
        """
        Generate a set of solutions for PaKS.
        """
        self.S = self.model._get_S(num_VI = 5, 
                                   N = 10, 
                                   logger = self.logger)
    
    @time_step
    def generate_regions(self,
                         ):
        """
            Generate the regions from S.
        """
        biclustering = BC(
                S=self.S,
                theta=0.05,
                logger = self.logger
                )
        
        self.Rcal = biclustering.Rcal
            
    @time_step
    def sort_variables(self,
                       ):
        """
        Sort y-variables and obtain dictionaries with sorting values for x-variables. 
        """
        self.logger.info("\n Start sorting variables...")
       
        # =============================================================================
        # Cache variables and use NumPy arrays for performance
        # =============================================================================
        # Extract solution keys and number of solutions
        S_keys = list(self.S.keys())
        num_solutions = len(S_keys)

        # ------------------------------------------------------------------
        # Solution weights:
        # - If there is only one solution, weight = 1
        # - Otherwise: solution 0 has weight 1, all others share weight 1
        # ------------------------------------------------------------------
        if num_solutions == 1:
            weights = np.array([1.0])
        else:
            weights = np.array([1.0] + [1.0 / (num_solutions - 1)] * (num_solutions - 1))    
        
        # Facility and customer index ranges
        I_keys = np.arange(self.instance.data["params"]["I"])
        J_keys = np.arange(self.instance.data["params"]["J"])

        # =============================================================================
        # Weighted demand calculation for open facilities
        # =============================================================================
        # x_data[s, i, j] = x_ij value of solution s
        x_data = np.array([
            [[self.S[s].data["dvars"]["x"][(i, j)] for j in J_keys] for i in I_keys]
            for s in S_keys
        ])

        # Per-customer demand (vector of length |J|)
        demand = np.array(self.instance.data["params"]["D"])

        # y_open[s, i] = 1 if facility i is open in solution s (0 otherwise)
        y_open = np.array([
            [self.S[s].data["dvars"]["y"][i] for i in I_keys] for s in S_keys
        ])

        # -------------------------------------------------------------------------
        # Weighted demand:
        #   For each facility i, sum over all customers j:
        #        (weighted frequency of assigning j to i) * demand[j]
        #
        # Explanation:
        #   • x_data * demand broadcasts demand[j] across all i, s.
        #   • tensordot(weights, …) computes the weighted sum across solutions.
        #   • sum(axis=1) then sums over customers j.
        # -------------------------------------------------------------------------
        # Weighted sum across solutions (contract over s)
        weighted_demand = np.tensordot(weights, x_data * demand, axes=(0, 0))

        # Aggregate over customers j → vector of size |I|
        weighted_demand = weighted_demand.sum(axis=1)

        # Identify facilities that are open in at least one solution
        I_open_mask = y_open.sum(axis=0) > 0

        # Build dictionary of weighted demand for open facilities only
        I_open = {i: weighted_demand[i] for i in I_keys[I_open_mask]}

        # Sort by descending weighted demand
        sorted_I_open = dict(
            sorted(I_open.items(), key=lambda item: item[1], reverse=True)
        )

        # =============================================================================
        # Handle closed facilities (if no_rest is False)
        # =============================================================================
        if not self.configuration.get("no_rest", True):
            # reduced_costs_y[s, i] = reduced cost of y_i in solution s
            reduced_costs_y = np.array([
                [self.S[s].data["reduced_costs"]["y"][i] for i in I_keys]
                for s in S_keys
            ])
            # Facilities never open in any solution
            I_closed_mask = ~I_open_mask
            # For closed facilities, score by weighted average reduced cost
            I_closed = {
                i: (weights @ reduced_costs_y[:, i])
                for i in I_keys[I_closed_mask]
            }

            # Sort ascending by reduced cost (lower is more attractive to open)
            sorted_I_closed = dict(
                sorted(I_closed.items(), key=lambda item: item[1])
            )
        else:
            sorted_I_closed = {}        
        # =============================================================================
        # Final sorted y-variable list
        # =============================================================================
        # First all currently open facilities (sorted by weighted demand),
        # then closed facilities (sorted by reduced cost).
        self.sorted_I = {**sorted_I_open, **sorted_I_closed}

        # =============================================================================
        # Reduced costs for x-variables
        # =============================================================================
        # All (i, j) pairs in the instance
        IJ_keys = itertools.product(range(self.instance.data["params"]["I"]), range(self.instance.data["params"]["J"]))
        # Compute a threshold (median) for reduced costs of x_{ij} in the first solution
        x_red_costs = list(self.S[0].data["reduced_costs"]["x"].values())
        threshold = np.percentile(x_red_costs, 50)
        
        # Keep only (i, j) pairs with reduced cost below the threshold
        self.sorted_red_costs = {
            ij for ij in IJ_keys
            if self.S[0].data["reduced_costs"]["x"][ij] < threshold
        }
        
        # =============================================================================
        # Transportation cost threshold (PaKS only)
        # =============================================================================
        if self.configuration["name"] == "PaKS":
            # Compute per-customer median transportation cost
            c_matrix = np.array(self.instance.data["params"]["c"])
            cost_thresholds = np.percentile(c_matrix, 50, axis=0)   # vector of size |J|
            self.sorted_transport =  {
                        (i, j)
                        for i in range(self.instance.data["params"]["I"])
                        for j in range(self.instance.data["params"]["J"])
                        if c_matrix[i, j] < cost_thresholds[j]
            }
        # -----------------------------------------------------------------------------
        # Logging
        # -----------------------------------------------------------------------------
        self.logger.info("Variable sorting completed.")
        self.logger.info(f"Number of sorted facilities (y-variables): {len(self.sorted_I)}")
        I_keys = list(self.sorted_I.keys())
        self.logger.info(f"Facility list length = {len(I_keys)}, unique = {len(set(I_keys))}")
    
    def _derive_x_for_y(self,
                         Y
                         ):
        """
        Derive the set of x_ij variables to include in the kernel (or bucket)
        given a set of facilities Y.
    
        Parameters
        ----------
        Y : set[int]
            Set of indices of y-variables (facilities).

        Returns
        -------
        list[tuple[int, int]]
            List of (i, j) pairs corresponding to selected x_ij variables.
        """
        K_x: list[tuple[int, int]] = []    
        
        if self.configuration["name"] == "PaKS":
           # Region-wise selection of x_ij
           for R in self.Rcal.values():
                I_R = [i for i in Y if i in R[0]]
                if not I_R:
                    continue

                # Inner-regional allocations: i in I_R, j in region's customer set,
                # and (i, j) passes the reduced-cost filter.
                inner_allocations = [
                    (i, j)
                    for i in I_R
                    for j in R[1]
                    if (i, j) in self.sorted_red_costs
                ]
                K_x.extend(inner_allocations)
        
                # Grey zone around region:
                # x_ij with i in I_R and (i, j) selected both by reduced cost
                # and transport-cost criteria.
                I_R_set = set(I_R)
                K_x.extend(
                    ij
                    for ij in self.sorted_red_costs
                    if ij[0] in I_R_set and ij in self.sorted_transport
                )
          
        elif self.configuration["name"] == "KS14":
            # KS14: all promising (i, j) arcs (by reduced cost) starting from facilities in Y
            K_x = [ij for ij in self.sorted_red_costs if ij[0] in Y]

        # Remove duplicates (e.g., if arcs were added via multiple paths)
        return list(set(K_x))    

    @time_step
    def derive_kernel_K(self,
                              ):
        """
        Derive the initial kernel K for the kernel search. 
        """   
        self.logger.info("\nDeriving the initial kernel.")

        # Initialize kernel container
        self.K = {"y": [], "x": []}
        sorted_I = list(self.sorted_I.keys())
        
        # ------------------------------------------------------------------
        # Step 1: derive kernel y-variables
        # ------------------------------------------------------------------
        if self.configuration["name"] == "KS14":
            # KS14: kernel = all facilities with y_i > 0 in the initial LP relaxation
            self.K["y"] = [
                i for i in sorted_I
                if self.S[0].data["dvars"]["y"][i] > 0
            ]
                 
        elif self.configuration["name"] == "PaKS":
            # PaKS: region-wise selection
            # For each region R:
            #   - count how many facilities are open in the initial LP,
            #   - take that many facilities from the region's sorted list. 
            for R in self.Rcal.values():
                I_R = set(R[0])  # facilities in region r_index
                m_R = sum(
                    1 for i in I_R if self.S[0].data["dvars"]["y"][i] > 0
                )
                sorted_I_R = [i for i in sorted_I if i in I_R]
                self.K["y"].extend(sorted_I_R[:m_R])                            
        
        # Store size of initial kernel
        self.len_initial_kernel = len(self.K["y"])
        self.logger.info(f"Number of facilities in initial kernel {len(self.K['y'])}.")
       
        # ------------------------------------------------------------------
        # Step 2: determine which facilities are not yet in the kernel
        # ------------------------------------------------------------------
        self.unassigned_Y_i = [i for i in sorted_I if i not in self.K["y"]]
    
        # ------------------------------------------------------------------
        # Step 3: derive kernel x-variables associated with K["y"]
        # ------------------------------------------------------------------
        self.K["x"] = self._derive_x_for_y(set(self.K["y"]))
    
        self.logger.info(
            f"Derived initial kernel of size "
            f"(|K_y|={len(self.K['y'])}, |K_x|={len(self.K['x'])})."
        )
        
    @time_step        
    def derive_buckets_Bcal(self):
        """
        Derive the bucket family Bcal for the kernel search algorithm.
        """    
        self.logger.info("\nDeriving buckets.")

        # =============================================================================
        # Step 1: determine bucket contents for y-variables
        # =============================================================================
        if self.configuration["name"] == "KS14":
            # KS14: fixed bucket size equal to initial kernel size
            self.length_bucket_y = self.len_initial_kernel
            bucket_indices_y = [
                self.unassigned_Y_i[i : i + self.length_bucket_y]
                for i in range(0, len(self.unassigned_Y_i), self.length_bucket_y)
            ]
        
        elif self.configuration["name"] == "PaKS":
            # PaKS: one bucket per region containing its unassigned facilities
            bucket_indices_y = []
            for R in self.Rcal.values():
                # Unassigned facilities in region r_index
                bucket = [i for i in R[0] if i in set(self.unassigned_Y_i)]
                if bucket:
                    bucket_indices_y.append(bucket)

        # Track all facilities that end up in the kernel or in a bucket (for logging)
        assigned_y = list(self.K["y"])
        for B in bucket_indices_y:
            assigned_y.extend(B)
        assigned_y = set(assigned_y)
        self.logger.info(
            f"Number of facilities in kernel and buckets: {len(assigned_y)}"
        )
            
        # =============================================================================
        # Step 2: determine B_h(x) for each bucket via the helper
        # =============================================================================
        bucket_indices_x = [
            self._derive_x_for_y(set(I_h)) for I_h in bucket_indices_y
        ]

    
         # Merge y and x to form full bucket structures
        self.Bcal = {
            h: {"y": bucket_indices_y[h], "x": bucket_indices_x[h]}
            for h in range(len(bucket_indices_y))
        }
    
        # For KS14, verify that every facility is either in the kernel or exactly one bucket
        if self.configuration["name"] == "KS14":
            total_assigned = len(self.K["y"]) + sum(
                len(B["y"]) for B in self.Bcal.values()
            )
            assert (
                total_assigned == self.instance.data["params"]["I"]
            ), "Not all y-variables have been assigned to a bucket."

        # Number of buckets
        self.NB = len(self.Bcal)

        # =============================================================================
        # Step 3: (optional) sort buckets by size for PaKS
        # =============================================================================
        self.Bcal = dict(
            sorted(
                self.Bcal.items(),
                key=lambda item: len(item[1]["y"]),
                reverse=True,
            )
        )
           
        self.logger.info(f"Derived {self.NB} buckets.")

    @time_step
    def solve_MIP_K(self,):
        """
        Solve the restricted MIP over the current kernel K, i.e., MIP(K).

        The method:
        - updates the effective time limit for this restricted solve,
        - restricts variable upper bounds to the current kernel K,
        - optionally attaches a reporting callback,
        - solves MIP(K) and updates incumbent information (z_H, sol_KS, I_K, etc.).

        Raises
        ------
        RuntimeError
            If the underlying solver routine raises an exception. Infeasibility
            of MIP(K) is handled gracefully and does not raise.
        """
        # ------------------------------------------------------------------
        # Step 1: distribute remaining time over upcoming restricted MIPs
        # ------------------------------------------------------------------
        remaining_time = self.configuration["total_timelimit"] - (
            time.time() - self.start_time_KS
        )
        self.update_time_for_restricted_MIPs(
            remaining_time,
            1 + min(self.configuration.get("NB_bar", float('inf')), self.NB),
        )        
        self.logger.info("\n Starting MIP(K) solve. Timelimit=%s", self.time_limit_restricted_MIP)
        
        # ------------------------------------------------------------------
        # Step 2: restrict bounds to current kernel K
        # ------------------------------------------------------------------
        self.model._restrict_to_current_kernel_ubs(
            {"y": list(self.model.dvars["y"].keys()),
            "x": list(self.model.dvars["x"].keys())},
            self.K,
            {"y": [], "x": []},
            {"y": [], "x": []},
        )

        # Optional: attach reporting callback for periodic checkpointing
        if self.file is not None:
            report_callback = self.model.m.register_callback(CB)
            report_callback.file = self.file
            report_callback.data = self.data

        # ------------------------------------------------------------------
        # Step 3: solve restricted MIP(K)
        # ------------------------------------------------------------------
        self.sol_initial_kernel = self.model._solve(
            timelimit=self.time_limit_restricted_MIP,
        )
        
        
        # ------------------------------------------------------------------
        # Step 4: process solution / infeasibility
        # ------------------------------------------------------------------
        if self.sol_initial_kernel.data["obj"] == "infeasible":
            self.logger.info("No feasible solution found for initial kernel.")
            self.z_H = float("inf")
        else: 
            # Feasible solution: update incumbent info
            self.MIPs_solved += 1
            self.z_H = self.sol_initial_kernel.data["obj"]
            self.sol_KS = self.sol_initial_kernel
            
            # Store information on initial kernel solution. 
            self.I_K = [
            i for i, val in self.sol_initial_kernel.data["dvars"]["y"].items()
                if val > 0
            ]
            self.I_KS = list(self.I_K)
            self.z_H_per_iteration.append(self.z_H)
            
            self.logger.info(f"Solved MIP(K) - z_H = {self.z_H}.")
        
        # Book-keeping for diagnostics and post-analysis
        self.optimally_solved.append(self.sol_initial_kernel.data["optimal"])
        self.time_per_MIP.append(self.sol_initial_kernel.data["time"])

        # Flag whether the last MIP was solved to proven optimality
        self.solved_previous_MIP_to_optimality = bool(
            self.sol_initial_kernel.data["optimal"]
        )

    @time_step
    def improvement(self,
                    ):
        """
        Improvement phase of the kernel search.
        """
        self.logger.info("\nStarting the improvement phase.")

        # Facilities currently in the kernel and how often they have been
        # closed (y_i = 0) in subsequent iterations.
        K_unused = {i: 0 for i in self.K["y"]}

        iteration = 0
        max_iterations = min(self.configuration.get("NB_bar", float('inf')), self.NB)

        while iteration < max_iterations:
            self.logger.info(f"\nIteration {iteration + 1}.")
            
            # ------------------------------------------------------------------
            # 1) Select current bucket B_h
            # ------------------------------------------------------------------
            B = self.Bcal[iteration]
            self.logger.info(
                f"K_y={self.K['y']} -- B_h_y={B['y']} -- size "
                f"(|K_y ∪ B_y|={len(B['y']) + len(self.K['y'])}, "
                f"|K_x ∪ B_x|={len(B['x']) + len(self.K['x'])})"
            )

                        
            # ------------------------------------------------------------------
            # 2) Update model constraints for the current bucket
            # ------------------------------------------------------------------
            self.model._remove_enforcing_of_previous_bucket()

            if iteration == 0:
                B_previous = {"y": [], "x": []}
                K_plus = {"y": [], "x": []}
                K_minus = {"y": [], "x": []}
            elif self.z_H == float("inf"):
                # No feasible solution so far: keep all previous kernel vars,
                # but do not enforce any previous bucket structure.
                B_previous = {"y": [], "x": []}
                K_minus = {"y": [], "x": []}
                # K_plus is intentionally left as defined in previous iteration,
                # as we do not shrink the kernel when z_H = ∞.
            else:
                B_previous = self.Bcal[iteration - 1]
            
            # Restrict upper bounds to current kernel + current / previous bucket
            self.model._restrict_to_current_kernel_ubs(B_previous, B, K_plus, K_minus)

            # Apply objective cutoff z ≤ z_H (if finite)
            self.model._add_objective_upper_bound(self.z_H, iteration)
            
            # Enforce that at least one bucket variable is used only if
            # the previous MIP was solved to optimality.
            if self.solved_previous_MIP_to_optimality:
                self.model._enforce_using_the_bucket_variables(
                    B_y=B["y"], iteration=iteration
                )
                       
            # ------------------------------------------------------------------
            # 3) Update time limit for the restricted MIP and solve
            # ------------------------------------------------------------------
            remaining_time = self.configuration["total_timelimit"] - (
                time.time() - self.start_time_KS
            )
            remaining_iterations = min(self.configuration.get("NB_bar", float('inf')), self.NB) - iteration

            self.update_time_for_restricted_MIPs(
                remaining_time,
                remaining_iterations,
            )

            self.logger.info(f"\nSolve MIP(K ∪ B_h) in {self.time_limit_restricted_MIP}s.")

            # Solve restricted MIP:
            # - if z_H = ∞, we are still looking for the first feasible incumbent
            # - otherwise, we look for improvements under the current cutoff
            current_sol = self.model._solve(
                timelimit=self.time_limit_restricted_MIP,
            )

            # Bookkeeping for this iteration
            self.MIPs_solved += 1
            self.optimally_solved.append(current_sol.data["optimal"])
            self.time_per_MIP.append(current_sol.data["time"])
            self.mipgap_per_MIP.append(current_sol.data["mip_rel_gap"])
            self.solved_previous_MIP_to_optimality = bool(current_sol.data["optimal"])

            # ------------------------------------------------------------------
            # 4) Check feasibility and update kernel / incumbent
            # ------------------------------------------------------------------
            if current_sol.data["obj"] == "infeasible":
                self.logger.info("--> Infeasible.")
                iteration += 1
                continue  # move on to next bucket

             # Feasible solution
            self.num_iterations_with_improvement += 1
            self.iterations_with_improvements.append(iteration + 1)

            self.z_H = current_sol.data["obj"]
            self.z_H_per_iteration.append(self.z_H)

            # Extract open / closed facilities from current solution
            I_open = [i for i, y in current_sol.data["dvars"]["y"].items() if y > 0]
            I_closed = [i for i, y in current_sol.data["dvars"]["y"].items() if y <= 0]                
            
            # --------------------------------------------------------------
            # K_+ : facilities in the current bucket that are now open
            # --------------------------------------------------------------
            K_plus = {"y": [i for i in I_open if i in B["y"]]}
            K_plus["x"] = [ij for ij in B["x"] if ij[0] in K_plus["y"]]
                
            if K_plus["y"]:
                self.iterations_with_change_in_I_KS.append(iteration + 1)
                
            # --------------------------------------------------------------
            # K_- : facilities in K that have been closed often enough
            # --------------------------------------------------------------
            K_unused = {i: val + int(i in I_closed) for i, val in K_unused.items()}
            K_minus = {
                "y": [i for i, val in K_unused.items() if val >= self.configuration["p"]]
            }
            K_minus["x"] = [ij for ij in self.K["x"] if ij[0] in K_minus["y"]]

            self.logger.info(
                f"z_H = {self.z_H}\n"
                f"  K_+ = {K_plus['y']}\n"
                f"  K_- = {K_minus['y']}"
            )
                
            # --------------------------------------------------------------
            # 5) Update kernel K = (K \ K_-) ∪ K_+
            # --------------------------------------------------------------
            # Add K_+
            self.K["y"] = self.K["y"] + K_plus["y"]
            self.K["x"] = self.K["x"] + K_plus["x"]

            # Remove K_-
            self.K["y"] = list(set(self.K["y"]) - set(K_minus["y"]))
            self.K["x"] = list(set(self.K["x"]) - set(K_minus["x"]))

            # Drop removed facilities from K_unused tracking
            K_unused = {
                i: val
                for i, val in K_unused.items()
                if i not in set(K_minus["x"])  # NOTE: original code uses "x" here
            }

            # Update currently best KS solution
            self.sol_KS = current_sol
            iteration += 1

            if self.file is not None:
                self.update_status(f"Solved iteration {iteration}")

        # Final extraction of open facilities from best solution
        if self.sol_KS is not None:
            self.I_KS = [
                i for i, y in self.sol_KS.data["dvars"]["y"].items() if y > 0
            ]

        self.logger.info("Improvement phase completed.")
        self.data["algorithm_terminated"] = True
    
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
        Check the remaining time and terminate if time limit is exceeded.
        """
        self.time_remaining = self.configuration["total_timelimit"] - (time.time() - self.start_time_KS)
        if self.time_remaining <= 0:
            self.done = True
            if self.verbose:
                print("Time limit exceeded. Terminating kernel search.")
            raise StopIteration("Time limit exceeded.")
       
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
        self.status = status_msg
        self.get_KPIs()
        with open(self.file, "w") as json_file:       
            json.dump(self.data, json_file, indent=4)
               
    def get_KPIs(self):
        """
            Retrieve KPIs to report on the performance of the heuristic.
        """
        KPIs = {#info on input parameters
                "Configuration":                self.configuration["name"],
                "total_timelimit":              self.configuration["total_timelimit"],
                "status":                       self.status,
                # objectives
                "z_LP_0":                       self.z_LP_0,
                #info on solution
                "z_KS" :                        self.z_H, #objective value 
                 "z_H_per_iteration":           self.z_H_per_iteration,
             #   "I_KS":                         self.I_KS,                  # facilities operating in KS final solution
                "len(I_KS)":                    len(self.I_KS),             # number of facilities operating in KS final solution
                #info on search process
                "len_S":                        len(self.S),                # number of solutions in initial set
                "N_suggested":                  self.model.N_suggested,
                "N":                            self.model.N,                
                "m":                            self.len_initial_kernel,                     # variables in initial kernel 
              #  "I_K":                          self.I_K,                   # facilities operating in initial kernel solution
                "len(I_K)":                     len(self.I_K),              # number of facilities operating in initial kernel solution
                "NB":                           self.NB,                    # parameter NB - how many restricted MIPs were to be solved                
                "NB_improvements":              self.num_iterations_with_improvement,       # for how many of these rounds were there actual improvements?
                "iterations_with_improvements": self.iterations_with_improvements, # during which iterations did the improvements occur
                "iterations_with_change_in_I_KS": self.iterations_with_change_in_I_KS, # during which iterations did the facilities actually change?
                "MIPs_solved":                  self.MIPs_solved,            # how many MIPs were solved in total?
                "optimally_solved":             self.optimally_solved,     # for how many of the MIPs did we find the optimal solution, for how many
                "mipgap_per_MIP":               self.mipgap_per_MIP,        # list of MIP gaps for individual MIPs
                "bucket_size_y":                0,
                "bucket_size_x":                0,
                "bucket_size_y_avg":            0,
                "bucket_size_x_avg":            0,        
                "r":                            self.model.r,   
                "added_special_constraint":     self.model.added_special_constraint,
                "VI_iterations":    self.model.VI_iterations,      
                }
        # =============================================================================
        # Retrieve number of variables (x and y) in kernel and buckets.       
        # =============================================================================
        if "y" in self.K:
            KPIs["bucket_size_y"] = [len(B["y"]) for B in self.Bcal.values()]
            KPIs["bucket_size_x"] = [len(B["x"]) for B in self.Bcal.values()]
            KPIs["bucket_size_y_avg"]  = np.mean(KPIs["bucket_size_y"]) # Get average siye     
            KPIs["bucket_size_x_avg"] = np.mean(KPIs["bucket_size_x"] )
        KPIs.update(self.method_times)
        KPIs.update(self.model.LP_times)
        if self.data is not None:
            
            self.data.update(KPIs)
        return KPIs