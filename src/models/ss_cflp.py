# -*- coding: utf-8 -*-
"""
ss_cflp.py

Defines the SS_CFLP class, which models the single-source capacitated facility location problem
(SSCFLP) as a mixed-integer program using docplex. It extends the base MIP class with problem-specific
constraints, and methods used throughout PaKS and KS2014 to identify solutions to LP relaxations and add or 
remove constraints in restricted BIPs.
"""

import numpy as np
from docplex.mp.relax_linear import LinearRelaxer as LR
from ..model.general_mip import MIP
from ..model.solution import Solution

class SS_CFLP(MIP):
    """
    SSCFLP model class extending the base MIP. 
    
    """

    NAME = "SS_CFLP"

    def __init__(self, instance, verbose: bool = True):
        """
        Initialize the SSCFLP model.

        Parameters
        ----------
        instance : object
            Problem instance data.
        verbose : bool
            If True, prints detailed logs. Default is True.
        """
        super().__init__(instance)
        self.verbose = verbose

        # Basic sets and index lists
        self.I = list(range(instance.data["params"]["I"]))
        self.J = list(range(instance.data["params"]["J"]))
        self.IJ = [(i, j) for i in self.I for j in self.J]
        
        self._build(instance.data["params"])

    def _build(self, params: dict) -> None:
        """
        Construct SSCFLP model: variables, constraints, objective function.

        Parameters
        ----------
        params : dict
            Instance parameters (e.g., demand, cost, capacity).
        """
        # Decision variables
        self.dvars["y"] = self.m.binary_var_dict(self.I, name="y_%s")
        self.dvars["x"] = self.m.binary_var_dict(self.IJ, name="x_%s")

        # Constraint: each customer is served by at least one facility
        self.m.add_constraints(
            (
                self.m.sum(self.dvars["x"][i, j] for i in self.I) >= 1.0
                for j in self.J
            ),
            names=[f"dem_sat_{j}" for j in self.J],
        )

        # Constraint: total demand served by a facility must not exceed its capacity
        self.m.add_constraints(
            (
                self.m.sum(
                    params["D_j"][j] * self.dvars["x"][i, j]
                    for j in self.J
                ) <= self.dvars["y"][i] * params["Q_i"][i]
                for i in self.I
            ),
            names=[f"cap_lim_{i}" for i in self.I],
        )

        # Objective: minimize fixed and variable cost
        self.fixed_cost = self.m.sum(params["F_i"][i] * self.dvars["y"][i] for i in self.I)
        self.var_cost = self.m.sum(
            params["c_ij"][i][j] * self.dvars["x"][i, j] * params["D_j"][j]
            for i in self.I for j in self.J
        )
        self.objective = self.fixed_cost + self.var_cost
        self.m.minimize(self.objective)
        
    def _add_VIs(self) -> None:
        """
        Add valid inequalities (4) of the form x_ij <= y_i.
        These ensure that a facility must be opened if it serves a customer.

        Updates self.constraints by adding a "VIs" entry that stores the added constraints.
        This allows for easy removal later if needed.
        """
        self.constraints["VIs"] = self.m.add_constraints(
            (self.dvars["x"][i, j] <= self.dvars["y"][i] for (i, j) in self.IJ),
            names=[f"VI_{i}_{j}" for (i, j) in self.IJ],
        )

    def _remove_VIs(self) -> None:
        """
        Remove the valid inequalities (x_ij <= y_i) from the model.

        Modifies:
        - self.m: by removing the constraints from the CPLEX model.
        - self.constraints: by deleting the "VIs" entry.
        """
        if "VIs" in self.constraints:
            self.m.remove_constraints(self.constraints["VIs"])
            self.constraints.pop("VIs")
    
    def _add_objective_upper_bound(self, z_H: float, iteration: int) -> None:
        """
        Add an upper bound constraint on the objective during kernel search.

        Parameters
        ----------
        z_H : float
            The current best (incumbent) objective value.
        iteration : int
            Iteration number, used for naming the constraint uniquely.
        """
        self.m.add_constraint(
            self.objective <= z_H,
            ctname=f"upper_bound_iteration_{iteration}"
        )
        
    def _enforce_using_the_bucket_variables(self, B_y: list[int], iteration: int = None) -> None:
        """
        Enforce that at least one facility in the current bucket is opened.

        Modifies:
            - self.constraints["use_bucket"]: Stores the created constraint object.
            - self.m: Adds a constraint to the model.

        Parameters
        ----------
        B_y : list[int]
            Indices of facilities (i ∈ I) in the current bucket.
        iteration : int, optional
            Iteration index used for naming the constraint uniquely.      
        """
        self.constraints["use_bucket"] = self.m.add_constraint(
            self.m.sum(self.dvars["y"][i] for i in B_y) >= 1.0,
            ctname=f"use_bucket_iteration_{iteration}"
        )

    def _remove_enforcing_of_previous_bucket(self) -> None:
        """
        Remove the constraint enforcing use of a facility from a previous bucket.

        Modifies:
        - self.m: Removes the constraint from the model.
        - self.constraints: Deletes the "use_bucket" entry.
        """
        if "use_bucket" in self.constraints:
            self.m.remove(self.constraints["use_bucket"])
            self.constraints.pop("use_bucket")
        
    def _restrict_to_current_kernel_ubs(self,
                                        B_previous: dict,
                                        B: dict,
                                        K_plus: dict,
                                        K_minus: dict) -> None:
        """
        Restrict variable upper bounds based on kernel and bucket membership.

        This method disables (sets UB = 0) for variables that were either in the 
        previous bucket or have been removed from the kernel, and enables (sets UB = 1) 
        for variables in the current bucket and those newly added to the kernel.

        Parameters
        ----------
        B_previous : dict
            Dictionary with keys 'x' and 'y' listing assignment and facility variables 
            from the previous bucket whose upper bounds will be set to 0.
        B : dict
            Dictionary with keys 'x' and 'y' listing variables in the current bucket 
            whose upper bounds will be enabled (set to 1).
        K_plus : dict
            Variables newly added to the kernel; their upper bounds will be set to 1.
        K_minus : dict
            Variables removed from the kernel; their upper bounds will be set to 0.
        """
        # Deactivate variables from previous bucket and those removed from the kernel
        self.m.change_var_upper_bounds([self.dvars['x'][ij] for ij in B_previous['x']], ubs=0)
        self.m.change_var_upper_bounds([self.dvars['y'][i] for i in B_previous['y']], ubs=0)
        self.m.change_var_upper_bounds([self.dvars['x'][ij] for ij in K_minus['x']], ubs=0)
        self.m.change_var_upper_bounds([self.dvars['y'][i] for i in K_minus['y']], ubs=0)

        # Activate variables from the current bucket and those added to the kernel
        self.m.change_var_upper_bounds([self.dvars['x'][ij] for ij in B['x']], ubs=1)
        self.m.change_var_upper_bounds([self.dvars['y'][i] for i in B['y']], ubs=1)
        self.m.change_var_upper_bounds([self.dvars['x'][ij] for ij in K_plus['x']], ubs=1)
        self.m.change_var_upper_bounds([self.dvars['y'][i] for i in K_plus['y']], ubs=1)
    
    def _resolve_decisions(self,
                           dvars
                           ) -> dict:
        """
        Extract solution values from model decision variables.

        This method reads the `.solution_value` from the CPLEX variables for each 
        variable in the 'y' and optionally in the 'x' dictionary. It returns a plain 
        Python dictionary of values, which can be used for reporting or post-processing.

        Parameters
        ----------
        dvars : dict
            Dictionary of CPLEX decision variables, with keys "y" (facility open) 
            and optionally "x" (assignments). Each value is a dict of variables.

        Returns
        -------
        dict
            A dictionary with keys "y" and optionally "x", where:
            - "y" maps facility indices to their solution values (float).
            - "x" maps (i, j) index pairs to their solution values (float).
        """
        solution_dvars = {}

        # Extract solution values for facility open variables
        solution_dvars["y"] = {
            i: dvars["y"][i].solution_value for i in dvars["y"]
        }

        # Extract solution values for assignment variables if present
        if "x" in dvars:
            solution_dvars["x"] = {
                ij: dvars["x"][ij].solution_value for ij in dvars["x"]
            }

        return solution_dvars         
    
    def _get_LP_solution(self, timelimit: int = 3600) -> Solution:
        """
        Solve the linear relaxation of the SSCFLP as required in KS2014.

        Parameters
        ----------
        timelimit : int, optional
            Maximum time in seconds to allow for solving the LP relaxation (default is 3600s).

        Returns
        -------
        Solution
            A `Solution` object containing:
            - The objective value.
            - Decision variable values (`dvars`).
            - Reduced costs (`reduced_costs`).
            - A flag for integer feasibility.
            - Solution time and model name.
        """
        self.log("Adding all valid inequalities...")
        self._add_VIs()
        
        self.log(f"Solving LP-relaxation with timelimit {timelimit}s...")
        
        # Generate and solve relaxed LP
        self.m_LR = LR.make_relaxed_model(self.m)
        self.m_LR.parameters.timelimit = timelimit
        self.m_LR.context.solver.log_output = self.verbose
        feasible = self.m_LR.solve()

        if not feasible:
            self.log("LP relaxation is infeasible.")
            self._remove_VIs()
            return Solution(
                instance=self.instance,
                name=str(self.m_LR.name),
                obj="infeasible",
                time=self.m_LR.solve_details.time,
            )

        self.log(f"LP solved: objective = {self.m_LR.solution.get_objective_value():.2f}")

        # Extract variables
        dvar_list = [var for var in self.m_LR.iter_variables()]

         # First I variables are y_i, rest are x_ij
        y = dvar_list[:len(self.I)]
        x_vars = sorted(dvar_list[len(self.I):], key=lambda var: (int(var.name.split('_')[1]), int(var.name.split('_')[2])))
        x = np.asarray(x_vars).reshape((len(self.I), len(self.J)))
        # Build variable dictionary
        dvars={"x":{(i,j):x[i,j] for j in range(len(self.J)) for i in range(len(self.I))},
                "y":{i:y_val for i,y_val in enumerate(y)}}
        # Extract numerical values
        solution_dvars = self._resolve_decisions(dvars)
            
        # Check if the LP solution is integer-feasible
        integer_feasible = (
            all(int(v) == v for v in solution_dvars["y"].values()) and
            all(int(v) == v for v in solution_dvars["x"].values())
        )
        
        # Extract reduced costs
        reduced_costs = {
            "y": {i: rc for i, rc in enumerate(self.m_LR.reduced_costs(y))},
            "x": {
                (i, j): val
                for i in range(len(self.I))
                for j, val in enumerate(self.m_LR.reduced_costs(x[i, :]))
            }
        }
        
        self._remove_VIs()
        
        # Return the LP solution object
        return Solution(
            instance=self.instance,
            problem_type=self.NAME,
            name=str(self.m_LR.name),
            obj=self.m_LR.solution.get_objective_value(),
            time=self.m_LR.solve_details.time,
            dvars=solution_dvars,
            reduced_costs=reduced_costs,
            integer_feasible=integer_feasible,
        )

      
    
    def _get_S(self, timelimit: int, num_VI: int = 5, N: int = 10, epsilon = 0.05) -> dict:
        """
        Generate LP relaxations by selectively removing facilities and solving smaller subproblems.

        Args:
            timelimit (int): Total allowed solving time (seconds).
            VIs (str): Type of valid inequalities. Options: "all" or "lazy".
            iterations_VIs (int): Number of VI addition iterations (only used if VIs="lazy").

        Returns:
            dict: Dictionary containing multiple Solution objects.
        """        
        S = {}

        # Produce the first solution s1 in S.
        self.rho = sum(self.instance.data["params"]["Q_i"])/sum(self.instance.data["params"]["D_j"])
        self.I_asterix = (len(self.I)/self.rho)

        # Solve initial relaxation
        self.m_LR = LR.make_relaxed_model(self.m)  # Generate linear relaxation     
        self.m_LR.parameters.timelimit = timelimit
        self.m_LR.context.solver.log_output = self.verbose
        self.m_LR.solve()

        dvar_list = [var for var in self.m_LR.iter_variables()]
        x = np.asarray(dvar_list[len(self.I):]).reshape((len(self.I),len(self.J)))
        dvars={"x":{(i,j):x[i,j] for j in range(len(self.J)) for i in range(len(self.I))},
                "y":{i:y for i, y in enumerate(dvar_list[:len(self.I)])}}
        solution_dvars_LR = self._resolve_decisions(dvars)

            
            
        I_dash = len([i for i, y_i in solution_dvars_LR["y"].items() if y_i>0])
        if I_dash < self.I_asterix:
            self.constraints["special"] = []
            self.constraints["special"].append(
                    self.m.add_constraint(
                        self.m.sum(self.dvars["y"][i] for i in range(len(self.I)))
                        >= len(self.I) * (
                            sum(self.instance.data["params"]["D_j"]) /
                            sum(self.instance.data["params"]["Q_i"])
                        )
                    )
                )
            self.m_LR = LR.make_relaxed_model(self.m)
            self.m_LR.solve()
            dvar_list = [var for var in self.m_LR.iter_variables()]
            x = np.asarray(dvar_list[len(self.I):]).reshape((len(self.I),len(self.J)))
            dvars={"x":{(i,j):x[i,j] for j in range(len(self.J)) for i in range(len(self.I))},
                    "y":{i:y for i, y in enumerate(dvar_list[:len(self.I)])}}
            solution_dvars_LR = self._resolve_decisions(dvars)  
            I_dash = len([i for i, y_i in solution_dvars_LR["y"].items() if y_i>0])
                  
        else:
            ell = 0
            VIs = set()
            while I_dash > self.I_asterix *(1 + epsilon) and ell < num_VI:
                ell+=1
                added = 0
                # Enforce VIs lazily
                for ij, xv in solution_dvars_LR["x"].items():
                    if xv>0 and ij not in VIs:
                        added+=1
                        VIs.add(ij)
                        self.m_LR.add_constraint(dvars["x"][ij]<=dvars["y"][ij[0]])
                self.m_LR.solve()     
                dvar_list = [var for var in self.m_LR.iter_variables()]
                x = np.asarray(dvar_list[len(self.I):]).reshape((len(self.I),len(self.J)))
                dvars={"x":{(i,j):x[i,j] for j in range(len(self.J)) for i in range(len(self.I))},
                        "y":{i:y for i, y in enumerate(dvar_list[:len(self.I)])}}
                solution_dvars_LR = self._resolve_decisions(dvars)  
                I_dash = len([i for i, y_i in solution_dvars_LR["y"].items() if y_i>0]) 
        
        I_1 = [i for i, y_i in solution_dvars_LR["y"].items() if y_i>0]
        self.log(f"Solved LP(Y U X).")   
        
        # Get reduced costs for the first solution
        reduced_costs = {}
        reduced_costs_y = self.m_LR.reduced_costs(dvars["y"].values())
        reduced_costs["y"] = {i:val for i, val in enumerate(reduced_costs_y)}
        reduced_costs["x"] = dict()
        for i in range(len(self.I)):
            reduced_costs_x_i = self.m_LR.reduced_costs(x[i,:])
            for j, val in enumerate(reduced_costs_x_i):
                reduced_costs["x"][(i,j)] = val     

        # Save s1.                   
        S[0] = Solution(
                instance = self.instance,
                problem_type = self.problem_type,
                name=str(self.m_LR.name),
                obj=self.m_LR.solution.get_objective_value(),
                time=self.m_LR.solve_details.time,
                dvars=solution_dvars_LR,
                reduced_costs = reduced_costs,
                )
        
        # Produce solutions s2,...,sN+1 in S.
        D_total = sum(self.instance.data["params"]["D_j"])
        Q_total = sum(self.instance.data["params"]["Q_i"])
        
        sublists = self._split_indices(I_1, N, epsilon)
                
        for s, I_n in enumerate(sublists):
            if not I_n or not self._check_lp_feasibility(D_total, Q_total, I_n):
                continue
            
            if "remove_i" in self.constraints:
                self.m_LR.remove_constraints(self.constraints["remove_i"])
            self.constraints["remove_i"] = self.m_LR.add_constraints((dvars["y"][i] == 0 for i in I_n),)
           
            
            if self.m_LR.solve(): # else None
                solution_dvars = self._resolve_decisions(dvars)
                 
                 
                 # Get reduced costs for the first solution
                reduced_costs = {}
                reduced_costs_y = self.m_LR.reduced_costs(dvars["y"].values())
                reduced_costs["y"] = {i:val for i, val in enumerate(reduced_costs_y)}
                reduced_costs["x"] = dict()
                for i in range(len(self.I)):
                     reduced_costs_x_i = self.m_LR.reduced_costs(x[i,:])
                     for j, val in enumerate(reduced_costs_x_i):
                         reduced_costs["x"][(i,j)] = val     
                                      
                S[len(S)] = Solution(
                     instance = self.instance,
                     problem_type = self.problem_type,
                     name=str(self.m_LR.name),
                     obj=self.m_LR.solution.get_objective_value(),
                     time=self.m_LR.solve_details.time,
                     dvars=solution_dvars,
                     reduced_costs = reduced_costs,
                     )    

        if "special" in self.constraints:
            self.m.remove_constraints(self.constraints["special"]) 
     
        return S
    
    def log(self, message):
        """
        Log a message if verbose mode is enabled.
        
        Parameters
        ----------
        message : str
            The message to log.
        """
        print(message)
    
    def _check_lp_feasibility(self, D_total, Q_total, indices_to_remove):
        """
        Check if removing a subset of facilities is feasible in terms of remaining capacity.
    
        Parameters
        ----------
        D_total : float
            Total demand.
        Q_total : float
            Total capacity.
        indices_to_remove : list
            Indices of facilities to remove.
    
        Returns
        -------
        bool
            True if feasible, False otherwise.
        """
        return Q_total - sum(self.instance.data["params"]["Q_i"][i] for i in indices_to_remove) > D_total
    
    def _split_indices(self, indices, N=10, epsilon=0.05):
        """
        Create N overlapping subsets from a given list of facility indices.

        Parameters
        ----------
        indices : list[int]
            List of facility indices open in the LP solution (used as base set).
        N : int, optional
            Number of subsets to create (default is 10).
        epsilon : float, optional
            Tolerance parameter for oversizing when capacity feasibility is close (default is 0.05).

        Returns
        -------
        list[list[int]]
            A list of N subsets (each a list of facility indices) where elements are sampled with
            controlled overlap, used for generating restricted LPs in kernel search.
        """
        np.random.seed(12051991)
        
        indices = np.array(indices)
        num_indices= len(indices)
        
        # Calculate subset size per split (alpha)
        alpha = int(np.ceil(num_indices/ min(int(np.ceil(self.rho)),N)))

        # Fallback if alpha is too large to ensure feasible removal/replacement
        if alpha > (len(self.I)-len(indices)): 
            indices_out = [i for i in range(len(self.I)) if i not in indices]
            capa_out = sum(self.instance.data["params"]["Q_i"][i] for i in indices_out)
            capa_in =  sum(self.instance.data["params"]["Q_i"][i] for i in indices)   
            alpha = int((capa_out/capa_in)*len(indices)*(1-epsilon ))
        
        total_slots = N * alpha # total index slots to fill across all subsets

    
        # Assign base number of appearances to each index
        base_app = total_slots // num_indices
        extra = total_slots - base_app * num_indices
    
        targets = {idx: base_app for idx in indices}
        if extra:
            # Randomly choose indices to get one extra appearance
            extra_idxs = np.random.choice(indices, extra, replace=False)
            for idx in extra_idxs:
                targets[idx] += 1
    
        # Initialize empty subsets and track remaining slots
        subsets = [set() for _ in range(N)]
        slots_left = [alpha] * N
    
         # First pass: assign each index to as many subsets as needed
        for idx in np.random.permutation(indices):
            needed = targets[idx]
            available = [j for j in range(N) if slots_left[j] > 0]
            np.random.shuffle(available)
            for j in available[:needed]:
                subsets[j].add(idx)
                slots_left[j] -= 1
    
        # Fill any remaining slots randomly while avoiding duplicates
        for j in range(N):
            while slots_left[j] > 0:
                candidate = np.random.choice(indices)
                if candidate not in subsets[j]:
                    subsets[j].add(candidate)
                    slots_left[j] -= 1
            
        return [list(s) for s in subsets]
