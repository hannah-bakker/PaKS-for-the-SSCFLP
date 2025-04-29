# -*- coding: utf-8 -*-
"""
ss_cflp.py

Defines the SS_CFLP class, which models the single-source capacitated facility location problem
(SSCFLP) as a mixed-integer program using CPLEX. It extends the base MIP class with problem-specific
constraints, valid inequalities, linear relaxations, and kernel-based heuristics.

"""

import time
import numpy as np
from docplex.mp.relax_linear import LinearRelaxer as LR
from ..model.general_mip import MIP
from ..model.solution import Solution

class SS_CFLP(MIP):
    """
    SSCFLP model class extending the base MIP. Models x_ij and y_i decision variables for
    assignment and facility opening. Supports optional valid inequalities, kernel and bucket constraints,
    and LP-based decomposition.
    """

    NAME = "SS_CFLP"

    def __init__(self, instance, verbose: bool = True):
        """
        Initialize the SSCFLP model.

        Args:
            instance: The Instance object containing problem data.
            verbose (bool, optional): Enable verbose logging. Defaults to True.
        """
        super().__init__(instance)
        self.verbose = verbose
        self.I = list(range(instance.data["params"]["I"]))
        self.J = list(range(instance.data["params"]["J"]))
        self.IJ = [(i, j) for i in self.I for j in self.J]
        self.N = 0  # Number of additional LP relaxations
        self.N_suggested = 0
        self.LP_times = dict()
        self._build(instance.data["params"])

    def _build(self, params: dict) -> None:
        """
        Define the SSCFLP model: variables, constraints, and objective function.

        Args:
            params (dict): Dictionary containing instance parameters.
        """
        # Decision variables
        self.dvars["y"] = self.m.binary_var_dict(self.I, name="y_%s")
        self.dvars["x"] = self.m.binary_var_dict(self.IJ, name="x_%s")

        # Demand satisfaction constraints
        self.m.add_constraints(
            (
                self.m.sum(self.dvars["x"][i, j] for i in self.I) >= 1.0
                for j in self.J
            ),
            names=[f"dem_sat_{j}" for j in self.J],
        )

        # Capacity constraints
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

        # Objective function: fixed + variable costs
        self.fixed_cost = self.m.sum(params["F_i"][i] * self.dvars["y"][i] for i in self.I)
        self.var_cost = self.m.sum(
            params["c_ij"][i][j] * self.dvars["x"][i, j] * params["D_j"][j]
            for i in self.I for j in self.J
        )
        self.objective = self.fixed_cost + self.var_cost
        self.m.minimize(self.objective)
        
    def _add_VIs(self) -> None:
        """
        Add valid inequalities of the form x_ij <= y_i.
        These ensure that a facility must be opened if it serves a customer.
        """
        self.constraints["VIs"] = self.m.add_constraints(
            (self.dvars["x"][i, j] <= self.dvars["y"][i] for (i, j) in self.IJ),
            names=[f"VI_{i}_{j}" for (i, j) in self.IJ],
        )

    def _remove_VIs(self) -> None:
        """
        Remove the valid inequalities (x_ij <= y_i) from the model.
        """
        if "VIs" in self.constraints:
            self.m.remove_constraints(self.constraints["VIs"])
            self.constraints.pop("VIs")
    
    def _add_objective_upper_bound(self, z_H: float, iteration: int) -> None:
        """
        Add an upper bound constraint on the objective during kernel search.

        Args:
            z_H (float): Current upper bound on the objective value.
            iteration (int): Iteration number for naming the constraint.
        """
        self.m.add_constraint(
            self.objective <= z_H,
            ctname=f"upper_bound_iteration_{iteration}"
        )
        
    def _enforce_using_the_bucket_variables(self, B_y: list[int], iteration: int = None) -> None:
        """
        Enforce that at least one facility in the current bucket is opened.

        Args:
            B_y (list[int]): List of facility indices (y variables) in the bucket.
            iteration (int, optional): For naming the constraint.
        """
        self.constraints["use_bucket"] = self.m.add_constraint(
            self.m.sum(self.dvars["y"][i] for i in B_y) >= 1.0,
            ctname=f"use_bucket_iteration_{iteration}"
        )

    def _remove_enforcing_of_previous_bucket(self) -> None:
        """
        Remove the constraint enforcing use of a facility from a previous bucket.
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

        Args:
            B_previous (dict): Previous bucket (keys: 'x', 'y').
            B (dict): Current bucket (keys: 'x', 'y').
            K_plus (dict): Added kernel variables (keys: 'x', 'y').
            K_minus (dict): Removed kernel variables (keys: 'x', 'y').
        """
        # Deactivate previous and removed variables
        self.m.change_var_upper_bounds([self.dvars['x'][ij] for ij in B_previous['x']], ubs=0)
        self.m.change_var_upper_bounds([self.dvars['y'][i] for i in B_previous['y']], ubs=0)
        self.m.change_var_upper_bounds([self.dvars['x'][ij] for ij in K_minus['x']], ubs=0)
        self.m.change_var_upper_bounds([self.dvars['y'][i] for i in K_minus['y']], ubs=0)

        # Activate current and added variables
        self.m.change_var_upper_bounds([self.dvars['x'][ij] for ij in B['x']], ubs=1)
        self.m.change_var_upper_bounds([self.dvars['y'][i] for i in B['y']], ubs=1)
        self.m.change_var_upper_bounds([self.dvars['x'][ij] for ij in K_plus['x']], ubs=1)
        self.m.change_var_upper_bounds([self.dvars['y'][i] for i in K_plus['y']], ubs=1)
    
    def _resolve_decisions(self,
                           dvars
                           ) -> dict:
        """
        Extract solution values from model decision variables.

        Args:
            dvars (dict): Dictionary of decision variables. 

        Returns:
            dict: Dictionary containing variable values for 'x' and 'y'.
        """
        solution_dvars = dict()        
        solution_dvars["y"] = {i: dvars["y"][i].solution_value for i in dvars["y"].keys()}
        if "x" in dvars:
            solution_dvars["x"] = {ij: dvars["x"][ij].solution_value for ij in dvars["x"].keys()} 
        return solution_dvars                  
    
    def _get_LR(self, timelimit: int = 3600) -> Solution:
        """
        Solve the linear relaxation of the SSCFLP as required in standard KS.

        Args:
            timelimit (int): Time limit for solving the LP relaxation.

        Returns:
            Solution: A Solution object with primal values and reduced costs.
        """
        # Add all valid inequalities to the original model
        self.log("Adding all valid inequalities...")
        self._add_VIs()
        
        self.log(f"Solving LP-relaxation with timelimit {timelimit}s...")
        self.m_LR = LR.make_relaxed_model(self.m)
        self.m_LR.parameters.timelimit = timelimit
        self.m_LR.context.solver.log_output = self.verbose
        
        feasible = self.m_LR.solve()

        if not feasible:
            self.log("LP-relaxation is infeasible.")
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

        y = dvar_list[:self.instance.data["params"]["I"]]
        x_vars = sorted(dvar_list[self.instance.data["params"]["I"]:], key=lambda var: (int(var.name.split('_')[1]), int(var.name.split('_')[2])))
        x = np.asarray(x_vars).reshape((self.instance.data["params"]["I"], self.instance.data["params"]["J"]))
        dvars={"x":{(i,j):x[i,j] for i, j in sorted(self.dvars["x"].keys())}},
        dvars={"x":{(i,j):x[i,j] for j in range(self.instance.data["params"]["J"]) for i in range(self.instance.data["params"]["I"])},
                "y":{i:y_val for i,y_val in enumerate(y)}}
        solution_dvars = self._resolve_decisions(dvars)
            
        # Determine if solution is integer feasible
        integer_feasible = (
            all(int(v) == v for v in solution_dvars["y"].values()) and
            all(int(v) == v for v in solution_dvars["x"].values())
        )
        
        # Get reduced costs
        reduced_costs = {
            "y": {i: rc for i, rc in enumerate(self.m_LR.reduced_costs(y))},
            "x": {
                (i, j): val
                for i in range(self.instance.data["params"]["I"])
                for j, val in enumerate(self.m_LR.reduced_costs(x[i, :]))
            }
        }
        
        self._remove_VIs()
        
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

      
    def _get_S_LR_I_setminus_i(self, timelimit: int, VIs: str = "all", iterations_VIs: int = 2) -> dict:
        """
        Generate LP relaxations by selectively removing facilities and solving smaller subproblems.

        Args:
            timelimit (int): Total allowed solving time (seconds).
            VIs (str): Type of valid inequalities. Options: "all" or "lazy".
            iterations_VIs (int): Number of VI addition iterations (only used if VIs="lazy").

        Returns:
            dict: Dictionary containing multiple Solution objects.
        """
        start_time = time.time()
        S = {}

        if VIs == "all":
            self._add_VIs()
            self.m_LR = LR.make_relaxed_model(self.m)
            self.log(f"Solving initial LP-relaxation with VIs (timelimit {timelimit*0.5}s)...")
            self.m_LR.solve()
            self.LP_times["LP_0"] = time.time() - start_time
        else:  # lazy VIs
            self.m_LR = LR.make_relaxed_model(self.m)
            solve_time_per_iter = timelimit * 0.5 / (iterations_VIs + 1)
            self.log(f"Solving initial LP with lazy VIs (solve time per iteration: {solve_time_per_iter:.1f}s)...")
            self.m_LR.solve()
            for iter_count in range(iterations_VIs):
                self._lazy_add_VIs()
                self.m_LR.solve()

        self.log(f"Initial LP objective: {self.m_LR.solution.get_objective_value():.2f}")

        # Extract reduced costs
        dvar_list = list(self.m_LR.iter_variables())
        I, J = self.instance.data["params"]["I"], self.instance.data["params"]["J"]
        x_vars = np.array(dvar_list[I:]).reshape((I, J))
        dvars = {
            "x": {(i, j): x_vars[i, j] for i in range(I) for j in range(J)},
            "y": {i: dvar_list[i] for i in range(I)}
        }
        solution_dvars = self._resolve_decisions(dvars)

        # Save the first solution
        S[0] = Solution(
            instance=self.instance,
            problem_type=self.NAME,
            name=str(self.m_LR.name),
            obj=self.m_LR.solution.get_objective_value(),
            time=self.m_LR.solve_details.time,
            dvars=solution_dvars,
        )

        # Prepare for subset LP relaxations
        open_facilities = [i for i, val in solution_dvars["y"].items() if val > 0]
        subsets = self._split_indices(open_facilities)

        self.LP_times["LP_set"] = time.time()
        D_total = sum(self.instance.data["params"]["D_j"])
        Q_total = sum(self.instance.data["params"]["Q_i"])

        for s, subset in enumerate(subsets):
            if not subset or not self._check_lp_feasibility(D_total, Q_total, subset):
                continue

            if "remove_i" in self.constraints:
                self.m_LR.remove_constraints(self.constraints["remove_i"])

            self.constraints["remove_i"] = self.m_LR.add_constraints(
                (dvars["y"][i] == 0 for i in subset)
            )

            if self.m_LR.solve():
                solution_dvars = self._resolve_decisions(dvars)
                S[len(S)] = Solution(
                    instance=self.instance,
                    problem_type=self.NAME,
                    name=str(self.m_LR.name),
                    obj=self.m_LR.solution.get_objective_value(),
                    time=self.m_LR.solve_details.time,
                    dvars=solution_dvars,
                )

        self.LP_times["LP_set"] = time.time() - self.LP_times["LP_set"]

        if VIs == "all":
            self._remove_VIs()

        return S
    
    def log(self, message: str) -> None:
        """
        Print a log message if verbose mode is enabled.

        Args:
            message (str): Log message.
        """
        if self.verbose:
            print(message)

    def _check_lp_feasibility(self, D_total: float, Q_total: float, indices_to_remove: list) -> bool:
        """
        Check if LP remains feasible after removing facilities.

        Args:
            D_total (float): Total customer demand.
            Q_total (float): Total available facility capacity.
            indices_to_remove (list[int]): List of facilities to remove.

        Returns:
            bool: True if enough capacity remains, False otherwise.
        """
        return (Q_total - sum(self.instance.data["params"]["Q_i"][i] for i in indices_to_remove)) > D_total
    
    
    def _split_indices(self, indices: list) -> list[list[int]]:
        """
        Split facility indices into 10 subsets ensuring enough capacity remains.

        Args:
            indices (list[int]): List of facility indices to split.

        Returns:
            list[list[int]]: List of subsets.
        """
        np.random.seed(12051991)
        indices = np.array(indices)
        n = len(indices)
        D_total = sum(self.instance.data["params"]["D_j"])
        Q_total = sum(self.instance.data["params"]["Q_i"])

        divisor = min(int(np.ceil(2 * (Q_total / D_total))), 10)
        subset_size = int(np.ceil(n / divisor))

        done = False
        while not done:
            total_slots = 10 * subset_size
            base_count = total_slots // n
            extra = total_slots - base_count * n

            targets = {idx: base_count for idx in indices}
            if extra > 0:
                for idx in np.random.choice(indices, extra, replace=False):
                    targets[idx] += 1

            subsets = [set() for _ in range(10)]
            slots_left = [subset_size] * 10

            for idx in np.random.permutation(indices):
                needed = targets[idx]
                available = [j for j, slot in enumerate(slots_left) if slot > 0]
                np.random.shuffle(available)
                for j in available[:needed]:
                    subsets[j].add(idx)
                    slots_left[j] -= 1

            subsets = [list(subset) for subset in subsets]
            valid_subsets = [
                sub for sub in subsets
                if sum(self.instance.data["params"]["Q_i"][i]
                       for i in range(self.instance.data["params"]["I"])
                       if i not in sub) >= D_total
            ]

            if len(valid_subsets) >= 10:
                done = True
            else:
                subset_size = int(subset_size * 0.9)

        return valid_subsets
