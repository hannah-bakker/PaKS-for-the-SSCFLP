# -*- coding: utf-8 -*-
"""
general_mip.py

Defines a general Mixed Integer Programming (MIP) class using the CPLEX solver.
Includes methods to configure, solve, and extract solutions from MIP models,
with support for conditional time limits via custom callbacks.

"""

import time
import json
from docplex.mp.model import Model
from docplex.mp.callbacks.cb_mixin import ModelCallbackMixin
import cplex.callbacks as cpx_cb
from ..model.solution import Solution  

class MIP:
    """
    Class for creating and solving Mixed Integer Programming (MIP) models.
    """

    problem_type = None

    def __init__(self, instance):
        """
        Initialize a generic MIP model.

        Parameters
        ----------
        instance : Instance
            An object containing all data relevant to the MIP model,
            including problem parameters and metadata.
        """
        self.instance = instance # Store the problem instance
        self.m = Model(instance.data["info"]["name"]) # Create a new CPLEX model with a descriptive name
        self.dvars = dict() # Dictionary to store decision variables by name
        self.constraints = dict() # Dictionary to store constraint sets by name

    def solve(self, 
              log_output: bool = False,
              timelimit: int = 3600,
              mipgap: float = 0.0001,
              conditional_timelimit: int = None) -> Solution:
        """
        Solve the MIP model with specified solver parameters.

        Parameters
        ----------
        log_output : bool, optional
            If True, display CPLEX solver log output. Default is False.
        timelimit : int, optional
            Maximum solving time in seconds. Default is 3600.
        mipgap : float, optional
            Relative optimality gap tolerance. Default is 0.0001.
        conditional_timelimit : int, optional
            If set, applies this extended time limit only when no feasible solution
            has been found within the standard timelimit.

        Returns
        -------
        Solution
            A Solution object containing the objective value, variable values,
            solve time, bound, and optimality metadata.
        """
        self.mipgap = mipgap
        self.timelimit = timelimit

        # Configure solver behavior
        self.m.context.solver.log_output = log_output
        self.m.parameters.mip.tolerances.mipgap = self.mipgap
        self.m.parameters.mip.display.set(4)  # Set verbosity level
        self.m.log_output = log_output

        # Apply conditional time limit logic (if provided)
        if conditional_timelimit is not None:
            conditional_timelimit_callback = self.m.register_callback(ConditionalTimelimitCallback)
            conditional_timelimit_callback.general_timelimit = self.timelimit
            conditional_timelimit_callback.conditional_timelimit = conditional_timelimit
        else:
            self.m.set_time_limit(self.timelimit)

        # Solve model
        if self.m.solve():
            solution_dvars = self._resolve_decisions()
            optimal = self.m.solve_details.mip_relative_gap <= mipgap
            sol = Solution(
                instance=self.instance,
                problem_type=self.problem_type,
                name=str(self.m.name),
                obj=self.m.solution.get_objective_value(),
                time=round(self.m.solve_details.time, 1),
                dvars=solution_dvars,
                best_bound=self.m.solve_details.best_bound,
                mip_rel_gap=round(self.m.solve_details.mip_relative_gap, 5),
                optimal=optimal,
            )
        else:
            # Infeasible or solver failed to return a solution
            sol = Solution(
                instance=self.instance,
                name=str(self.m.name),
                obj="infeasible",
                best_bound="infeasible",
                time=round(self.m.solve_details.time, 1),
                optimal=False,
                mip_rel_gap=None,
            )

        return sol

    def _resolve_decisions(self) -> dict:
        """
        Resolve and return decision variable values after solving.

        Returns:
            dict: Dictionary of variable names and their solved values.
        """
        return dict()

class ConditionalTimelimitCallback(cpx_cb.MIPInfoCallback):
    """
    Custom callback to implement a two-stage time limit strategy for MIP solving.

    If the solver does not find a feasible solution within `general_timelimit`,
    the search continues until `conditional_timelimit` is reached. This helps 
    avoid early termination in hard instances that might take longer to find 
    a feasible solution.

    Attributes
    ----------
    general_timelimit : float
        Time limit (in seconds) for the first phase before checking feasibility.
    conditional_timelimit : float
        Total time limit (in seconds) if no feasible solution is found in the first phase.
    timelimit_exceeded : bool
        Tracks whether the general time limit has been passed already.
    """
    def __init__(self, env):
        super().__init__(env)
        self.general_timelimit = 0
        self.conditional_timelimit = 0
        self.timelimit_exceeded = False

    def __call__(self):
        """
        Check elapsed time and decide whether to stop the solver 
        based on feasibility and configured time limits.
        """
        elapsed_time = self.get_time() - self.get_start_time()

        if elapsed_time > self.general_timelimit:
            # If no feasible solution found yet
            if not self.has_incumbent():
                if not self.timelimit_exceeded:
                    print(
                        f"General timelimit {self.general_timelimit}s reached with no feasible solution. "
                        f"Continue search with conditional timelimit {self.conditional_timelimit}s."
                    )
                    self.timelimit_exceeded = True

                if not self.has_incumbent() and elapsed_time > self.conditional_timelimit:
                    print(f"Conditional timelimit {self.conditional_timelimit}s reached. Stopping search.")
                    self.abort()
            else:
                 # If a solution has been found in the meantime, stop as usual
                self.abort()
