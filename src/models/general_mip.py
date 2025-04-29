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
        Initialize the MIP model.

        Args:
            instance (Instance): The problem instance containing input data.
        """
        self.instance = instance
        self.m = Model(instance.data["info"]["name"])
        self.dvars = dict()
        self.constraints = dict()

    def solve(self, 
              log_output: bool = False,
              timelimit: int = 3600,
              mipgap: float = 0.0001,
              conditional_timelimit: int = None) -> Solution:
        """
        Solve the MIP model with specified solver parameters.

        Args:
            log_output (bool): If True, display solver logs.
            timelimit (int): Maximum solving time (seconds).
            mipgap (float): Relative optimality gap tolerance.
            conditional_timelimit (int, optional): Extended time limit if no feasible solution is found.

        Returns:
            Solution: A Solution object containing results from the solve.
        """
        self.mipgap = mipgap
        self.timelimit = timelimit

        # Set solving parameters
        self.m.context.solver.log_output = log_output
        self.m.parameters.mip.tolerances.mipgap = self.mipgap
        self.m.parameters.mip.display.set(4)  # Verbosity
        self.m.log_output = log_output

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

class SimpleCallback(cpx_cb.MIPInfoCallback):
    """
    Simple callback for demonstration purposes.
    """

    def __init__(self, env):
        super().__init__(env)
        print("SimpleCallback initialized.")

    def __call__(self):
        print("SimpleCallback triggered.")

class ConditionalTimelimitCallback(cpx_cb.MIPInfoCallback):
    """
    Custom callback to implement conditional time limits based on feasibility.
    """

    def __init__(self, env):
        super().__init__(env)
        self.general_timelimit = 0
        self.conditional_timelimit = 0
        self.timelimit_exceeded = False

    def __call__(self):
        elapsed_time = self.get_time() - self.get_start_time()

        if elapsed_time > self.general_timelimit:
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
                self.abort()
