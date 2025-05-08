# -*- coding: utf-8 -*-
"""
general_mip.py

Defines a general Mixed Integer Programming (MIP) class using the CPLEX solver.
Includes methods to configure, solve, and extract solutions from MIP models.

"""

from docplex.mp.model import Model
import cplex.callbacks as cpx_cb
from models.solution import Solution  

class MIP:
    """
    Class for creating and solving Mixed Integer Programming (MIP) models.
    """

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

    def _solve(self, 
              log_output: bool = False,
              timelimit: int = 3600,
              mipgap: float = 0.0001) -> Solution:
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
       
        # Configure solver behavior
        self.mipgap = mipgap
        self.timelimit = timelimit
        self.m.context.solver.log_output = log_output
        self.m.parameters.mip.tolerances.mipgap = self.mipgap
        self.m.parameters.mip.display.set(4)  # Set verbosity level
        self.m.log_output = log_output

        self.m.set_time_limit(self.timelimit)

        # Solve model
        if self.m.solve():
            solution_dvars = self._resolve_decisions(self.dvars)
            optimal = self.m.solve_details.mip_relative_gap <= mipgap
            sol = Solution(
                instance=self.instance,
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