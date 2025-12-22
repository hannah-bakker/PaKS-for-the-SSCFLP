# -*- coding: utf-8 -*-
from docplex.mp.model import Model
from models.solution import Solution

class MIP:
    """
    Base class for formulating and solving Mixed Integer Programs (MIPs)
    using the CPLEX/DOcplex API.
    """
   
    def __init__(self, instance):
        """
        Create an empty DOcplex model and initialize data structures.

        Parameters
        ----------
        instance : object
            Problem instance providing the input data.
        """
        self.instance = instance
        self.m = Model(instance.data["info"]["name"])
        self.dvars = {}
        self.constraints = {}


    # ------------------------------------------------------------------
    # Solve wrapper
    # ------------------------------------------------------------------  
    def _solve(self, 
               log_output = False,
               timelimit= 3600, 
               mipgap = 0.0001, 
               ):
        """
        Solve the MIP using the given solver parameters.

        Parameters
        ----------
        log_output : bool, optional
            Enable solver log output to stdout.
        timelimit : int, optional
            Maximum solving time in seconds.
        mipgap : float, optional
            Relative optimality gap tolerance.

        Returns
        -------
        Solution
            Object collecting model status, objective value, and variable values.
        """
        self.mipgap = mipgap
        self.timelimit = timelimit

        # Configure solver behaviour
        self.m.context.solver.log_output = log_output
        self.m.parameters.mip.tolerances.mipgap = mipgap
        self.m.parameters.mip.display.set(4)       # Log verbosity
        self.m.log_output = log_output
        self.m.set_time_limit(timelimit)

        solved = self.m.solve()

        
        if solved:
            # Extract decision variable values from the model
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
            # Infeasible or no solution returned within limits
            sol = Solution(
                instance=self.instance,
                name=str(self.m.name),
                obj="infeasible",
                best_bound="infeasible",
                time=round(self.m.solve_details.time, 1),
                mip_rel_gap=None,
                optimal=False,
            )

        return sol

    # ------------------------------------------------------------------
    # Resolve decision variables after solving
    # ------------------------------------------------------------------
    def _resolve_decisions(self):
        """
        Extract and return solved decision variable values.

        Returns
        -------
        dict
            Mapping of variable identifiers → values.
        """
        return {}
