# -*- coding: utf-8 -*-
import numpy as np

from docplex.mp.relax_linear import LinearRelaxer as LR

from models.mip import MIP 
from models.solution import Solution

class SSCFLP(MIP):
    """
    Type-1 model with binary location variables y_i and assignment variables x_ij.
    """
    
    def __init__(self, instance, verbose: bool = True):
        """
        Initialize the Type1 MIP instance with problem data.

        Parameters
        ----------
        instance : object
            Problem instance containing parameters in `instance.data["params"]`.
        verbose : bool, optional
            If True, logging/printing is enabled.
        """
        super().__init__(instance)

        p = instance.data["params"]
        self.I = [i for i in range(p["I"])]
        self.J = [j for j in range(p["J"])]
        self.IJ = [(i, j) for i in self.I for j in self.J]

        self.verbose = verbose
        self.N = 0
        self.N_suggested = 0
        self.LP_times = {}

        self.added_special_constraint = False
        self.VI_iterations = 0
        self._build(instance.data["params"])
    
    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------
    def _build(self, params):
        """
        Build decision variables, constraints, and objective for SS-CFLP.

        Parameters
        ----------
        params : dict
            Dictionary with CFLP parameters:
            - I, J, Q, D, F, c, etc. (assumed to be accessible via Type1).
        """
        # ------------------------------------------------------------------
        # Decision variables
        # ------------------------------------------------------------------
        self.dvars["y"] = self.m.binary_var_dict(keys=self.I, name="y_%s")
        self.dvars["x"] = self.m.binary_var_dict(
            [ij for ij in self.IJ],
            name="x_%s",
        )

        # ------------------------------------------------------------------
        # Constraints
        # ------------------------------------------------------------------

        # Demand satisfaction: each customer j must be assigned to at least one facility
        self.m.add_constraints(
            (
                self.m.sum(self.dvars["x"][ij] for ij in self.IJ if ij[1] == j) >= 1.0
                for j in self.J
            ),
            names=["dem_sat" + str(j) for j in self.J],
        )

        # Capacity limits: total assigned demand cannot exceed facility capacity
        self.m.add_constraints(
            (
                self.m.sum(
                    params["D"][ij[1]] * self.dvars["x"][ij]
                    for ij in self.IJ
                    if ij[0] == i
                )
                <= self.dvars["y"][i] * params["Q"][i]
                for i in self.I
            ),
            names=["cap_lim" + str(i) for i in self.I],
        )

        # ------------------------------------------------------------------
        # Objective function
        # ------------------------------------------------------------------
        self.fixed_cost = self.m.sum(
            params["F"][i] * self.dvars["y"][i] for i in self.I
        )
        self.var_cost = self.m.sum(
            params["c"][ij[0]][ij[1]] * self.dvars["x"][ij] * params["D"][ij[1]]
            for ij in self.IJ
        )

        self.objective = self.fixed_cost + self.var_cost
        self.m.objective = self.m.minimize(self.objective)
        
    # ------------------------------------------------------------------
    # Valid inequalities x_ij <= y_i
    # ------------------------------------------------------------------
    def _add_VIs(self):
        """
        Add valid inequalities of the form x_ij <= y_i for all (i, j) in IJ.
        """
        self.constraints["VIs"] = self.m.add_constraints(
            (self.dvars["x"][ij] <= self.dvars["y"][ij[0]] for ij in self.IJ),
            names=[f"VI{ij}" for ij in self.IJ],
        )

    def _remove_VIs(self):
        """
        Remove all previously added valid inequalities x_ij <= y_i.
        """
        if "VIs" in self.constraints:
            self.m.remove_constraints(self.constraints["VIs"])
            self.constraints.pop("VIs")
    
    # ------------------------------------------------------------------
    # Kernel search helpers
    # ------------------------------------------------------------------
    def _add_objective_upper_bound(self, z_H, iteration):
        """
        Add an upper bound constraint on the objective for a minimization model.

        Parameters
        ----------
        z_H : float
            Current incumbent objective value (upper bound).
        iteration : int
            Iteration index to uniquely name the constraint.
        """
        self.m.add_constraint(
            self.objective <= z_H,
            ctname=f"upper_bound_iteration_{iteration}",
        )
        
    def _enforce_using_the_bucket_variables(self, B_y, iteration = None):
        """
        Enforce that at least one facility in the bucket B_y is opened.

        Parameters
        ----------
        B_y : list[int]
            Indices of y-variables belonging to the current bucket.
        iteration : int, optional
            Iteration index for constraint naming.
        """
        self.constraints["use_bucket"] = self.m.add_constraint(
            self.m.sum(self.dvars["y"][i] for i in B_y) >= 1.0,
            ctname=f"use_bucket_iteration_{iteration}",
        )
    
    def _enforce_change_in_incumbent(self, K_y_inactive, K_x_inactive, B_y, iteration = None):
        """
        Enforce a change in the incumbent by activating at least one new
        variable in the bucket or previously inactive sets.

        Parameters
        ----------
        K_y_inactive : iterable
            Indices of y-variables that were inactive in the incumbent.
        K_x_inactive : iterable
            Indices (i, j) of x-variables that were inactive in the incumbent.
        B_y : iterable
            Indices of bucket y-variables.
        iteration : int, optional
            Iteration index for constraint naming.
        """
        self.constraints["change_incumbent"] = self.m.add_constraint(
            self.m.sum(self.dvars["y"][i] for i in B_y)
            + self.m.sum(self.dvars["y"][i] for i in K_y_inactive)
            + self.m.sum(self.dvars["x"][ij] for ij in K_x_inactive)
            >= 1.0,
            ctname=f"use_bucket_iteration_{iteration}",
        )

    def _remove_enforcing_of_previous_bucket(self):
        """
        Remove the constraint enforcing the use of a previous bucket.
        """
        if "use_bucket" in self.constraints:
            self.m.remove(self.constraints["use_bucket"])
            self.constraints.pop("use_bucket")
        
    def _restrict_to_current_kernel_ubs(self, B_previous, B, K_plus, K_minus):
        """
        Restrict upper bounds of x/y variables according to current kernel/bucket.

        Parameters
        ----------
        B_previous : dict
            Keys "x" and "y" with indices whose UB should be set to 0.
        B : dict
            Keys "x" and "y" with indices in the current bucket, UB set to 1.
        K_plus : dict
            Keys "x" and "y" newly added to the kernel, UB set to 1.
        K_minus : dict
            Keys "x" and "y" removed from the kernel, UB set to 0.
        """
        # Remove from previous bucket and K_minus
        self.m.change_var_upper_bounds(
            [self.dvars["x"][ij] for ij in B_previous["x"]], ubs=0
        )
        self.m.change_var_upper_bounds(
            [self.dvars["y"][i] for i in B_previous["y"]], ubs=0
        )

        self.m.change_var_upper_bounds(
            [self.dvars["x"][ij] for ij in K_minus["x"]], ubs=0
        )
        self.m.change_var_upper_bounds(
            [self.dvars["y"][i] for i in K_minus["y"]], ubs=0
        )
        
        # Activate current bucket and K_plus
        self.m.change_var_upper_bounds(
            [self.dvars["x"][ij] for ij in B["x"]], ubs=1
        )
        self.m.change_var_upper_bounds(
            [self.dvars["y"][i] for i in B["y"]], ubs=1
        )

        self.m.change_var_upper_bounds(
            [self.dvars["x"][ij] for ij in K_plus["x"]], ubs=1
        )
        self.m.change_var_upper_bounds(
            [self.dvars["y"][i] for i in K_plus["y"]], ubs=1
        )
    
    # ------------------------------------------------------------------
    # Solution extraction
    # ------------------------------------------------------------------
    def _resolve_decisions(self,
                           dvars
                           ):
        """
        Extract current solution values of decision variables.

        Parameters
        ----------
        dvars : dict
            Mapping "y" and optionally "x" to DOcplex variable objects.

        Returns
        -------
        dict
            Dictionary with numerical values for "y" (and "x" if present).
        """
        solution_dvars = {}
        solution_dvars["y"] = {
            i: dvars["y"][i].solution_value for i in dvars["y"].keys()
        }
        if "x" in dvars:
            solution_dvars["x"] = {
                ij: dvars["x"][ij].solution_value for ij in dvars["x"].keys()
            }
        return solution_dvars                  
    

    # ------------------------------------------------------------------
    # Set of LP relaxations (PaKS Phase 1)
    # ------------------------------------------------------------------  
    def _get_S(self, num_VI: int = 5, N: int = 10, epsilon = 0.05, logger = None): 
        """
        Compute a set of LP-relaxation solutions by enforcing/removing subsets
        of facilities, based on an initial LP solution.
        
        Returns
        -------
        S : dict
            A dictionary of solutions.
        """
        
        S = {}
                                
        if logger: logger.info(f"Start solving LP-relaxation and lazy VIs.")
        
        # Solve initial relaxation
        self.m_LR = LR.make_relaxed_model(self.m)
        self.m_LR.solve()
        dvar_list = [var for var in self.m_LR.iter_variables()]
        I = self.instance.data["params"]["I"]
        J = self.instance.data["params"]["J"]
        x = np.asarray(dvar_list[I:]).reshape((I, J))
        dvars = {
            "x": {
                (i, j): x[i, j]
                for j in range(J)
                for i in range(I)
            },
            "y": {i: y for i, y in enumerate(dvar_list[:I])},
        }
        solution_dvars_LR = self._resolve_decisions(dvars)

        self.r = sum(self.instance.data["params"]["Q"]) / sum(
            self.instance.data["params"]["D"]
        )

        self.r = sum(self.instance.data["params"]["Q"]) / sum(
                self.instance.data["params"]["D"]
            )
        self.num_target_facilities = self.instance.data["params"]["I"] * 1 / self.r
        I_open_in_LR = [
            i for i, y_i in solution_dvars_LR["y"].items() if y_i > 0
        ]
        
        if len(I_open_in_LR) < self.num_target_facilities:
            self.added_special_constraint = True
            self.constraints["special"] = []
            self.constraints["special"].append(
                self.m.add_constraint(
                    self.m.sum(
                        self.dvars["y"][i]
                        for i in range(self.instance.data["params"]["I"])
                    )
                    >= self.instance.data["params"]["I"]
                    * (
                        sum(self.instance.data["params"]["D"])
                        / sum(self.instance.data["params"]["Q"])
                    )
                )
            )
            self.m_LR = LR.make_relaxed_model(self.m)
            self.m_LR.solve()
            dvar_list = [var for var in self.m_LR.iter_variables()]
            x = np.asarray(dvar_list[I:]).reshape((I, J))
            dvars = {
                "x": {
                    (i, j): x[i, j]
                    for j in range(J)
                    for i in range(I)
                },
                "y": {i: y for i, y in enumerate(dvar_list[:I])},
            }
            solution_dvars_LR = self._resolve_decisions(dvars)
            I_open_in_LR = [
                i for i, y_i in solution_dvars_LR["y"].items() if y_i > 0
            ] 
                
        else:
            counter = 0
            VIs_set = set()
            while len(I_open_in_LR) > self.num_target_facilities * (1+epsilon) and counter < num_VI:
                counter += 1
                added = 0
                for i, yi in solution_dvars_LR["y"].items():
                    if yi>0:
                        added+=self.instance.data["params"]["J"]
                        VIs_set.add(i)
                        self.m_LR.add_constraints((dvars["x"][(i, j)] <= dvars["y"][i] for j in self.J))
                self.m_LR.solve()
                dvar_list = [var for var in self.m_LR.iter_variables()]
                x = np.asarray(dvar_list[I:]).reshape((I, J))
                dvars = {
                    "x": {
                        (i, j): x[i, j]
                        for j in range(J)
                        for i in range(I)
                    },
                    "y": {i: y for i, y in enumerate(dvar_list[:I])},
                }
                solution_dvars_LR = self._resolve_decisions(dvars)
                I_open_in_LR = [
                    i for i, y_i in solution_dvars_LR["y"].items() if y_i > 0
                ]

                #
               
        if logger: logger.info(f"Solved LP0.")   
        
        # Reduced costs for the base LP solution
        reduced_costs = {}
        reduced_costs_y = self.m_LR.reduced_costs(dvars["y"].values())
        reduced_costs["y"] = {i: val for i, val in enumerate(reduced_costs_y)}
        reduced_costs["x"] = {}
        for i in range(self.instance.data["params"]["I"]):
            reduced_costs_x_i = self.m_LR.reduced_costs(x[i, :])
            for j, val in enumerate(reduced_costs_x_i):
                reduced_costs["x"][(i, j)] = val

        # Save initial LP-relaxation solution
        S[0] = Solution(
            instance=self.instance,
            name=str(self.m_LR.name),
            obj=self.m_LR.solution.get_objective_value(),
            time=self.m_LR.solve_details.time,
            dvars=solution_dvars_LR,
            reduced_costs=reduced_costs,
        )

        
        # Split indices and derive additional LPs with subsets of facilities
        D_total = sum(self.instance.data["params"]["D"])
        Q_total = sum(self.instance.data["params"]["Q"])

        sublists = self._split_indices(I_open_in_LR, N, epsilon)
                
        if logger: logger.debug(f"Sublists: {sublists}")   
        for I_n in sublists:
            if not I_n or not self._check_lp_feasibility(D_total, Q_total, I_n):
                continue
            
            if "remove_i" in self.constraints:
                self.m_LR.remove_constraints(self.constraints["remove_i"])
            self.constraints["remove_i"] = self.m_LR.add_constraints(
                (dvars["y"][i] == 0 for i in I_n),
            )
            
            
            if self.m_LR.solve():
                solution_dvars = self._resolve_decisions(dvars)
                    
                reduced_costs = {}
                reduced_costs_y = self.m_LR.reduced_costs(dvars["y"].values())
                reduced_costs["y"] = {i: val for i, val in enumerate(reduced_costs_y)}
                reduced_costs["x"] = {}
                for i in range(self.instance.data["params"]["I"]):
                    reduced_costs_x_i = self.m_LR.reduced_costs(x[i, :])
                    for j, val in enumerate(reduced_costs_x_i):
                        reduced_costs["x"][(i, j)] = val

                S[len(S)] = Solution(
                    instance=self.instance,
                    name=str(self.m_LR.name),
                    obj=self.m_LR.solution.get_objective_value(),
                    time=self.m_LR.solve_details.time,
                    dvars=solution_dvars,
                    reduced_costs=reduced_costs,
                )
                
        if "special" in self.constraints:
            self.m.remove_constraints(self.constraints["special"]) 
            
        if logger: logger.debug("S="+str(len(S)))
        return S
    
    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------   
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
        return (
            Q_total
            - sum(self.instance.data["params"]["Q"][i] for i in indices_to_remove)
            > D_total
        )
    
    def _split_indices(self, 
                       indices,
                       N,
                       epsilon,
                       logger = None):
        """
        Randomly split `indices` into 10 subsets with controlled redundancy.

        Design goals
        ------------
        - Each subset contains about n / divisor indices (rounded up),
        where divisor = min(ceil(r), 10) and r is the capacity ratio.
        - Every index appears at least once across all subsets.
        - On average, each index appears roughly the same number of times.
        - Subsets are filled without duplicates.
        """
        # Fix random seed for reproducibility
        np.random.seed(12051991)

        # Normalize indices and basic counts
        indices = np.array(indices)
        n = len(indices)
        
        # Determine target subset size
        num_subsets = N
        divisor = min(int(np.ceil(self.r)), num_subsets)
        subset_size = int(np.ceil(n / divisor))

        # If we'd remove more facilities than we have capacity to "replace",
        # recompute subset_size based on capacity ratios.
        if subset_size > (self.instance.data["params"]["I"] - len(indices)):
            # Facilities not in `indices` (i.e., remaining open facilities)
            indices_out = [
                i for i in range(self.instance.data["params"]["I"]) if i not in indices
            ]
            capa_out = sum(self.instance.data["params"]["Q_i"][i] for i in indices_out)
            capa_in = sum(self.instance.data["params"]["Q_i"][i] for i in indices)

            if logger: logger.debug(f"capa_out {capa_out}")
            if logger: logger.debug(f"capa_in {capa_in}")

            # Scale subset size so that total removable capacity stays reasonable
            subset_size = int((capa_out / capa_in) * len(indices) * (1-epsilon))
            print("Subset size via formula was too big!")

        if logger: logger.info(f"Remove {subset_size}")
        total_slots = num_subsets * subset_size

    
        # ------------------------------------------------------------------
        # Step 1: determine how often each index should appear
        # ------------------------------------------------------------------
        # On average, each index should appear total_slots / n times.
        base_app = total_slots // n
        extra = total_slots - base_app * n
    
        # Start with 'base_app' appearances for every index
        targets = {idx: base_app for idx in indices}

        if extra:
            # Randomly choose 'extra' indices to receive one additional appearance
            extra_idxs = np.random.choice(indices, extra, replace=False)
            for idx in extra_idxs:
                targets[idx] += 1
    
        # ------------------------------------------------------------------
        # Step 2: create empty subsets and track remaining slots
        # ------------------------------------------------------------------
        subsets = [set() for _ in range(num_subsets)]  # use sets to avoid duplicates
        slots_left = [subset_size] * num_subsets       # remaining capacity per subset
    
        # ------------------------------------------------------------------
        # Step 3: assign indices according to their target counts
        # ------------------------------------------------------------------
        for idx in indices:
            needed = targets[idx]
            # Subsets that still have room
            available = [j for j in range(num_subsets) if slots_left[j] > 0]
            np.random.shuffle(available)
            # Assign index to up to 'needed' available subsets
            for j in available[:needed]:
                subsets[j].add(idx)
                slots_left[j] -= 1
    
        # ------------------------------------------------------------------
        # Step 4: fill any remaining slots without duplicates
        # ------------------------------------------------------------------
        for j in range(num_subsets):
            while slots_left[j] > 0:
                candidate = np.random.choice(indices)
                if candidate not in subsets[j]:
                    subsets[j].add(candidate)
                    slots_left[j] -= 1
            
        # Convert each subset from a set to a list for downstream usage
        return [list(s) for s in subsets]

    