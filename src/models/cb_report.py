import json
import time
import cplex.callbacks as cpx_cb


class IntermediateReportingCallback(cpx_cb.MIPInfoCallback):
    """
    CPLEX MIP information callback for periodic checkpointing.

    Responsibilities
    ----------------
    - Periodically write the current incumbent objective (and optionally
      additional data) to a JSON file.
    - Can be extended to monitor runtime or memory usage.
    """

    def __init__(self, env):
        """
        Initialize callback state.

        Parameters
        ----------
        env : cplex._internal._subinterfaces.MIPInfoCallback
            CPLEX callback environment passed by the solver.
        """
        super().__init__(env)

        # Timestamp of last checkpoint
        self.last_save_time = time.time()

        # Minimum time between checkpoints [seconds]
        self.save_interval = 300

        # Memory threshold placeholder (currently unused, but kept for extension)
        self.memory_threshold = 2048  # MB

        # Output file path; must be set by the caller before solving
        self.file = None

        # Arbitrary data container written to JSON (e.g. best bound, gaps, etc.)
        self.data = {}
    
    def __call__(self):
        """
        Periodic callback entry point, invoked by CPLEX during the solve.

        If the save interval has elapsed and an incumbent is available, the
        current incumbent objective is written to `self.file` in JSON format.
        """
        current_time = time.time()

        # Check whether it is time to perform a checkpoint
        if current_time - self.last_save_time > self.save_interval:
            if self.has_incumbent():
                # Retrieve incumbent variable values if needed
                # (currently unused, but can be stored in self.data["x"] etc.)
                values = self.get_incumbent_values()

                # Store current incumbent objective
                self.data["z_KS"] = self.get_incumbent_objective_value()

                if self.file is not None:
                    with open(self.file, "w") as json_file:
                        json.dump(self.data, json_file, indent=4)

            # Update last checkpoint time regardless of whether we had an incumbent
            self.last_save_time = current_time