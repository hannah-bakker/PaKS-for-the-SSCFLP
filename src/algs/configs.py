# =============================================================================
# PaKS Configuration (Pattern-based Kernel Search)
# =============================================================================

default = {
    "name": "default",
    "num_VI": 5,                                # Maximum number of iterations of adding inequalities (4)
    "N": 10,                                    # N+1 is the number of LP relaxations soved
    "theta":0.05,                               # Threshold on inter-regional validity
    "lambda":"(1,N^-1,...,N^-1)",               # Weights assigneed to solutions in S   
    "p":2,                                      # Number of iterations before removing variables from kernel K
}