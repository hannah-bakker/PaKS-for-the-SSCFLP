# =============================================================================
# PaKS Configuration (Pattern-based Kernel Search)
# =============================================================================

default = {
    "name": "PaKS",
    "num_VI": 5,                                # Maximum number of iterations of adding inequalities (4)
    "N": 10,                                    # N+1 is the number of LP relaxations soved
    "theta":0.05,                               # Threshold on inter-regional validity  
}
