# =============================================================================
# PaKS Configuration (Pattern-based Kernel Search)
# =============================================================================

PaKS = {
    "name": "PaKS",
    "num_VI": 5,                                # Maximum number of iterations of adding inequalities (4)
    "N": 10,                                    # N+1 is the number of LP relaxations soved
    "epsilon": 1e-8,                            # feasibility tolerance
    "theta":0.05,                               # Threshold on inter-regional validity  
}

# =============================================================================
#KS-2014 Configuration (Guastaroba et al. 2014)
# =============================================================================

KS14 = {
    "name": "KS14",
    "epsilon": 1e-8,          # feasibility tolerance
    # --- Improvement Settings ---
    "NB_bar": float('inf'),   # maximum number of buckets to consider
    "no_rest": False,         # consider all buckets (including rest-bucket)
}