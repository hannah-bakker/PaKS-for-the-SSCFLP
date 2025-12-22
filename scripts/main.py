## -*- coding: utf-8 -*-
# python main.py "../data/i300_1.json" default 3600     
import os
import sys
from datetime import datetime
import json
import logging
from logging.handlers import RotatingFileHandler
import warnings

# -------------------------------
# Paths & repo layout
# -------------------------------
HERE = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
LOGS_DIR = os.path.join(REPO_ROOT, "logs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

sys.path.append(SRC_DIR)

# -------------------------------
# Local imports
# -------------------------------
from models.instance import Instance
from models.sscflp import SSCFLP
from models.cb_report import IntermediateReportingCallback as CB
from algs.paks import PaKS
from algs.configs import default

warnings.filterwarnings(
    "ignore",
    message="Boolean Series key will be reindexed to match DataFrame index."
)

# -------------------------------
# Logging setup
# -------------------------------
def setup_logging() -> logging.Logger:
    """Configure the logger used by run_instance() with console + rotating file handlers."""
    # --- Clean root logger so docplex / cplex do not attach extra handlers
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    log_level_str = os.getenv("LOG_LEVEL", "DEBUG").upper()
    level = getattr(logging, log_level_str, logging.DEBUG)

    logger = logging.getLogger("run_instance")
    logger.setLevel(level)
    logger.propagate = False  # avoid duplicate logs if root handlers exist

    # Shared formatter for all handlers
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)

    # Rotating file handler (~4 backups of 1 MB each)
    logfile = os.path.join(LOGS_DIR, "run.log")
    fh = RotatingFileHandler(logfile, maxBytes=1_000_000, backupCount=4, encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)

    # Reset handlers to avoid duplicates when re-running in notebooks or REPL
    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Context information
    logger.info("Repo root: %s", REPO_ROOT)
    logger.info("Results dir: %s", RESULTS_DIR)
    logger.info("Logs dir: %s", LOGS_DIR)
    logger.debug("Python sys.path includes: %s", SRC_DIR)

    return logger

logger = setup_logging()

def _map_config(cfg: str):
    if cfg == "default": return dict(default)
    raise ValueError(f"Unknown config '{cfg}'.")

# -------------------------------
# Core logic
# -------------------------------
def run_instance(path_to_instance: str, config: str, timelimit: int) -> str:
    """
    Run Algorithm on a single instance and write KPIs to a JSON file.

    Returns
    -------
    output_path : str
        Full path to the written JSON results file.
    """
    logger.info("Parameters | instance='%s' | config='%s' | timelimit=%s",
                path_to_instance, config, timelimit)

    # Load instance
    try:
        inst = Instance(path_to_instance)
        logger.debug("Loaded instance. I=%s, J=%s",
                     inst.data["params"].get("I"), inst.data["params"].get("J"))
    except Exception as e:
        logger.exception("Failed to load instance from '%s'.", path_to_instance)
        raise
    
    # Initialize
    name = os.path.basename(path_to_instance).rsplit('.', 1)[0]
    timelimit = int(timelimit)

    info = inst.data.get("info", {}); 
    params = inst.data.get("params", {})
    data = {
        "inst": name,
        "set": info.get("set", ""),
        "subgroup": info.get("subgroup", ""),
        "problem": getattr(SSCFLP, "NAME", "SSCFLP"),
        "I": params.get("I"),
        "J": params.get("J"),
    }

    outfile = f"{timelimit}s-{config}-{name}.json"
    output_path = os.path.join(RESULTS_DIR, outfile)

    try:
            cfg = _map_config(config)
            cfg["total_timelimit"] = timelimit
            logger.debug("Run PaKS with timelimit=%s",timelimit)
            alg = PaKS(instance=inst, problem=SSCFLP, configuration=cfg, data = data, file = output_path, logger = logger)
            logger.info("PaKS initialized.")
            alg.run_kernel_search()
            logger.info("PaKS finished.")
            alg.get_timings()
            kpis = alg.get_KPIs()
            logger.debug("KPIs collected: %s", {k: kpis.get(k) for k in list(kpis)[:10]})
            data.update(kpis)
    except Exception:
            logger.exception("PaKS execution failed.")
            raise

    

    try:
        with open(output_path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False)
        logger.info("Results written to: %s", output_path)
    except Exception:
        logger.exception("Failed to write results to '%s'.", output_path)
        raise

    return output_path

# -------------------------------
# Command-line interface entry point
# -------------------------------
if __name__ == "__main__":
    # Expect: python main.py <path_to_instance> <config> <timelimit>
    if len(sys.argv) < 4:
        logger.error("Usage: python main.py <path_to_instance> <config> <timelimit>")
        sys.exit(1)

    path_to_instance = sys.argv[1]
    config = sys.argv[2]
    timelimit = sys.argv[3]

    try:
        # Run the algorithm on the given instance
        out = run_instance(path_to_instance, config, timelimit)
        logger.info("=== Completed successfully ===")
        logger.info("Output file: %s", out)
        sys.exit(0)

    except Exception as exc:
        # Any failure triggers a non-zero return code
        logger.error("=== Failed: %s ===", exc)
        sys.exit(2)
