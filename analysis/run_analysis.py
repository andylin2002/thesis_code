from __future__ import annotations

import argparse

from common import get_project_root, load_yaml
from run_neural_param_search import run_neural_param_search
from run_neural_seed_search import run_neural_seed_search
from run_symbolic_ablation import run_symbolic_ablation


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified analysis entry point.")
    parser.add_argument("--analysis-config", default="analysis/analysis_config.yaml")
    parser.add_argument("--mode", default=None, choices=["symbolic_ablation", "neural_param_search", "neural_seed_search"])
    parser.add_argument("--dry-run", action="store_true", default=False)
    args = parser.parse_args()
    project_root = get_project_root()
    analysis_config = load_yaml(project_root / args.analysis_config)
    mode = args.mode or str(analysis_config.get("mode", "symbolic_ablation"))
    print(f"[Analysis] mode={mode}")
    if mode == "symbolic_ablation":
        run_symbolic_ablation(analysis_config, cases_override=None, dry_run=args.dry_run)
    elif mode == "neural_param_search":
        run_neural_param_search(analysis_config, max_runs_override=None, dry_run=args.dry_run)
    elif mode == "neural_seed_search":
        run_neural_seed_search(analysis_config, seeds_override=None, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()
