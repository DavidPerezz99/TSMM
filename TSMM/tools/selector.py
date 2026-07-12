import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Find top-10 JSON files by R2 for each model type")
    p.add_argument("directory", help="Folder that contains the JSON files")
    p.add_argument("--model", help="Limit to a single model type (e.g. ulr, xgboost, …)")
    p.add_argument("--topk", type=int, default=10, help="How many top entries to print (default: 10)")
    return p.parse_args()

def read_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception as exc:
        print(f"[WARN] Skipping {path} – could not parse JSON ({exc})", file=sys.stderr)
        return None

def collect_scores(folder: Path, wanted_model: str | None) -> dict[str, list[tuple[float, Path, str]]]:
    """
    Returns: mapping model_name -> list of (R2, file_path, config_path)
    """
    scores: dict[str, list[tuple[float, Path, str]]] = defaultdict(list)

    for file in folder.rglob("*.json"):
        obj = read_json(file)
        if obj is None or "metric" not in obj:
            continue

        metrics_block = obj.get("metric") or {}
        config_path = obj.get("config_path", "<missing>")

        for model_name, metric_values in metrics_block.items():
            if wanted_model and model_name != wanted_model:
                continue
            r2 = metric_values.get("R2")
            if isinstance(r2, (float, int)):
                scores[model_name].append((r2, file, config_path))

    return scores

def print_topk(scores: dict[str, list[tuple[float, Path, str]]], k: int):
    if not scores:
        print("No matching data found.")
        return

    for model_name, lst in sorted(scores.items()):
        lst.sort(key=lambda t: t[0], reverse=True)   # highest R2 first
        print(f"\n=== {model_name} : top {min(k,len(lst))} ===")
        for rank, (r2, file_path, cfg) in enumerate(lst[:k], start=1):
            print(f"{rank:2d}. R2={r2:8.5f}   JSON={file_path.name}   cfg={cfg}")

def main():
    args = parse_args()
    folder = Path(args.directory)
    if not folder.is_dir():
        sys.exit(f"Error: {folder} is not a directory")

    scores = collect_scores(folder, args.model)
    print_topk(scores, args.topk)

if __name__ == "__main__":
    main()