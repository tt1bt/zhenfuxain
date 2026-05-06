import argparse
import csv
import os
import re
from typing import Dict, List


def parse_one_log(path: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "mAP": float("nan"),
        "P@10": float("nan"),
        "R@10": float("nan"),
        "P@50": float("nan"),
        "R@50": float("nan"),
        "P@100": float("nan"),
        "R@100": float("nan"),
    }

    map_pat = re.compile(r"^mAP:\s*([0-9]*\.?[0-9]+)")
    pr_pat = re.compile(r"^P@(10|50|100):\s*([0-9]*\.?[0-9]+)\s+R@\1:\s*([0-9]*\.?[0-9]+)")

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            m1 = map_pat.match(line)
            if m1:
                metrics["mAP"] = float(m1.group(1))
                continue

            m2 = pr_pat.match(line)
            if m2:
                k = m2.group(1)
                metrics[f"P@{k}"] = float(m2.group(2))
                metrics[f"R@{k}"] = float(m2.group(3))

    return metrics


def infer_case_from_name(name: str) -> str:
    s = name.lower()
    if "1beta" in s or "1-beta" in s:
        return "class_balanced_1-beta"
    if "cb_1" in s or "cb1" in s:
        return "class_balanced_1"
    if "none" in s:
        return "none"
    return "unknown"


def collect_logs(log_dir: str) -> List[str]:
    files = []
    for n in os.listdir(log_dir):
        if n.lower().endswith(".log"):
            files.append(os.path.join(log_dir, n))
    files.sort()
    return files


def main():
    parser = argparse.ArgumentParser(description="Summarize CLRS test metrics from logs")
    parser.add_argument("--log_dir", type=str, default="logs_clrs_if0.05_bits32")
    parser.add_argument("--out_csv", type=str, default="summary_clrs_if0.05_bits32.csv")
    args = parser.parse_args()

    if not os.path.isdir(args.log_dir):
        raise FileNotFoundError(f"log_dir not found: {args.log_dir}")

    log_files = collect_logs(args.log_dir)
    if not log_files:
        raise RuntimeError(f"No .log files found in: {args.log_dir}")

    rows = []
    for p in log_files:
        base = os.path.basename(p)
        case = infer_case_from_name(base)
        metrics = parse_one_log(p)
        row = {
            "case": case,
            "log_file": base,
            "mAP": metrics["mAP"],
            "P@10": metrics["P@10"],
            "R@10": metrics["R@10"],
            "P@50": metrics["P@50"],
            "R@50": metrics["R@50"],
            "P@100": metrics["P@100"],
            "R@100": metrics["R@100"],
        }
        rows.append(row)

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "log_file", "mAP", "P@10", "R@10", "P@50", "R@50", "P@100", "R@100"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved summary to {args.out_csv}")


if __name__ == "__main__":
    main()
