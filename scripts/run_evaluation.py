from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metarag.evaluation import write_evaluation
from metarag.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-openai-answers", action="store_true")
    parser.add_argument("--include-openai-judge", action="store_true")
    args = parser.parse_args()

    evaluation_path = ROOT / "artifacts" / "evaluation.json"
    result = write_evaluation(
        evaluation_path,
        use_openai_answers=args.use_openai_answers,
        include_openai_judge=args.include_openai_judge,
    )
    generate_report(evaluation_path, ROOT / "docs")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
