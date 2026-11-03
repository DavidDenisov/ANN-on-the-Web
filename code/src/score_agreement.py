#!/usr/bin/env python3
"""Score saved browser top-1 results against matching exhaustive Faiss results."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def score(reference, result):
    expected = reference["exact_top1_ids"]
    sources = reference["source_ids"]
    count = reference["queries"]
    if (result["fixture_id"] != reference["fixture_id"]
            or result["noise"]["level"] != reference["noise_level"]
            or result["queries"] != count or result["k"] != 1 or result["repeats"] != 1
            or len(result["trials"]) != count or len(expected) != count or len(sources) != count
            or count <= 0):
        raise ValueError("Browser run and exact reference must describe the same top-1 queries")
    passages = reference["passages"]
    if any(type(value) is not int or not 0 <= value < passages for value in (*expected, *sources)):
        raise ValueError("Invalid passage ID in exact reference")
    seen = set()
    matches = recovered = 0
    for trial in result["trials"]:
        index = trial["query"]
        ids = trial["resultIds"]
        if (type(index) is not int or not 0 <= index < count or index in seen or trial["repeat"] != 0
                or trial["targetId"] != sources[index] or not isinstance(ids, list) or len(ids) > 1
                or any(type(value) is not int or not 0 <= value < passages for value in ids)):
            raise ValueError(f"Invalid or mismatched browser trial for query {index}")
        seen.add(index)
        winner = ids[0] if ids else None
        matches += winner == expected[index]
        recovered += winner == sources[index]
    return {
        "fixture_id": reference["fixture_id"], "noise_level": reference["noise_level"],
        "clusters": result["clusters"], "probes": result["probes"], "codec": result["codec"],
        "accuracy_only": result["accuracy_only"], "queries": count,
        "exact_top1_matches": matches, "top1_agreement": matches / count,
        "source_matches": recovered, "exact_source_matches": sum(a == b for a, b in zip(expected, sources)),
    }


def report_results(path):
    import ijson

    with path.open("rb") as stream:
        yield from ijson.items(stream, "measurements.item.result", use_float=True)


def load_references(paths):
    references = {}
    for path in paths:
        reference = json.loads(path.read_bytes())
        key = reference["fixture_id"]
        if key in references:
            raise ValueError(f"Duplicate exact reference for fixture {key}")
        references[key] = reference
    return references


def browser_references(references):
    fields = ("fixture_id", "noise_level", "passages", "queries", "exact_top1_ids")
    return {fixture: {key: reference[key] for key in fields} for fixture, reference in references.items()}


def write_scores(output, rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".top1-agreement-", dir=output.parent) as directory:
        staged = Path(directory) / "agreement.csv"
        with staged.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        output.hardlink_to(staged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact", type=Path, nargs="+", required=True, help="exhaustive Faiss result JSON files")
    parser.add_argument("--report", type=Path, nargs="+", default=[], help="browser runner results.json files")
    parser.add_argument("--result", type=Path, nargs="+", default=[], help="individual browser result JSON files")
    parser.add_argument("--output", type=Path, required=True, help="new CSV output path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    references = load_references(args.exact)
    rows = []
    skipped = 0
    for path in args.report:
        for result in report_results(path):
            reference = references.get(result["fixture_id"])
            if reference is None:
                skipped += 1
                continue
            rows.append({**score(reference, result), "origin": str(path)})
    for path in args.result:
        result = json.loads(path.read_bytes())
        reference = references.get(result["fixture_id"])
        if reference is None:
            parser.error(f"No matching exact reference for {path}")
        rows.append({**score(reference, result), "origin": str(path)})
    if not rows:
        parser.error("No browser results matched the exact references")
    write_scores(args.output, rows)
    print(f"Scored {len(rows)} runs. Skipped {skipped} runs with no matching exact reference.")


if __name__ == "__main__":
    main()
