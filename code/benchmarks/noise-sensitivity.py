#!/usr/bin/env python3
"""Check noise geometry and cluster coverage on 1,000 retained query sources."""

import argparse
import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cluster import copy_rows, noisy_queries, prepare_vectors


def analyze(clusters=16384) -> tuple[dict, list[dict]]:
    prepared = ROOT / "data/benchmark/prepared"
    suite = json.loads((prepared / "noise-large/suite.json").read_text())
    ids = np.fromfile(prepared / "truth.u32.bin", dtype="<u4")
    dimensions = suite["dimensions"]
    if not (len(ids) == suite["query_count"] == 10000 and np.all(np.diff(ids.astype(np.int64)) > 0)):
        raise ValueError('Expected 10,000 sorted distinct source IDs')
    shards = [{**s, "path": ROOT / "data/wikipedia" / Path(s["name"]).stem / "vectors.npy"}
              for s in suite["source"]["shards"]]
    raw = np.empty((len(ids), dimensions), dtype=np.float32)
    copy_rows(shards, ids, raw, "Reading retained query sources")
    stored, decoded = prepare_vectors(raw)
    base = stored.astype(np.float64)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    direction = np.random.default_rng(suite["seed"]).normal(size=base.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    # Verify that this analysis starts from the queries used in the paper.
    for label, alpha in (("small", .05), ("medium", .15), ("large", .4)):
        saved = np.fromfile(prepared / f"noise-{label}/queries.f32.bin", dtype="<f4").reshape(base.shape)
        generated = noisy_queries(raw, alpha, suite["seed"])
        if not np.array_equal(saved, generated):
            raise ValueError(f"Saved {label} queries do not match their source vectors")

    sampling_seed = 20261027
    rng = np.random.default_rng(sampling_seed)
    positions = np.sort(rng.choice(len(ids), 1000, replace=False))
    x, u = base[positions], direction[positions]
    partners = rng.integers(len(ids) - 1, size=len(positions))
    partners += partners >= positions
    pair_cosines = np.clip(np.sum(x * base[partners], axis=1), -1, 1)
    pair_distances = np.linalg.norm(x - base[partners], axis=1)
    if not np.all(ids[positions] != ids[partners]):
        raise ValueError('Sampled partner IDs must differ from source IDs')
    if not np.allclose(pair_distances**2, 2 * (1 - pair_cosines), atol=1e-12):
        raise ValueError('Random-pair distances do not match their cosines')
    random_pairs = {"samples": len(x),
                    "sampling": "Source and different donor passages from the uniform 10,000-passage corpus sample; IDs are source_ids and partner_ids.",
                    "cosine_mean": float(pair_cosines.mean()),
                    "cosine_median": float(np.median(pair_cosines)),
                    "l2_distance_mean": float(pair_distances.mean()),
                    "l2_distance_median": float(np.median(pair_distances))}
    tangent = base[partners] - np.sum(base[partners] * x, axis=1, keepdims=True) * x
    lengths = np.linalg.norm(tangent, axis=1, keepdims=True)
    if not np.all(lengths > 1e-8):
        raise ValueError('A donor is indistinguishable from its source')
    tangent /= lengths

    manifest_path = prepared / f"c{clusters}/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    centroid_path = manifest_path.parent / manifest["centroids"]["f16"]["file"]
    centroids = np.frombuffer(gzip.decompress(centroid_path.read_bytes()), dtype="<f2")
    centroids = centroids.astype(np.float32).reshape(manifest["clusters"], dimensions)
    faiss.omp_set_num_threads(4)
    index = faiss.IndexFlatL2(dimensions)
    index.add(centroids)
    # The exporter assigns passages against these same FP16 centroids.
    _, assigned = index.search(np.ascontiguousarray(decoded[positions]), 1)
    assigned = assigned[:, 0]
    _, clean = index.search(np.ascontiguousarray(x, dtype=np.float32), 1)
    if not np.array_equal(clean[:, 0], assigned):
        raise ValueError('Clean queries changed their assigned cluster')

    cases, samples = [], []
    for alpha in (0, .05, .15, .25, .4, .6, .8, 1.0):
        gaussian = x + alpha * u
        gaussian /= np.linalg.norm(gaussian, axis=1, keepdims=True)
        cosine = np.clip(np.sum(gaussian * x, axis=1), -1, 1)
        # Change direction while matching each Gaussian query's source angle exactly.
        corpus_directed = cosine[:, None] * x + np.sqrt(np.maximum(0, 1 - cosine**2))[:, None] * tangent
        for kind, query in (("gaussian-l2", gaussian), ("corpus-directed-matched-angle", corpus_directed)):
            if not np.allclose(np.linalg.norm(query, axis=1), 1, atol=1e-7):
                raise ValueError('Perturbed queries are not normalized')
            cosines = np.clip(np.sum(query * x, axis=1), -1, 1)
            if not np.allclose(cosines, cosine, atol=1e-7):
                raise ValueError('Perturbation does not preserve the matched source angle')
            angles = np.degrees(np.arccos(cosines))
            distances = np.linalg.norm(query - x, axis=1)
            _, selected = index.search(np.ascontiguousarray(query, dtype=np.float32), 8)
            hits = selected == assigned[:, None]
            row = {"kind": kind, "alpha": alpha, "samples": len(x),
                   "cosine_mean": float(cosines.mean()),
                   "cosine_median": float(np.median(cosines)),
                   "cosine_5th_percentile": float(np.percentile(cosines, 5)),
                   "cosine_95th_percentile": float(np.percentile(cosines, 95)),
                   "angle_mean_degrees": float(angles.mean()),
                   "angle_5th_percentile_degrees": float(np.percentile(angles, 5)),
                   "angle_95th_percentile_degrees": float(np.percentile(angles, 95)),
                   "l2_distance_mean": float(distances.mean()),
                   "l2_distance_median": float(np.median(distances))}
            row.update({f"source_cluster_coverage_{p}": float(hits[:, :p].any(axis=1).mean())
                        for p in (1, 4, 8)})
            cases.append(row)
            for i, passage_id in enumerate(ids[positions]):
                samples.append({"kind": kind, "alpha": alpha, "source_id": int(passage_id),
                                "cosine": float(cosines[i]), "angle_degrees": float(angles[i]),
                                "source_cluster": int(assigned[i]),
                                **{f"covered_{p}": bool(hits[i, :p].any()) for p in (1, 4, 8)}})

    with (ROOT / "data/benchmark/retrieval/summary.csv").open(newline="", encoding="utf-8") as stream:
        original = list(csv.DictReader(stream))
    original = [r for r in original if r["clusters"] == str(clusters) and r["codec"] == "f16"
                and "counts" in r["checks"].split(";") and r["probes"] in ("1", "4", "8")]
    if len(original) != 9:
        raise ValueError('Expected nine retained source recovery configurations')
    full_recovery = [{"alpha": float(r["noise_level"]), "probes": int(r["probes"]),
                      "queries": int(r["queries"]), "source_recovery": float(r["source_recovery"])}
                     for r in original]
    return {"samples": len(x), "sampling_seed": sampling_seed, "query_seed": suite["seed"],
            "source": {k: v for k, v in suite["source"].items() if k != "shards"},
            "fixture_id": suite["id"], "clusters": len(centroids), "centroid_codec": "f16",
            "centroid_sha256": hashlib.sha256(centroid_path.read_bytes()).hexdigest(),
            "saved_query_reproduction": "bit-identical for all 10,000 queries at all three original levels",
            "source_ids": ids[positions].tolist(), "partner_ids": ids[partners].tolist(),
            "random_passage_pairs": random_pairs,
            "metric": "Source-cluster coverage: the source's assigned cluster is among the probed clusters. "
                      "This is a routing diagnostic, not source recovery, exact-neighbor recall, or semantic relevance.",
            "corpus_direction": "Tangent toward a different retained corpus passage, with source cosine "
                                "matched per query to the original Gaussian perturbation. This is not a user-query model.",
            "cases": cases, "saved_full_run_source_recovery": full_recovery}, samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory")
    parser.add_argument("--clusters", type=int, choices=(16384, 32768), default=16384)
    args = parser.parse_args()
    output = args.output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    report, samples = analyze(args.clusters)
    output.mkdir(parents=True)
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    for name, rows in (("sensitivity.csv", report["cases"]), ("samples.csv", samples)):
        with (output / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    for row in report["cases"]:
        print(f"{row['kind']:30s} alpha={row['alpha']:<4g} cosine={row['cosine_mean']:.4f} "
              f"angle={row['angle_mean_degrees']:.2f}deg "
              + " ".join(f"coverage@{p}={100*row[f'source_cluster_coverage_{p}']:.1f}%" for p in (1, 4, 8)))
