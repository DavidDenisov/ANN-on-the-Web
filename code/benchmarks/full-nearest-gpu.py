#!/usr/bin/env python3
"""Find exact non-self neighbours for 1,000 global sources, one local shard at a time."""

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from cluster import copy_rows, prepare_vectors


def closest_nonself(index, queries, source_ids, passage_ids):
    scores, positions = index.search(queries, min(2, len(passage_ids)))
    valid = positions >= 0
    valid &= passage_ids[np.maximum(positions, 0)] != source_ids[:, None]
    scores[~valid] = -np.inf
    choice = np.argmax(scores, axis=1)
    rows = np.arange(len(queries))
    return scores[rows, choice], positions[rows, choice]


def read_shard(shard, dimensions):
    raw = np.load(shard["path"], mmap_mode="r", allow_pickle=False)
    if raw.dtype != np.dtype("<f2") or raw.shape != (shard["count"], dimensions):
        raise ValueError(f"Unexpected vector format or count in {shard['path']}")
    stored, values = prepare_vectors(raw)
    ids = np.arange(shard["offset"], shard["offset"] + len(values), dtype=np.int64)
    return ids, stored, values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new run directory; existing results are never overwritten")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if not hasattr(faiss, "StandardGpuResources") or faiss.get_num_gpus() < 1:
        raise RuntimeError("GPU-enabled Faiss and a visible CUDA GPU are required; no CPU fallback is used")

    manifest_path = CODE / "data/wikipedia/manifest.json"
    sample_path = CODE / "data/benchmark/geometry-ratio-32k/summary.json"
    suite_path = CODE / "data/benchmark/prepared/noise-large/suite.json"
    manifest = json.loads(manifest_path.read_text())
    sample = json.loads(sample_path.read_text())
    suite = json.loads(suite_path.read_text())
    source_ids = np.asarray(sample["source_ids"], dtype=np.int64)
    assert suite["count"] == 35167920
    assert manifest["dimensions"] == suite["dimensions"] == 384 and manifest["dtype"] == "<f2"
    assert manifest["revision"] == suite["source"]["revision"]
    assert manifest["dataset"] == suite["source"]["dataset"]
    assert sample["fixture_id"] == suite["id"]
    assert len(source_ids) == 1000 and np.all(np.diff(source_ids) > 0)
    assert source_ids[0] >= 0 and source_ids[-1] < suite["count"]
    shards = [{**s, "path": CODE / "data/wikipedia" / Path(s["name"]).stem / "vectors.npy"}
              for s in suite["source"]["shards"]]
    assert [s["name"] for s in shards] == manifest["shards"]
    offset = 0
    for shard in shards:
        assert shard["offset"] == offset and shard["count"] > 0 and shard["path"].is_file()
        offset += shard["count"]
    assert offset == suite["count"]
    raw = np.empty((len(source_ids), manifest["dimensions"]), dtype=np.float32)
    copy_rows(shards, source_ids, raw, "Reading 1,000 globally sampled sources")
    query_stored, queries = prepare_vectors(raw)

    faiss.omp_set_num_threads(4)
    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.device = 0
    config.useFloat16 = False
    index = faiss.GpuIndexFlatIP(resources, manifest["dimensions"], config)
    total_bytes = sum(shard["path"].stat().st_size for shard in shards)
    output.mkdir(parents=True)
    inputs = {"sources": {str(p.relative_to(CODE)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (manifest_path, sample_path, suite_path, Path(__file__).resolve(), CODE / "src/cluster.py")},
              "source_ids": source_ids.tolist(), "sampling": "Reuse the uniform global 1,000-source noise-check sample",
              "corpus_count": suite["count"], "shards": len(shards), "local_vector_bytes": total_bytes,
              "backend": "GpuIndexFlatIP", "float16_compute": False,
              "faiss_version": faiss.__version__, "gpu_device": 0}
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    best_scores = np.full(len(queries), -np.inf, dtype=np.float32)
    best_ids = np.full(len(queries), -1, dtype=np.int64)
    best_stored = np.zeros_like(query_stored)
    completed = scanned = 0
    started = last_report = time.monotonic()

    def progress(complete=False, error=None):
        elapsed = time.monotonic() - started
        eta = elapsed * (suite["count"] - scanned) / scanned if scanned else None
        state = {"complete": complete, "completed_shards": completed, "total_shards": len(shards),
                 "scanned_vectors": scanned, "total_vectors": suite["count"], "sources": len(queries),
                 "elapsed_seconds": elapsed, "eta_seconds": eta, "error": error}
        temporary = output / "progress.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(output / "progress.json")
        # These neighbours are provisional until every shard has been searched.
        with (output / "nearest-so-far.tmp").open("wb") as stream:
            np.savez(stream, source_ids=source_ids, neighbour_ids=best_ids,
                     scores=best_scores, source_vectors=query_stored, neighbour_vectors=best_stored)
        (output / "nearest-so-far.tmp").replace(output / "nearest-so-far.npz")
        eta_text = f"{eta / 60:.1f} min" if eta is not None else "pending"
        print(f"{completed:,}/{len(shards):,} local shards; {scanned:,}/{suite['count']:,} vectors "
              f"({100 * scanned / suite['count']:.1f}%); all {len(queries):,} sources compared; "
              f"elapsed {elapsed / 60:.1f} min; ETA {eta_text}", flush=True)

    print(f"GPU 0: all 1,000 sources per local shard; exhaustive scan of {suite['count']:,} vectors; "
          f"{total_bytes / 1e9:.2f} GB read locally; no downloads.", flush=True)
    progress()
    try:
        for shard in shards:
            assert shard["offset"] == scanned
            ids, stored, values = read_shard(shard, manifest["dimensions"])
            index.reset()
            index.add(values)
            scores, positions = closest_nonself(index, queries, source_ids, ids)
            improved = scores > best_scores
            best_scores[improved] = scores[improved]
            best_ids[improved] = ids[positions[improved]]
            best_stored[improved] = stored[positions[improved]]
            completed += 1
            scanned += len(ids)
            if time.monotonic() - last_report >= 10 or completed == len(shards):
                progress()
                last_report = time.monotonic()
        assert scanned == suite["count"] and np.isfinite(best_scores).all()
        assert np.all(best_ids >= 0) and np.all(best_ids != source_ids)
        left, right = query_stored.astype(np.float64), best_stored.astype(np.float64)
        left /= np.linalg.norm(left, axis=1, keepdims=True)
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        cosines = np.clip(np.sum(left * right, axis=1), -1, 1)
        distances = np.linalg.norm(left - right, axis=1)
        assert np.allclose(distances**2, 2 * (1 - cosines), atol=1e-12)
        noise, = [c for c in sample["cases"] if c["kind"] == "gaussian-l2" and c["alpha"] == .4]
        mean_distance, median_distance = float(distances.mean()), float(np.median(distances))
        summary = {"complete": True, "samples": len(queries), "corpus_count": scanned,
                   "shards_searched": completed, "self_matches_excluded": True,
                   "identical_vector_neighbours": int(np.count_nonzero(distances < 1e-12)),
                   "cosine_mean": float(cosines.mean()), "cosine_median": float(np.median(cosines)),
                   "l2_distance_mean": mean_distance, "l2_distance_median": median_distance,
                   "noise_mean_l2_ratio": noise["l2_distance_mean"] / mean_distance if mean_distance else None,
                   "noise_median_l2_ratio": noise["l2_distance_median"] / median_distance if median_distance else None,
                   "elapsed_seconds": time.monotonic() - started}
        with (output / "neighbours.csv").open("x", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["source_id", "neighbour_id", "cosine", "l2_distance"])
            writer.writerows(zip(source_ids.tolist(), best_ids.tolist(), cosines.tolist(), distances.tolist()))
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        progress(complete=True)
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException as error:
        progress(error=str(error))
        raise


if __name__ == "__main__":
    main()
