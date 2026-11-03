#!/usr/bin/env python3
"""Measure source recovery@1 with an exhaustive Faiss GPU scan."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urljoin
from urllib.request import Request, urlopen

import faiss
import numpy as np
import torch
from faiss.contrib import torch_utils  # Enables CUDA tensor inputs to Faiss.
from tqdm import tqdm

from cluster import load_shards, require_gpu, stream_vectors
from paper_settings import CLUSTER_COUNTS, CORPUS_COUNT, DIMENSIONS, NOISE_LABELS, QUERY_COUNT
from run_comparisons import internet_url, load_prepared_data


CLUSTER_RECORD = np.dtype([("id", "<u4"), ("vector", "<f2", (DIMENSIONS,))])


def read_asset(root, path, hosting, *, download=False):
    path = path.resolve()
    key = path.relative_to(root).as_posix()
    expected = hosting["source_files"][key] if hosting is not None else None
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        if expected is None or not download:
            raise
        url = hosting["file_routes"].get(key, urljoin(hosting["asset_base_url"], key))
        url = internet_url(url, hosting["asset_base_urls"])
        request = Request(url, headers={"User-Agent": "WWW-exact-recovery/1.0"})
        with urlopen(request, timeout=60) as response:
            data = response.read(expected["bytes"] + 1)
    if expected is not None and (len(data) != expected["bytes"]
                                 or hashlib.sha256(data).hexdigest() != expected["sha256"]):
        raise ValueError(f"Asset differs from the prepared deployment: {key}")
    return data


def normalize(vectors, label):
    values = vectors.astype(np.float32, order="C")
    norms = np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError(f"Non-finite or zero vector in {label}")
    values /= norms
    return values


def cluster_batches(cluster_sizes, load_cluster):
    for cluster, size in enumerate(cluster_sizes):
        expected_bytes = size * CLUSTER_RECORD.itemsize
        with gzip.GzipFile(fileobj=io.BytesIO(load_cluster(cluster))) as stream:
            raw = stream.read(expected_bytes + 1)
        if len(raw) != expected_bytes:
            raise ValueError(f"Wrong record count in cluster {cluster}")
        records = np.frombuffer(raw, dtype=CLUSTER_RECORD)
        yield f"cluster {cluster}", records["id"], records["vector"]


def wikipedia_batches(directory, source):
    manifest, shards, _ = load_shards(directory)
    snapshot = [{key: shard[key] for key in ("name", "offset", "count")} for shard in shards]
    if (snapshot != source["shards"]
            or any(manifest.get(key) != source.get(key) for key in ("dataset", "revision"))):
        raise ValueError("Local Wikipedia shards do not match the query corpus")
    for offset, stored, _ in stream_vectors(shards):
        ids = np.arange(offset, offset + len(stored), dtype="<u4")
        yield f"Wikipedia batch at passage {offset}", ids, stored


def scan_top1(index, queries, passage_count, batches):
    seen = np.zeros(passage_count, dtype=bool)
    best_distances = np.full(len(queries), np.inf, dtype=np.float32)
    best_ids = np.full(len(queries), passage_count, dtype=np.int64)
    with tqdm(total=passage_count, desc="Exact L2 scan", unit="passage", unit_scale=True) as progress:
        for label, ids, stored in batches:
            size = len(ids)
            if size == 0:
                continue
            if (ids >= passage_count).any() or seen[ids].any() or len(np.unique(ids)) != size:
                raise ValueError(f"Invalid or repeated passage ID in {label}")
            seen[ids] = True
            index.reset()
            index.add(normalize(stored, label))
            distances, positions = index.search(queries, 1)
            distances = distances.cpu().numpy().ravel()
            positions = positions.cpu().numpy().ravel()
            if not np.isfinite(distances).all() or (positions < 0).any() or (positions >= size).any():
                raise ValueError(f"Invalid Faiss result in {label}")
            candidate_ids = ids[positions]
            better = (distances < best_distances) | ((distances == best_distances) & (candidate_ids < best_ids))
            best_distances[better] = distances[better]
            best_ids[better] = candidate_ids[better]
            progress.update(size)
    if not seen.all():
        raise ValueError("The scan did not cover the whole corpus")
    return best_ids, np.maximum(best_distances, 0)


def prepared_clusters(root, suite_dir, suite, clusters, hosting):
    descriptor = next(item for item in suite["indexes"] if item["clusters"] == clusters)
    manifest_path = (suite_dir / descriptor["path"]).resolve()
    manifest_bytes = read_asset(root, manifest_path, hosting)
    manifest = json.loads(manifest_bytes)
    sizes = manifest["cluster_sizes"]
    if (manifest["version"] != 1 or manifest["dimensions"] != DIMENSIONS
            or manifest["count"] != CORPUS_COUNT or manifest["clusters"] != clusters
            or manifest["metric"] != "normalized-l2" or manifest["record_bytes"] != CLUSTER_RECORD.itemsize
            or manifest["source_shards"] != suite["source"]["shards"]
            or any(manifest["source"].get(key) != suite["source"].get(key) for key in ("dataset", "revision"))
            or len(sizes) != clusters or any(type(size) is not int or size < 0 for size in sizes)
            or sum(sizes) != CORPUS_COUNT):
        raise ValueError("Cluster manifest does not match the query corpus")

    def load_cluster(cluster):
        path = manifest_path.parent / manifest["cluster_pattern"].format(id=cluster)
        return read_asset(root, path, hosting, download=True)

    layout = {"layout": "clusters", "clusters": clusters,
              "index_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest()}
    return cluster_batches(sizes, load_cluster), layout


def run(args):
    require_gpu()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required to keep the queries on the GPU")
    root = args.prepared_data.resolve()
    suite = load_prepared_data(root)[NOISE_LABELS.index(args.noise)]
    hosting = None
    if args.hosting_manifest is not None:
        hosting = json.loads(args.hosting_manifest.read_bytes())
        if hosting["source_fixture_ids"][args.noise] != suite["id"]:
            raise ValueError("Hosting manifest and query fixture do not match")

    suite_dir = (root / suite["suite_path"]).parent
    if args.wikipedia is None:
        batches, layout = prepared_clusters(root, suite_dir, suite, args.clusters or CLUSTER_COUNTS[0], hosting)
    else:
        batches = wikipedia_batches(args.wikipedia, suite["source"])
        layout = {"layout": "wikipedia-shards"}

    query_data = read_asset(root, suite_dir / suite["queries"]["file"], hosting, download=True)
    truth_data = read_asset(root, suite_dir / suite["truth"]["file"], hosting, download=True)
    if len(query_data) != QUERY_COUNT * DIMENSIONS * 4 or len(truth_data) != QUERY_COUNT * 4:
        raise ValueError("Unexpected query or source-ID file length")
    source_ids = np.frombuffer(truth_data, dtype="<u4")
    if (source_ids >= CORPUS_COUNT).any() or len(np.unique(source_ids)) != QUERY_COUNT:
        raise ValueError("Invalid or repeated query source ID")
    queries = np.frombuffer(query_data, dtype="<f4").reshape(QUERY_COUNT, DIMENSIONS)
    queries = torch.from_numpy(normalize(queries, "query fixture")).to("cuda:0")
    resources = faiss.StandardGpuResources()
    config = faiss.GpuIndexFlatConfig()
    config.useFloat16 = False
    index = faiss.GpuIndexFlatL2(resources, DIMENSIONS, config)

    nearest_ids, distances = scan_top1(index, queries, CORPUS_COUNT, batches)
    matches = int(np.count_nonzero(nearest_ids == source_ids))
    return {
        "fixture_id": suite["id"], "noise_level": suite["noise"]["level"],
        **layout, "passages": CORPUS_COUNT, "queries": QUERY_COUNT,
        "source_matches": matches, "source_recovery_at_1": matches / QUERY_COUNT,
        "source_ids": source_ids.tolist(), "exact_top1_ids": nearest_ids.tolist(),
        "squared_l2_distances": distances.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-data", type=Path, required=True)
    parser.add_argument("--hosting-manifest", type=Path, help="fetch missing binary assets from this deployment")
    parser.add_argument("--noise", choices=NOISE_LABELS, default="large")
    layout = parser.add_mutually_exclusive_group()
    layout.add_argument("--clusters", type=int, choices=CLUSTER_COUNTS, help="cluster layout (default: 8192)")
    layout.add_argument("--wikipedia", type=Path, help="scan the local Wikipedia shard directory")
    parser.add_argument("--output", type=Path, required=True, help="new JSON result file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".exact-recovery-", dir=args.output.parent) as temporary:
        staged = Path(temporary) / "result.json"
        staged.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        args.output.hardlink_to(staged)
    print(f"Exact source recovery@1: {report['source_matches']:,}/{QUERY_COUNT:,} "
          f"({report['source_recovery_at_1']:.2%})")


if __name__ == "__main__":
    main()
