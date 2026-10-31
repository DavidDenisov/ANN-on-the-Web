#!/usr/bin/env python3
"""Build Wikipedia indexes and noisy-passage comparison datasets."""

from collections.abc import Callable, Iterator
import gzip
import hashlib
import json
import math
import random
import secrets
import shutil
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict

import faiss
import numpy as np
from tqdm import tqdm

from download_dataset import validate_shard
from paper_settings import CODECS, CLUSTER_COUNTS, CORPUS_COUNT, DIMENSIONS, NOISE_LABELS, NOISE_LEVELS, QUERY_COUNT, TRAINING_SAMPLE_SIZE


BATCH_SIZE = 65_536


class Shard(TypedDict):
    name: str
    offset: int
    count: int
    path: Path


def require_gpu() -> int:
    count = faiss.get_num_gpus() if hasattr(faiss, "StandardGpuResources") else 0
    if count < 1:
        raise RuntimeError("Clustering requires GPU-enabled Faiss and a visible CUDA GPU; "
                           "install faiss-gpu as described in code/docs/benchmarks.md")
    return count


def prepare_vectors(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return stored float16 vectors and their normalized float32 search values."""
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != DIMENSIONS or not np.isfinite(values).all():
        raise ValueError("Expected a finite vector matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Cannot normalize zero or overflowing vectors")
    stored = (values / norms).astype("<f2")
    decoded = stored.astype(np.float32)
    decoded /= np.linalg.norm(decoded, axis=1, keepdims=True)
    return stored, decoded


def load_shards(source: Path) -> tuple[dict, list[Shard], int]:
    """Freeze and validate the complete downloaded corpus before reading vectors."""
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    names = manifest.get("shards")
    if (manifest.get("dimensions") != DIMENSIONS or manifest.get("dtype") != "<f2"
            or not isinstance(names, list) or not names):
        raise ValueError("Expected the downloader's float16 Wikipedia manifest")
    stems = []
    for name in names:
        if not isinstance(name, str) or Path(name).stem in ("", ".", ".."):
            raise ValueError("Invalid source shard name")
        stems.append(Path(name).stem)
    if len(set(stems)) != len(stems):
        raise ValueError("Duplicate source shard names")
    published = [(name, source / stem) for name, stem in zip(names, stems)
                 if (source / stem).exists()]
    if len(published) != len(names):
        raise ValueError(f"Download incomplete: {len(published)}/{len(names)} shards are ready")
    shards, offset = [], 0
    for name, directory in tqdm(published, desc="Checking completed shards", unit="shard"):
        rows = validate_shard(directory, Path(name).name)
        shards.append({"name": name, "offset": offset, "count": rows,
                       "path": directory / "vectors.npy"})
        offset += rows
    if offset != CORPUS_COUNT:
        raise ValueError("The paper requires the complete 35,167,920-passage corpus")
    return manifest, shards, offset


def stream_vectors(shards: list[Shard]) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    for shard in shards:
        values = np.load(shard["path"], mmap_mode="r", allow_pickle=False)
        for start in range(0, shard["count"], BATCH_SIZE):
            stored, normalized = prepare_vectors(values[start:start + BATCH_SIZE])
            yield shard["offset"] + start, stored, normalized
        del values


def centroid_payload(centroids: np.ndarray, codec: str, seed: int) -> tuple[bytes, np.ndarray]:
    values = np.asarray(centroids, dtype="<f4")
    if (values.ndim != 2 or values.shape[1] != DIMENSIONS
            or len(values) not in CLUSTER_COUNTS or not np.isfinite(values).all()):
        raise ValueError("Expected finite centroids")
    if codec == "f16":
        encoded = values.astype("<f2")
        return encoded.tobytes(), encoded.astype(np.float32)
    if codec not in CODECS:
        raise ValueError(f"Unknown centroid codec: {codec}")
    if codec == "int8":
        # I-BERT Eq. (1)-(2), with the reference implementation's calibration floor.
        maximum = np.maximum(np.max(np.abs(values), axis=0), np.float32(1e-8))
        scale = (maximum / 127).astype("<f4")
        rounded = np.rint(np.clip(values, -maximum, maximum) / scale)
        codes = np.clip(rounded, -127, 127).astype(np.int8)
        return scale.tobytes() + codes.tobytes(), scale * codes.astype(np.float32)
    if codec == "pq4":
        block_size = 8
        blocks = DIMENSIONS // block_size
        codes = np.empty((len(values), blocks), dtype=np.uint8)
        decoded = np.empty_like(values)
        codebooks = []
        for block in range(blocks):
            start, stop = block * block_size, (block + 1) * block_size
            subspace = np.ascontiguousarray(values[:, start:stop])
            words = 16
            model = faiss.Kmeans(stop - start, words, niter=20, nredo=1, seed=seed,
                                 min_points_per_centroid=1,
                                 max_points_per_centroid=math.ceil(len(values) / words))
            model.train(subspace)
            codebook = np.asarray(model.centroids, dtype="<f4")
            _, labels = model.index.search(subspace, 1)
            codes[:, block] = labels[:, 0]
            decoded[:, start:stop] = codebook[labels[:, 0]]
            codebooks.append(codebook.tobytes())
        header = np.asarray([block_size], dtype="<u4").tobytes()
        return header + b"".join(codebooks) + pack_codes(codes, 4), decoded
    bits = int(codec[2:])
    levels = (1 << bits) - 1
    minimum = values.min(axis=0).astype("<f4")
    span = values.max(axis=0) - minimum
    scale = np.where(span == 0, 1, span / levels).astype("<f4")
    codes = np.clip(np.rint((values - minimum) / scale), 0, levels).astype(np.uint8)
    decoded = minimum + scale * codes.astype(np.float32)
    return minimum.tobytes() + scale.tobytes() + pack_codes(codes, bits), decoded


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    bit_values = (codes.reshape(-1, 1) >> np.arange(bits, dtype=np.uint8)) & 1
    return np.packbits(bit_values, bitorder="little").tobytes()


def write_gzip(path: Path, payload: bytes) -> None:
    with (path.open("wb") as output,
          gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed):
        compressed.write(payload)


def cluster_and_save(source: Path, output: Path, clusters: int, seed: int) -> dict[str, object]:
    """Write one of the paper's three full-corpus indexes."""
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if clusters not in CLUSTER_COUNTS or type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Invalid paper cluster count or recorded seed")
    gpu_count = require_gpu()
    input_manifest, shards, count = load_shards(source)
    dimensions = input_manifest["dimensions"]
    sample_size = TRAINING_SAMPLE_SIZE
    positions = np.asarray(sorted(random.Random(seed).sample(range(count), sample_size)), dtype=np.uint64)
    sample = np.empty((sample_size, dimensions), dtype=np.float32)
    copy_rows(shards, positions, sample, "Reading training sample")
    for start in range(0, sample_size, BATCH_SIZE):
        stop = min(start + BATCH_SIZE, sample_size)
        sample[start:stop] = prepare_vectors(sample[start:stop])[1]
    faiss.omp_set_num_threads(4)
    started = time.perf_counter()
    model = faiss.Kmeans(dimensions, clusters, niter=20, nredo=1, seed=seed, gpu=True,
                         max_points_per_centroid=math.ceil(sample_size / clusters))
    model.train(sample)
    training_seconds = time.perf_counter() - started
    centroids = np.asarray(model.centroids, dtype="<f4")
    del sample
    # Assign against the exact float16 centroids that the baseline browser downloads.
    assignment = model.index
    assignment.reset()
    assignment.add(centroids.astype("<f2").astype(np.float32))
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cluster-", dir=output.parent) as temporary:
        staged = Path(temporary) / "index"
        raw = Path(temporary) / "raw"
        for folder in (staged / "clusters", staged / "centroids", raw):
            folder.mkdir(parents=True)
        codec_manifest = {}
        for codec in CODECS:
            payload, decoded = centroid_payload(centroids, codec, seed=seed)
            path = staged / "centroids" / f"{codec}.bin.gz"
            write_gzip(path, payload)
            codec_manifest[codec] = {
                "file": f"centroids/{codec}.bin.gz", "bytes": path.stat().st_size,
                "rmse": float(np.sqrt(np.mean((decoded.astype(np.float64) - centroids) ** 2))),
            }
        record_type = np.dtype([("id", "<u4"), ("vector", "<f2", (dimensions,))])
        sizes = np.zeros(clusters, dtype=np.int64)
        with tqdm(total=count, desc="Assigning and saving", unit="passage", leave=False) as progress:
            for offset, stored, values in stream_vectors(shards):
                _, nearest = assignment.search(values, 1)
                labels = nearest[:, 0]
                records = np.empty(len(values), dtype=record_type)
                records["id"] = np.arange(offset, offset + len(values), dtype=np.uint64).astype("<u4")
                records["vector"] = stored
                counts = np.bincount(labels, minlength=clusters)
                records = records[np.argsort(labels, kind="stable")]
                ends = np.cumsum(counts)
                for cid in np.flatnonzero(counts):
                    selected = records[ends[cid] - counts[cid]:ends[cid]]
                    with (raw / f"{cid}.bin").open("ab") as target:
                        target.write(selected.tobytes())
                sizes += counts
                progress.update(len(values))
        for cid in tqdm(range(clusters), desc="Compressing clusters", unit="cluster", leave=False):
            raw_path = raw / f"{cid}.bin"
            target = staged / "clusters" / f"cluster_{cid}.bin.gz"
            with (target.open("wb") as output_file,
                  gzip.GzipFile(filename="", fileobj=output_file, mode="wb", mtime=0) as compressed):
                if raw_path.exists():
                    with raw_path.open("rb") as source_file:
                        shutil.copyfileobj(source_file, compressed)
            raw_path.unlink(missing_ok=True)
        manifest = {
            "version": 1, "dimensions": dimensions, "count": count, "clusters": clusters,
            "cluster_sizes": sizes.tolist(), "record_bytes": record_type.itemsize,
            "metric": "normalized-l2", "centroids": codec_manifest,
            "cluster_pattern": "clusters/cluster_{id}.bin.gz",
            "source_shards": [{key: shard[key] for key in ("name", "offset", "count")} for shard in shards],
            "training": {"seconds": training_seconds, "sample_size": sample_size,
                         "seed": seed, "iterations": 20, "threads": 4,
                         "backend": "faiss-gpu", "gpu_count": gpu_count},
            "source": {key: input_manifest[key] for key in ("dataset", "revision") if key in input_manifest},
        }
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # shortcut: one writer per destination; add a filesystem lock before parallel exports.
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        staged.rename(output)
    tqdm.write(f"Saved {count:,} vectors in {clusters:,} clusters: {output}")
    return manifest


def copy_rows(shards: list[Shard], positions: np.ndarray, target: np.ndarray, description: str) -> None:
    """Read sorted snapshot ordinals into an aligned, bounded-memory destination."""
    with tqdm(total=len(positions), desc=description, unit="passage", leave=False) as progress:
        for shard in shards:
            first, last = np.searchsorted(positions, [shard["offset"], shard["offset"] + shard["count"]])
            if first == last:
                continue
            values = np.load(shard["path"], mmap_mode="r", allow_pickle=False)
            for start in range(first, last, BATCH_SIZE):
                stop = min(start + BATCH_SIZE, last)
                rows = (positions[start:stop] - shard["offset"]).astype(np.intp)
                target[start:stop] = values[rows]
                progress.update(stop - start)
            del values


def noisy_queries(values: np.ndarray, noise: float, seed: int) -> np.ndarray:
    """Perturb normalized exported passage vectors by a fixed L2 noise magnitude."""
    if noise not in NOISE_LEVELS:
        raise ValueError("Use the paper noise levels: 0.05, 0.15 or 0.4")
    stored, _ = prepare_vectors(values)
    base = stored.astype(np.float64)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    direction = np.random.default_rng(seed).normal(size=base.shape)
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    perturbed = base + noise * direction
    norms = np.linalg.norm(perturbed, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Noise produced an invalid query vector")
    return (perturbed / norms).astype("<f4")


def build_comparison_data(source: Path, output: Path,
                          publish_index: Callable[[Path, int], None] | None = None) -> list[dict]:
    """Build the paper suites, releasing local cluster payloads after each successful publication.

    Without publish_index, retain every cluster payload for offline packaging.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if publish_index is not None and not callable(publish_index):
        raise ValueError("publish_index must be callable")
    seed = secrets.randbelow(2**31)
    noise_levels = tuple(zip(NOISE_LABELS, NOISE_LEVELS))
    require_gpu()
    original, shards, count = load_shards(source)
    query_noises = [{"kind": "gaussian-l2", "level": noise}
                    for _, noise in noise_levels]
    provenance = {
        "dataset": original.get("dataset"), "revision": original.get("revision"),
        "planned_shards": len(original["shards"]), "completed_shards": len(shards),
        "available_passages": count,
        "shards": [{key: shard[key] for key in ("name", "offset", "count")} for shard in shards],
    }
    tqdm.write(f"Snapshot: {len(shards)}/{len(original['shards'])} shards; "
               f"{count:,} indexed passages, {QUERY_COUNT} noisy passage queries "
               f"(noise={', '.join(f'{noise:g}' for _, noise in noise_levels)})")
    rng = random.Random(seed)
    query_ids = np.asarray(sorted(rng.sample(range(count), QUERY_COUNT)), dtype="<u4")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".real-benchmark-", dir=output.parent) as temporary:
        staged = Path(temporary) / "benchmark"
        staged.mkdir()
        identities = [hashlib.sha256(json.dumps({"source": provenance, "metric": "source-recovery", "noise": query_noise},
                                              sort_keys=True).encode("utf-8"))
                      for query_noise in query_noises]
        for start in range(0, count, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, count)
            row_bytes = np.arange(start, stop, dtype="<u8").tobytes()
            for identity in identities:
                identity.update(row_bytes)
        for identity in identities:
            identity.update(query_ids.tobytes())
        query_vectors = np.empty((QUERY_COUNT, DIMENSIONS), dtype="<f4")
        copy_rows(shards, query_ids, query_vectors, "Reading query source passages")
        query_ids.tofile(staged / "truth.u32.bin")
        indexes = [{"clusters": clusters, "path": f"c{clusters}/manifest.json"} for clusters in CLUSTER_COUNTS]
        suites = []
        for (name, noise), query_noise, identity in zip(noise_levels, query_noises, identities):
            directory = staged / f"noise-{name}"
            directory.mkdir(exist_ok=True)
            perturbed = noisy_queries(query_vectors, noise, seed)
            identity.update(perturbed.tobytes())
            perturbed.tofile(directory / "queries.f32.bin")
            suite = {
                "version": 1, "id": identity.hexdigest(), "synthetic": False,
                "label": f"Wikipedia passages · noisy passage recovery · {name} noise",
                "query_kind": "noisy-passages", "metric": "source-recovery",
                "noise": query_noise, "source": provenance,
                "dimensions": DIMENSIONS, "count": count, "query_count": QUERY_COUNT,
                "truth_k": 1, "seed": seed,
                "queries": {"file": "queries.f32.bin", "count": QUERY_COUNT},
                "truth": {"file": "../truth.u32.bin", "k": 1},
                "indexes": [{**index, "path": "../" + index["path"]} for index in indexes],
                "suite_path": f"noise-{name}/suite.json",
            }
            (directory / "suite.json").write_text(json.dumps(suite, indent=2) + "\n", encoding="utf-8")
            suites.append(suite)
        for clusters in tqdm(CLUSTER_COUNTS, desc="Cluster-count comparisons", unit="index"):
            cluster_and_save(source, staged / f"c{clusters}", clusters, seed)
            if publish_index is not None:
                publish_index(staged, clusters)
                shutil.rmtree(staged / f"c{clusters}" / "clusters")
        # shortcut: use one writer per destination, as with the existing exporters.
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        staged.rename(output)
    tqdm.write(f"Real-data comparisons ready: {output}")
    return suites
