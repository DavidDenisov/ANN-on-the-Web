#!/usr/bin/env python3
"""Prepare full-corpus Wikipedia text while reusing the three uploaded vector indexes."""

import argparse
import gzip
import json
from pathlib import Path
import shutil
import struct
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import pyarrow.parquet as pq

from download_dataset import DEFAULT_OUTPUT as SOURCE, DIMENSIONS, validate_shard
from paper_settings import DEMO_INDEX_PATHS


CODE = Path(__file__).resolve().parents[1]
UPLOADED = CODE / "data/benchmark/pages/root"
BATCH_SIZE = 500
READ_BATCH_SIZE = 4096


def is_positive_integer(value):
    return type(value) is int and value > 0


def load_indexes() -> list[tuple[str, dict]]:
    indexes = []
    for clusters, relative in DEMO_INDEX_PATHS:
        index = json.loads((UPLOADED / f"c{clusters}/manifest.json").read_text())
        if (index.get("version") != 1 or index.get("metric") != "normalized-l2"
                or index.get("dimensions") != DIMENSIONS or index.get("record_bytes") != 4 + 2 * DIMENSIONS
                or index.get("count") != 35167920 or index.get("clusters") != clusters
                or not isinstance(index.get("cluster_sizes"), list) or len(index["cluster_sizes"]) != clusters
                or any(type(size) is not int or size < 0 for size in index["cluster_sizes"])
                or sum(index["cluster_sizes"]) != index["count"]
                or not isinstance(index.get("centroids"), dict) or "f16" not in index["centroids"]
                or not isinstance(index.get("cluster_files"), list) or len(index["cluster_files"]) != clusters):
            raise ValueError(f"Invalid full-corpus c{clusters} index")
        if indexes and any(index.get(key) != indexes[0][1].get(key) for key in
                           ("version", "count", "dimensions", "record_bytes", "metric", "source", "source_shards")):
            raise ValueError("Uploaded indexes must share the same corpus and source snapshot")
        for descriptor in [*index["centroids"].values(), *index["cluster_files"]]:
            if (not isinstance(descriptor, dict) or not isinstance(descriptor.get("file"), str)
                    or not is_positive_integer(descriptor.get("bytes"))):
                raise ValueError("Index assets need a URL and positive byte size")
            url = urlsplit(descriptor["file"])
            if url.scheme != "https" or not url.hostname or url.username or url.password:
                raise ValueError("Demo preparation requires uploaded HTTPS vector and centroid files")
        indexes.append((relative, index))
    return indexes


def metadata_source(index: dict) -> list[dict]:
    source = json.loads((SOURCE / "manifest.json").read_text())
    if source.get("dimensions") != DIMENSIONS or source.get("dtype") != "<f2":
        raise ValueError("Expected the downloaded float16 Wikipedia source")
    if any(not isinstance(source.get(key), str) or not source[key]
           or index.get("source", {}).get(key) != source[key] for key in ("dataset", "revision")):
        raise ValueError("Index and metadata must use the same dataset and revision")
    names, frozen = source.get("shards"), index.get("source_shards")
    if (not isinstance(names, list) or not names or any(not isinstance(name, str) for name in names)
            or len(set(names)) != len(names) or len({Path(name).stem for name in names}) != len(names)
            or any(Path(name).stem in ("", ".", "..") for name in names)
            or not isinstance(frozen, list) or any(not isinstance(shard, dict) for shard in frozen)
            or [shard.get("name") for shard in frozen] != names):
        raise ValueError("Full-corpus metadata requires the index's complete frozen shard order")
    shards, offset = [], 0
    for shard in frozen:
        if shard.get("offset") != offset or not is_positive_integer(shard.get("count")):
            raise ValueError("Source shard offsets or counts do not match")
        directory = SOURCE / Path(shard["name"]).stem
        if validate_shard(directory, Path(shard["name"]).name) != shard["count"]:
            raise ValueError(f"Source shard row count changed: {shard['name']}")
        shards.append({**shard, "path": directory / "passages.parquet"})
        offset += shard["count"]
    if offset != index["count"]:
        raise ValueError("Index rows do not match the complete metadata source")
    return shards


def write_batch(staged, records, first, count):
    payload = struct.pack("<4sII", b"WMD1", first, count) + records
    with ((staged / "metadata" / f"meta_{first // BATCH_SIZE}.bin.gz").open("wb") as output,
          gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0, compresslevel=1) as compressed):
        compressed.write(payload)


def write_metadata(shards, staged) -> int:
    pending, written = bytearray(), 0
    for shard in shards:
        offset = shard["offset"]
        with pq.ParquetFile(shard["path"]) as parquet:
            for batch in parquet.iter_batches(batch_size=READ_BATCH_SIZE, columns=["title", "text"]):
                for record in batch.to_pylist():
                    if not isinstance(record["title"], str) or not isinstance(record["text"], str):
                        raise ValueError("Passage title and text must be strings")
                    title, text = record["title"].encode("utf-8"), record["text"].encode("utf-8")
                    pending.extend(struct.pack("<II", len(title), len(text)))
                    pending.extend(title)
                    pending.extend(text)
                    written += 1
                    if written % BATCH_SIZE == 0:
                        write_batch(staged, pending, written - BATCH_SIZE, BATCH_SIZE)
                        pending.clear()
                offset += batch.num_rows
        if offset != shard["offset"] + shard["count"]:
            raise ValueError("Incomplete passage metadata shard")
        print(f"Exported metadata for {written:,} passages", flush=True)
    if pending:
        count = written % BATCH_SIZE
        write_batch(staged, pending, written - count, count)
    return written


def prepare_demo(output: Path) -> None:
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    indexes = load_indexes()
    count = indexes[0][1]["count"]
    shards = metadata_source(indexes[0][1])
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".demo-export-", dir=output.parent) as temporary:
        staged = Path(temporary) / "bundle"
        (staged / "metadata").mkdir(parents=True)
        shutil.copy2(CODE / "client_wiki/DATA-NOTICE.txt", staged / "DATA-NOTICE.txt")
        if write_metadata(shards, staged) != count:
            raise ValueError("Exported passage count does not match the indexes")
        for relative, index in indexes:
            index["metadata"] = {"format": "binary-gzip-v1", "batch_size": BATCH_SIZE,
                                 "pattern": ("" if relative == "manifest.json" else "../") + "metadata/meta_{id}.bin.gz",
                                 "count": count}
            index["embedding_model"] = "Xenova/all-MiniLM-L6-v2"
            target = staged / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(index, indent=2) + "\n")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        staged.rename(output)
    print(f"Saved all three full-corpus demo indexes: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new full-corpus gzip staging directory")
    prepare_demo(parser.parse_args().output)


if __name__ == "__main__":
    main()
