#!/usr/bin/env python3
"""Download Wikipedia shards and retain float16 vectors plus passage metadata."""

import argparse
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


DATASET = "maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2"
DIMENSIONS = 384
META_COLUMNS = ["id", "wiki_id", "paragraph_id", "title", "text"]
BATCH_SIZE = 4096
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "wikipedia"


def validate_shard(directory: Path, source_name: str) -> int:
    info = json.loads((directory / "info.json").read_text(encoding="utf-8"))
    rows = info["rows"]
    if info["source"] != source_name or type(rows) is not int or rows <= 0:
        raise ValueError(f"Invalid source or row count in {directory}")
    vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
    if vectors.dtype != np.dtype("<f2") or vectors.shape != (rows, DIMENSIONS):
        raise ValueError(f"Invalid float16 vector matrix in {directory}")
    del vectors
    with pq.ParquetFile(directory / "passages.parquet") as metadata:
        if metadata.metadata.num_rows != rows or metadata.schema_arrow.names != META_COLUMNS:
            raise ValueError(f"Passage rows do not match vectors in {directory}")
    return rows


def convert_shard(source: Path, destination: Path) -> int:
    """Publish a complete directory atomically; never delete or alter the source."""
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    with pq.ParquetFile(source) as parquet:
        missing = set(META_COLUMNS + ["emb"]) - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"{source.name} is missing columns: {sorted(missing)}")
        rows = parquet.metadata.num_rows
        if rows <= 0:
            raise ValueError(f"{source.name} contains no passages")
        schema = pa.schema([parquet.schema_arrow.field(name) for name in META_COLUMNS])
        with TemporaryDirectory(prefix=".convert-", dir=destination.parent) as temporary:
            converted = Path(temporary) / "converted"
            converted.mkdir()
            vectors = np.lib.format.open_memmap(
                converted / "vectors.npy", mode="w+", dtype="<f2", shape=(rows, DIMENSIONS)
            )
            try:
                offset = 0
                with pq.ParquetWriter(converted / "passages.parquet", schema, compression="zstd") as writer:
                    with tqdm(total=rows, desc="Converting to float16", unit="passage", leave=False) as progress:
                        for batch in parquet.iter_batches(batch_size=BATCH_SIZE, columns=META_COLUMNS + ["emb"]):
                            values = np.asarray(batch.column("emb").to_pylist(), dtype=np.float32)
                            if values.shape != (batch.num_rows, DIMENSIONS):
                                raise ValueError(f"Expected {DIMENSIONS}-dimensional vectors in {source.name}")
                            if not np.isfinite(values).all() or not values.any(axis=1).all():
                                raise ValueError(f"Non-finite or zero vector in {source.name}")
                            with np.errstate(over="ignore", invalid="ignore"):
                                half = values.astype("<f2")
                            if not np.isfinite(half).all() or not half.any(axis=1).all():
                                raise ValueError(f"Vector cannot be represented in float16 in {source.name}")
                            vectors[offset:offset + batch.num_rows] = half
                            writer.write_table(pa.Table.from_batches([batch]).select(META_COLUMNS))
                            offset += batch.num_rows
                            progress.update(batch.num_rows)
                if offset != rows:
                    raise ValueError(f"Incomplete conversion of {source.name}")
                vectors.flush()
            finally:
                del vectors
            (converted / "info.json").write_text(json.dumps({
                "source": source.name, "rows": rows,
                "dimensions": DIMENSIONS, "dtype": "<f2",
            }, indent=2) + "\n", encoding="utf-8")
            validate_shard(converted, source.name)
            converted.rename(destination)
    return rows


def prepare_dataset(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["dataset"] != DATASET or manifest["dimensions"] != DIMENSIONS or manifest["dtype"] != "<f2":
            raise ValueError("Output belongs to a different dataset or vector format")
    else:
        if any(output.iterdir()):
            raise ValueError("Choose an empty output directory for a new download")
        info = HfApi().dataset_info(DATASET)
        shards = sorted(file.rfilename for file in info.siblings
                        if file.rfilename.startswith("data/train-") and file.rfilename.endswith(".parquet"))
        if not shards:
            raise ValueError("No train Parquet shards found on Hugging Face")
        manifest = {
            "dataset": DATASET, "revision": info.sha, "dimensions": DIMENSIONS,
            "dtype": "<f2", "metadata_columns": META_COLUMNS, "shards": shards,
        }
        with TemporaryDirectory(prefix=".manifest-", dir=output) as temporary:
            staged = Path(temporary) / "manifest.json"
            staged.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            staged.rename(manifest_path)

    # shortcut: one writer per output directory; add locking if parallel runs are needed.
    total_rows = 0
    for filename in tqdm(manifest["shards"], desc="Wikipedia shards", unit="shard"):
        stem = Path(filename).stem
        destination = output / stem
        download_dir = output / ".downloads" / stem
        if destination.exists():
            rows = validate_shard(destination, Path(filename).name)
            tqdm.write(f"Already prepared: {stem} ({rows:,} passages)")
        else:
            source = Path(hf_hub_download(
                DATASET, filename, repo_type="dataset", revision=manifest["revision"],
                local_dir=download_dir,
            ))
            rows = convert_shard(source, destination)
            tqdm.write(f"Saved {stem}: {rows:,} passages")
        # Only this script's redundant download and its local cache metadata are removed.
        if download_dir.exists():
            shutil.rmtree(download_dir)
        total_rows += rows
    tqdm.write(f"Ready: {total_rows:,} passages in {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="output directory (default: code/data/wikipedia)")
    args = parser.parse_args()
    prepare_dataset(args.output)


if __name__ == "__main__":
    main()
