#!/usr/bin/env python3
"""Package the search client and 500-passage Brotli files into 1,000-file Pages uploads."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from cloudflare import HEADERS, deployment_digest, valid_project_name


CODE = Path(__file__).resolve().parents[1]
BATCH_FILES = 1_000
CLIENT_FILES = ("index.html", "benchmark.html", "assets/styles.css", "assets/search.mjs",
                "assets/embedding-worker.mjs", "assets/search-index.mjs", "assets/benchmark.mjs", "assets/retrieval.mjs", "assets/lab.mjs",
                "assets/fonts/inter-v20-latin-regular.woff2", "assets/fonts/inter-v20-latin-500.woff2",
                "assets/fonts/inter-v20-latin-600.woff2", "assets/fonts/inter-v20-latin-700.woff2",
                "assets/fonts/OFL.txt")
BROTLI_HEADERS = HEADERS.replace("Content-Length, CF-Cache-Status",
                                "Content-Encoding, Content-Length, CF-Cache-Status") + """
/*.br
  Cache-Control: public, max-age=0, s-maxage=604800, no-transform
  Content-Type: application/octet-stream
  Content-Encoding: br
"""
CLIENT_HEADERS = BROTLI_HEADERS + """
/client_wiki/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
"""


def copy_client(public):
    public = Path(public)
    reference = json.loads((public / "data/search-full-brotli/manifest.json").read_text())
    metadata = reference["metadata"]
    if (metadata["format"] != "binary-http-v1" or metadata["batch_size"] != 500
            or metadata["count"] != reference["count"]):
        raise ValueError("Expected complete 500-passage Brotli metadata")
    client = public / "client_wiki"
    for name in CLIENT_FILES:
        target = client / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CODE / "client_wiki" / name, target)
    script = client / "assets/search.mjs"
    script.write_text(script.read_text().replace("../data/demo/pages/root/data/search-full-brotli/",
                                                "../data/search-full-brotli/"))
    (public / "_headers").write_text(CLIENT_HEADERS)


def prepare_demo_pages(bundle, output, project_name):
    project_name = valid_project_name(project_name)
    bundle, output = Path(bundle).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    reference = json.loads((bundle / "manifest.json").read_text())
    metadata = reference["metadata"]
    if ((metadata.get("format"), metadata.get("batch_size")) != ("binary-http-v1", 500)
            or reference.get("dimensions") != 384
            or reference.get("count") != 35167920
            or metadata["count"] != reference["count"]):
        raise ValueError("Expected complete 500-passage Brotli metadata for a 384-dimensional index")
    total = math.ceil(metadata["count"] / 500)
    expected = {f"meta_{number}.bin.br" for number in range(total)}
    if {path.name for path in (bundle / "metadata").iterdir()} != expected:
        raise ValueError("Missing or unexpected metadata files")
    base_url = f"https://{project_name}.pages.dev/"
    batches = [{"branch": f"brotli-batch-{part}", "path": f"batches/brotli-batch-{part}",
                "base_url": f"https://brotli-batch-{part}.{project_name}.pages.dev/",
                "file_count": min(BATCH_FILES, total - part * BATCH_FILES) + 1, "bytes": 0}
               for part in range(math.ceil(total / BATCH_FILES))]
    published_metadata = {**metadata, "pattern": "meta_{id}.bin.br",
                          "files_per_deployment": BATCH_FILES,
                          "base_urls": [batch["base_url"] for batch in batches]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".demo-pages-", dir=output.parent) as temporary:
        staged = Path(temporary) / "deployment"
        public = staged / "root"
        for clusters, relative in [(16384, "manifest.json"), (8192, "c8192/manifest.json"),
                                   (32768, "c32768/manifest.json")]:
            index = json.loads((bundle / relative).read_text())
            if (index["clusters"] != clusters
                    or any(index["metadata"][key] != metadata[key] for key in ("format", "batch_size", "count"))
                    or any(index[key] != reference[key] for key in
                           ("count", "dimensions", "source", "source_shards"))):
                raise ValueError("Search indexes do not share the same passage metadata")
            index["metadata"] = published_metadata
            target = public / "data/search-full-brotli" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(index, indent=2) + "\n")
        copy_client(public)
        (public / "_redirects").write_text("/ /client_wiki/ 302\n")
        for directory in [staged / batch["path"] for batch in batches]:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "_headers").write_text(BROTLI_HEADERS)
        for number in range(total):
            descriptor = batches[number // BATCH_FILES]
            source = bundle / "metadata" / f"meta_{number}.bin.br"
            target = staged / descriptor["path"] / source.name
            os.link(source, target)
            descriptor["bytes"] += source.stat().st_size
        identity = hashlib.sha256()
        for directory in [public] + [staged / batch["path"] for batch in batches]:
            print(f"Checking {directory.name}...", flush=True)
            identity.update(deployment_digest(directory).encode())
        hosting = {"version": 1, "kind": "search-demo", "project_name": project_name,
                   "branch_prefix": "brotli-", "bundle_id": identity.hexdigest(),
                   "asset_base_url": base_url, "batches": batches,
                   "root": {"branch": "main", "path": "root", "base_url": base_url}}
        text = json.dumps(hosting, indent=2) + "\n"
        (public / "routing.json").write_text(text)
        (staged / "hosting.json").write_text(text)
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        staged.rename(output)
    print(f"Ready: {output}; {len(batches)} text deployments plus the client. No upload performed.")
    return hosting


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=CODE / "data/demo/metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    args = parser.parse_args()
    prepare_demo_pages(args.bundle, args.output, args.project_name)


if __name__ == "__main__":
    main()
