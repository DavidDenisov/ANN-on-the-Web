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

from cloudflare import BATCH_FILES, HEADERS, deployment_digest, valid_project_name
from paper_settings import DEMO_INDEX_PATHS


CODE = Path(__file__).resolve().parents[1]
CLIENT_FILES = ("index.html", "benchmark.html", "DATA-NOTICE.txt", "assets/styles.css", "assets/search.mjs",
                "assets/embedding-worker.mjs", "assets/search-index.mjs", "assets/benchmark.mjs", "assets/retrieval.mjs", "assets/lab.mjs",
                "assets/fonts/inter-v20-latin-regular.woff2", "assets/fonts/inter-v20-latin-500.woff2",
                "assets/fonts/inter-v20-latin-600.woff2", "assets/fonts/inter-v20-latin-700.woff2",
                "assets/fonts/OFL.txt")
BROTLI_HEADERS = HEADERS.replace("Content-Length, CF-Cache-Status",
                                "Content-Encoding, Content-Length, CF-Cache-Status, Link") + """
/*.br
  Cache-Control: public, max-age=0, s-maxage=604800, no-transform
  Content-Type: application/octet-stream
  Content-Encoding: br
  Link: <https://creativecommons.org/licenses/by-sa/3.0/>; rel="license", </DATA-NOTICE.txt>; rel="describedby"
"""
CLIENT_HEADERS = BROTLI_HEADERS + """
/client_wiki/*
  Cross-Origin-Opener-Policy: same-origin
  Cross-Origin-Embedder-Policy: require-corp
"""


def copy_client(public: Path) -> None:
    public = Path(public)
    reference = json.loads((public / "data/search-full-brotli/manifest.json").read_text())
    metadata = reference["metadata"]
    if (metadata["format"] != "binary-http-v1" or metadata["batch_size"] != 500
            or metadata["count"] != reference["count"]):
        raise ValueError("Expected complete 500-passage Brotli metadata")
    client = public / "client_wiki"
    with TemporaryDirectory(prefix=".client-", dir=public) as temporary:
        staged = Path(temporary)
        for name in CLIENT_FILES:
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CODE / "client_wiki" / name, target)
        script = staged / "assets/search.mjs"
        source = script.read_text()
        prefix = "../data/demo/pages/root/data/search-full-brotli/"
        # shortcut: three fixed local manifests; use explicit page config if deployment paths become configurable.
        if source.count(prefix) != 3:
            raise ValueError("The search client's three manifest paths changed; update the package mapping")
        script.write_text(source.replace(prefix, "../data/search-full-brotli/"))
        for name in CLIENT_FILES:
            target = client / name
            target.parent.mkdir(parents=True, exist_ok=True)
            (staged / name).replace(target)
        headers = staged / "_headers"
        headers.write_text(CLIENT_HEADERS)
        headers.replace(public / "_headers")


def prepare_demo_pages(bundle, output, project_name) -> dict[str, object]:
    """Write the package to output and return its hosting descriptor, also saved as hosting.json."""
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
                "file_count": min(BATCH_FILES, total - part * BATCH_FILES) + 2,
                "bytes": (CODE / "client_wiki/DATA-NOTICE.txt").stat().st_size}
               for part in range(math.ceil(total / BATCH_FILES))]
    published_metadata = {**metadata, "pattern": "meta_{id}.bin.br",
                          "files_per_deployment": BATCH_FILES,
                          "base_urls": [batch["base_url"] for batch in batches]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".demo-pages-", dir=output.parent) as temporary:
        staged = Path(temporary) / "deployment"
        staged.mkdir()
        shutil.copy2(CODE / "client_wiki/DATA-NOTICE.txt", staged / "DATA-NOTICE.txt")
        public = staged / "root"
        for clusters, relative in DEMO_INDEX_PATHS:
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
        batch_directories = [staged / batch["path"] for batch in batches]
        for directory in batch_directories:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "_headers").write_text(BROTLI_HEADERS)
            shutil.copy2(CODE / "client_wiki/DATA-NOTICE.txt", directory / "DATA-NOTICE.txt")
        for number in range(total):
            descriptor = batches[number // BATCH_FILES]
            source = bundle / "metadata" / f"meta_{number}.bin.br"
            target = staged / descriptor["path"] / source.name
            os.link(source, target)
            descriptor["bytes"] += source.stat().st_size
        identity = hashlib.sha256()
        for directory in [public, *batch_directories]:
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=CODE / "data/demo/metadata")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    args = parser.parse_args()
    prepare_demo_pages(args.bundle, args.output, args.project_name)


if __name__ == "__main__":
    main()
