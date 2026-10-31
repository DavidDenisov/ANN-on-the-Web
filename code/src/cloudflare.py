#!/usr/bin/env python3
"""Package and deploy comparison assets to Cloudflare Pages."""

import argparse
import copy
import hashlib
from itertools import groupby
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory

from paper_settings import NOISE_LABELS


BATCH_FILES = 1000
MAX_FILES = 20_000
MAX_FILE_BYTES = 25 * 1024 * 1024
HEADERS = """/*
  Access-Control-Allow-Origin: *
  Timing-Allow-Origin: *
  Access-Control-Expose-Headers: Content-Length, CF-Cache-Status

/*.gz
  Cache-Control: public, max-age=0, s-maxage=604800, no-transform
  Content-Type: application/octet-stream
  ! Content-Encoding

/*.bin
  Cache-Control: public, max-age=0, s-maxage=604800, no-transform
  Content-Type: application/octet-stream
  ! Content-Encoding
"""


def _file_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.digest()


def _link_or_copy(source: Path, target: Path) -> None:
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def comparison_assets(prepared_data, cluster_counts=None) -> dict[Path, Path]:
    """Follow only browser runtime dependencies; keep measurements offline."""
    root = Path(prepared_data).resolve(strict=True)
    files = {}
    selected = None if cluster_counts is None else set(cluster_counts)
    if selected is not None and (not selected or not selected.issubset((8192, 16384, 32768))):
        raise ValueError("Only the paper cluster counts can be packaged")

    def add(reference, parent=root) -> Path:
        if (not isinstance(reference, str) or not reference or Path(reference).is_absolute()
                or any(value in reference for value in ("\\", ":", "?", "#"))):
            raise ValueError("Assets must use relative local paths")
        path = (parent / reference).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Asset leaves the prepared directory or is not a file: {reference}")
        if path != Path(os.path.abspath(parent / reference)):
            raise ValueError(f"Asset symlinks are not supported: {reference}")
        files[path.relative_to(root)] = path
        return path

    def read_json(path) -> dict[str, object]:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Expected an object in {path}")
        return value

    try:
        for label in NOISE_LABELS:
            suite_path = add(f"noise-{label}/suite.json")
            suite = read_json(suite_path)
            add(suite["queries"]["file"], suite_path.parent)
            add(suite["truth"]["file"], suite_path.parent)
            indexes = suite["indexes"]
            if (suite.get("count") != 35167920 or suite.get("dimensions") != 384
                    or suite.get("query_count") != 10000 or suite.get("truth_k") != 1
                    or suite.get("metric") != "source-recovery"
                    or not isinstance(indexes, list)
                    or sorted(index["clusters"] for index in indexes) != [8192, 16384, 32768]):
                raise ValueError("Package the full paper corpus, query suites and indexes")
            if selected is not None:
                advertised = {index["clusters"] for index in indexes}
                if not selected.issubset(advertised):
                    raise ValueError(f"Requested cluster counts are missing from the {label} suite")
                indexes = [index for index in indexes if index["clusters"] in selected]
            for index in indexes:
                manifest_path = add(index["path"], suite_path.parent)
                manifest = read_json(manifest_path)
                clusters = manifest["clusters"]
                if type(clusters) is not int or clusters < 1:
                    raise ValueError("Invalid cluster count")
                if "clusters" in index and index["clusters"] != clusters:
                    raise ValueError("Suite and index cluster counts do not match")
                centroids = manifest["centroids"]
                if not isinstance(centroids, dict) or not centroids:
                    raise ValueError("The indexes need centroid files")
                for centroid in centroids.values():
                    add(centroid["file"], manifest_path.parent)
                pattern = manifest["cluster_pattern"]
                if pattern != "clusters/cluster_{id}.bin.gz":
                    raise ValueError("Unsupported cluster filename pattern")
                for number in range(clusters):
                    add(pattern.format(id=number), manifest_path.parent)
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete prepared comparison data") from error
    return files


def valid_project_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,56}[a-z0-9])?", value):
        raise ValueError("Use a Pages project name of 1–58 lowercase letters, digits or internal hyphens")
    return value


def ensure_pages_project(project) -> None:
    valid_project_name(project)
    try:
        subprocess.run(["wrangler", "whoami"], check=True, timeout=60)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Install Wrangler and run 'wrangler login' once before uploading") from error
    projects = subprocess.run(["wrangler", "pages", "project", "list"], check=True, capture_output=True, text=True, timeout=60)
    table = re.sub(r"\x1b\[[0-9;]*m", "", projects.stdout)
    if not re.search(r"(?m)^\s*[│|]\s*" + re.escape(project) + r"\s*[│|]", table):
        subprocess.run(["wrangler", "pages", "project", "create", project,
                        "--production-branch", "main"], check=True, timeout=300)


def valid_branch_prefix(value):
    if not isinstance(value, str) or (value and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,46}-", value)):
        raise ValueError("Use an empty branch prefix or up to 48 lowercase letters, digits and hyphens ending in '-'")
    return value


def prepare_cloudflare_pages(prepared_data: Path, output: Path, project_name: str,
                             cluster_counts: list[int] | None = None, branch_prefix: str = "") -> Path:
    """Give each upload batch its own Pages branch and publish one root routing map."""
    project_name = valid_project_name(project_name)
    branch_prefix = valid_branch_prefix(branch_prefix)
    source_root, output = Path(prepared_data).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new deployment directory")
    files = comparison_assets(source_root, cluster_counts=cluster_counts)
    root_url = f"https://{project_name}.pages.dev/"
    binary_paths = sorted(relative for relative in files if relative.suffix == ".gz")
    root_paths = set(files) - set(binary_paths)
    if len(root_paths) + 2 > MAX_FILES:
        raise ValueError(f"Free Cloudflare Pages allows {MAX_FILES:,} files per root deployment")
    fingerprints = {}
    from tqdm import tqdm

    for relative, source in tqdm(sorted(files.items()), desc="Checking hosting assets", unit="file"):
        size = source.stat().st_size
        if size > MAX_FILE_BYTES:
            raise ValueError(f"{relative} is {size / (1024 * 1024):.2f} MiB, above Cloudflare Pages' "
                             "25 MiB per-file limit; clusters are not split. Use a host supporting larger files "
                             "or prepare a clustering whose files fit the limit.")
        fingerprints[relative.as_posix()] = {"bytes": size, "sha256": _file_digest(source).hex()}
    bundle_id = hashlib.sha256(json.dumps({"schema": 7, "batch_files": BATCH_FILES,
                                          "headers": HEADERS, "files": fingerprints},
                                         sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    batches, routes, locations = [], {}, {}
    for _, group in groupby(binary_paths, key=lambda path: path.parts[0]):
        group_paths = list(group)
        for start in range(0, len(group_paths), BATCH_FILES):
            branch = f"{branch_prefix}batch-{len(batches)}"
            batch_paths = group_paths[start:start + BATCH_FILES]
            if len({path.name for path in batch_paths}) != len(batch_paths):
                raise ValueError("Duplicate filenames in a flat Pages batch")
            base = f"https://{branch}.{project_name}.pages.dev/"
            batches.append({"branch": branch, "path": f"batches/{branch}", "base_url": base,
                            "file_count": len(batch_paths) + 1,
                            "bytes": sum(fingerprints[path.as_posix()]["bytes"] for path in batch_paths)})
            for relative in batch_paths:
                routes[relative.as_posix()] = base + relative.name
                locations[relative] = Path(batches[-1]["path"]) / relative.name
    suites = {label: json.loads(files[Path(f"noise-{label}/suite.json")].read_text())
              for label in NOISE_LABELS}
    manifests = {relative: json.loads(files[relative].read_text()) for relative in root_paths
                 if relative.name == "manifest.json"}
    first = suites["small"]
    hosting = {
        "version": 1, "bundle_id": bundle_id, "project_name": project_name, "branch_prefix": branch_prefix,
        "asset_base_url": root_url,
        "asset_base_urls": [root_url] + [batch["base_url"] for batch in batches],
        "source_fixture_ids": {label: suite["id"] for label, suite in suites.items()},
        "data": {key: first.get(key) for key in ("count", "dimensions", "query_count", "source")},
        "noise_levels": {label: suite.get("noise") for label, suite in suites.items()},
        "indexes": [{"clusters": manifest["clusters"], "codecs": sorted(manifest["centroids"]),
                     "training": manifest.get("training")}
                    for manifest in sorted(manifests.values(), key=lambda value: value["clusters"])],
        "file_routes": routes, "source_files": fingerprints, "batches": batches,
        "root": {"branch": "main", "path": "root", "base_url": root_url},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".cloudflare-pages-", dir=output.parent) as temporary:
        staged = Path(temporary) / "deployment"
        public = staged / "root"
        public.mkdir(parents=True)
        for relative in sorted(root_paths):
            source, target = files[relative], public / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if relative in manifests:
                manifest = manifests[relative]
                for centroid in manifest["centroids"].values():
                    local = (source.parent / centroid["file"]).resolve().relative_to(source_root)
                    centroid["file"] = routes[local.as_posix()]
                manifest["cluster_files"] = []
                for number in range(manifest["clusters"]):
                    local = (relative.parent / manifest["cluster_pattern"].format(id=number)).as_posix()
                    manifest["cluster_files"].append({"file": routes[local], "bytes": fingerprints[local]["bytes"]})
                target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
            else:
                _link_or_copy(source, target)
        for relative in tqdm(binary_paths, desc="Packaging Pages batches", unit="file"):
            target = staged / locations[relative]
            target.parent.mkdir(parents=True, exist_ok=True)
            _link_or_copy(files[relative], target)
        for directory in [public] + [staged / batch["path"] for batch in batches]:
            (directory / "_headers").write_text(HEADERS, encoding="utf-8")
        text = json.dumps(hosting, indent=2) + "\n"
        (public / "routing.json").write_text(text, encoding="utf-8")
        (staged / "hosting.json").write_text(text, encoding="utf-8")
        for path in staged.rglob("*"):
            if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError(f"{path.relative_to(staged)} exceeds Cloudflare Pages' 25 MiB per-file limit")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        staged.rename(output)
    print(f"Pages package ready: {output} ({len(batches):,} binary batches plus root; bundle {bundle_id[:12]})")
    print(f"Deploy with: python3 code/src/cloudflare.py deploy --deployment {output}")
    return output


def deployment_digest(directory) -> str:
    """Content hashes let an interrupted upload skip unchanged completed branches."""
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files or len(files) > MAX_FILES:
        raise ValueError("A Pages deployment must contain between 1 and 20,000 files")
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink() or path.resolve() != path.absolute() or path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"Unsafe path or file above Pages' 25 MiB limit: {path}")
        digest.update(path.relative_to(directory).as_posix().encode("utf-8") + b"\0")
        digest.update(_file_digest(path))
    return digest.hexdigest()


def deploy_cloudflare_pages(deployment, force=False, publish_root=True) -> dict[str, object]:
    deployment = Path(deployment).resolve(strict=True)
    hosting = json.loads((deployment / "hosting.json").read_text(encoding="utf-8"))
    project = valid_project_name(hosting.get("project_name"))
    branch_prefix = valid_branch_prefix(hosting.get("branch_prefix", ""))
    batches, root = hosting.get("batches"), hosting.get("root")
    if (hosting.get("version") != 1 or not isinstance(batches, list) or not batches
            or not isinstance(root, dict) or root.get("branch") != "main"
            or not isinstance(hosting.get("bundle_id"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", hosting["bundle_id"])
            or hosting.get("asset_base_url") != f"https://{project}.pages.dev/"):
        raise ValueError("Invalid Pages hosting manifest")
    uploads = []
    for number, descriptor in enumerate(batches + [root]):
        expected = f"{branch_prefix}batch-{number}" if number < len(batches) else "main"
        if not isinstance(descriptor, dict) or descriptor.get("branch") != expected:
            raise ValueError("Invalid Pages batch order")
        relative = descriptor.get("path")
        expected_path = f"batches/{expected}" if expected != "main" else "root"
        if relative != expected_path:
            raise ValueError("Invalid Pages upload path")
        expected_base = (f"https://{expected}.{project}.pages.dev/"
                         if expected != "main" else hosting["asset_base_url"])
        if descriptor.get("base_url") != expected_base:
            raise ValueError("Invalid Pages branch asset URL")
        directory = (deployment / relative).resolve(strict=True)
        if not directory.is_relative_to(deployment) or not directory.is_dir():
            raise ValueError("Upload path leaves the deployment directory")
        uploads.append((descriptor, directory))
    if json.loads((deployment / "root/routing.json").read_text(encoding="utf-8")) != hosting:
        raise ValueError("Local Pages routing does not match hosting.json")
    receipt_path = deployment / "deployment.json"
    receipt = (json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.exists()
               else {"version": 1, "project_name": project, "completed": {}})
    if (receipt.get("version") != 1 or receipt.get("project_name") != project
            or not isinstance(receipt.get("completed"), dict)):
        raise ValueError("Invalid Pages deployment progress file")
    for descriptor, directory in uploads:
        branch = descriptor["branch"]
        if branch == "main" and not publish_root:
            continue
        print(f"Checking {branch} upload contents...", flush=True)
        digest = deployment_digest(directory)
        previous = receipt["completed"].get(branch)
        if not force and isinstance(previous, dict) and previous.get("sha256") == digest:
            print(f"Already uploaded: {branch}", flush=True)
            continue
        print(f"Uploading {branch} to Pages project {project}...", flush=True)
        subprocess.run(["wrangler", "pages", "deploy", str(directory), "--project-name", project,
                        "--branch", branch], check=True, timeout=1800)
        receipt["completed"][branch] = {"sha256": digest, "base_url": descriptor["base_url"]}
        temporary = deployment / ".deployment.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)
    print(f"Pages {'comparison assets' if publish_root else 'binary batches'} deployed: {hosting['asset_base_url']}")
    return receipt


class StreamingPagesPublisher:
    def __init__(self, output: Path, project: str) -> None:
        self.output = Path(output).resolve()
        self.project = valid_project_name(project)
        self.hosting = None
        self.output.mkdir(parents=True, exist_ok=False)

    def __call__(self, data: Path, clusters: int) -> None:
        data = Path(data)
        if self.hosting and str(clusters) in self.hosting["stage_bundle_ids"]:
            raise ValueError(f"Clustering c{clusters} was already uploaded")
        stage = prepare_cloudflare_pages(data, self.output / "stages" / f"c{clusters}", self.project,
                                         cluster_counts=[clusters],
                                         branch_prefix=f"c{clusters}-")
        deploy_cloudflare_pages(stage, publish_root=False)
        incoming = json.loads((stage / "hosting.json").read_text(encoding="utf-8"))
        merged = copy.deepcopy(self.hosting or incoming)
        if self.hosting is None:
            merged.update(publication="streamed-cluster-counts", stage_bundle_ids={},
                          indexes=[], batches=[], file_routes={}, source_files={},
                          asset_base_urls=[incoming["asset_base_url"]])
            merged.pop("branch_prefix", None)
        for key in ("project_name", "asset_base_url", "source_fixture_ids", "data", "noise_levels"):
            if merged[key] != incoming[key]:
                raise ValueError(f"Clustering upload changed the experiment's {key}")
        for key in ("source_files", "file_routes"):
            for name, value in incoming[key].items():
                if name in merged[key] and merged[key][name] != value:
                    raise ValueError(f"Conflicting uploaded asset: {name}")
                merged[key][name] = value
        merged["stage_bundle_ids"][str(clusters)] = incoming["bundle_id"]
        merged["indexes"].extend(incoming["indexes"])
        merged["batches"].extend({**batch, "path": f"stages/c{clusters}/{batch['path']}"}
                                 for batch in incoming["batches"])
        merged["asset_base_urls"].extend(incoming["asset_base_urls"][1:])
        merged["bundle_id"] = hashlib.sha256(json.dumps(merged["stage_bundle_ids"], sort_keys=True).encode()).hexdigest()
        root = self.output / "root"
        for source in (stage / "root").rglob("*"):
            if not source.is_file() or source.name == "routing.json":
                continue
            target = root / source.relative_to(stage / "root")
            if target.exists() and source.read_bytes() != target.read_bytes():
                raise ValueError(f"Conflicting root asset: {target.relative_to(root)}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        text = json.dumps(merged, indent=2) + "\n"
        routing = root / ".routing.json.tmp"
        routing.write_text(text, encoding="utf-8")
        routing.replace(root / "routing.json")
        checkpoint = self.output / ".hosting.json.tmp"
        checkpoint.write_text(text, encoding="utf-8")
        checkpoint.replace(self.output / "hosting.json")
        self.hosting = merged
        # Keep upload receipts and manifests outside the builder's temporary directory.
        shutil.rmtree(stage / "batches")
        print(f"c{clusters} uploaded; upload staging payloads cleared.", flush=True)

    def finish(self) -> Path:
        if self.hosting is None:
            raise ValueError("No cluster exports were uploaded")
        root = self.output / "root"
        for label in NOISE_LABELS:
            suite = json.loads((root / f"noise-{label}/suite.json").read_text(encoding="utf-8"))
            if {item["clusters"] for item in suite["indexes"]} != {item["clusters"] for item in self.hosting["indexes"]}:
                raise ValueError("Cannot publish before every advertised clustering is uploaded")
        digest = deployment_digest(root)
        subprocess.run(["wrangler", "pages", "deploy", str(root), "--project-name", self.project,
                        "--branch", "main"], check=True, timeout=1800)
        receipt = {"version": 1, "project_name": self.project,
                   "completed": {"main": {"sha256": digest, "base_url": self.hosting["asset_base_url"]}}}
        temporary = self.output / ".deployment.json.tmp"
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output / "deployment.json")
        return self.output / "hosting.json"

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="package prepared comparison assets")
    prepare.add_argument("--prepared-data", required=True, type=Path, help="the prepared run's data directory")
    prepare.add_argument("--output", required=True, type=Path, help="new deployment directory")
    prepare.add_argument("--project-name", required=True, help="one Cloudflare Pages project for root and batch branches")
    deploy = commands.add_parser("deploy", help="upload comparison batches and their root map")
    deploy.add_argument("--deployment", required=True, type=Path, help="directory made by cloudflare.py prepare")
    deploy.add_argument("--force", action="store_true", help="upload every branch again if another package replaced this project")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_cloudflare_pages(args.prepared_data, args.output, args.project_name)
    else:
        deploy_cloudflare_pages(args.deployment, force=args.force)


if __name__ == "__main__":
    main()
