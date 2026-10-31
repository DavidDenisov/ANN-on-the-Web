"""Validate shared browser measurement records without loading plotting libraries."""

from collections.abc import Iterable
import ipaddress
import math
from typing import TypedDict

from paper_settings import CODECS


class CentroidTrial(TypedDict):
    setup_id: str
    sample: int
    requested_url: str
    bytes: int
    decodeMs: int | float
    downloadMs: int | float


class CentroidSamples(TypedDict):
    centroidTrials: list[CentroidTrial]
    centroid_setup_id: str
    centroid_setup_repeats: int


class QueryTiming(TypedDict, total=False):
    queryMs: int | float


class QueryTrial(QueryTiming):
    repeat: int
    query: int
    recall: int | float


class ValidatedResult(CentroidSamples):
    clusters: int
    probes: int
    k: int
    queries: int
    repeats: int
    codec: str
    metric: object
    accuracy_only: bool
    trials: list[QueryTrial]


def is_public_host(host: str) -> bool:
    """Check an IP address or normalized DNS hostname against the public-asset policy."""
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return ("." in host and not host.replace(".", "").isdigit()
                and not host.endswith((".localhost", ".local", ".internal", ".test", ".invalid")))


def positive_integer(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def finite_number(value: object, name: str, minimum=0, maximum=None, positive=False) -> int | float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        finite = False
    if (not finite
            or value < minimum or (positive and value <= 0)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'in range'}")
    return value


def validate_centroid_samples(row: dict) -> CentroidSamples:
    setup_id = row.get("centroid_setup_id")
    if not isinstance(setup_id, str) or not setup_id:
        raise ValueError("Missing shared centroid setup identity")
    repeats = positive_integer(row.get("centroid_setup_repeats"), "centroid_setup_repeats")
    trials = row.get("centroidTrials")
    if not isinstance(trials, list) or len(trials) != repeats:
        raise ValueError("centroidTrials must match the shared setup sample count")
    seen = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("Each centroid trial must be an object")
        if (trial.get("setup_id") != setup_id or not isinstance(trial.get("requested_url"), str)
                or not trial["requested_url"] or trial["requested_url"] != trials[0].get("requested_url")):
            raise ValueError("Shared centroid setup identity does not match its download")
        sample = trial.get("sample")
        if type(sample) is not int or not 0 <= sample < repeats or sample in seen:
            raise ValueError("Invalid or duplicate centroid setup sample")
        seen.add(sample)
        finite_number(trial.get("decodeMs"), "centroid decodeMs")
        finite_number(trial.get("downloadMs"), "centroid downloadMs", positive=True)
        positive_integer(trial.get("bytes"), "centroid bytes")
    if len({trial["bytes"] for trial in trials}) != 1:
        raise ValueError("Shared centroid samples must use the same payload size")
    return {"centroidTrials": trials, "centroid_setup_id": setup_id, "centroid_setup_repeats": repeats}


def validate_result(row: dict) -> ValidatedResult:
    if not isinstance(row, dict):
        raise ValueError("Each result must be an object")
    codec = row.get("codec")
    if not isinstance(codec, str) or codec not in CODECS:
        raise ValueError(f"Unsupported centroid codec: {codec!r}")
    values = {key: positive_integer(row.get(key), key)
              for key in ("clusters", "probes", "k", "queries", "repeats")}
    accuracy_only = row.get("accuracy_only")
    if type(accuracy_only) is not bool or values["repeats"] != 1:
        raise ValueError("Measurements require one pass")
    if row.get("cluster_fetch_cache") != ("force-cache" if accuracy_only else "no-store"):
        raise ValueError("Result cache policy does not match its test")
    if accuracy_only and any(key in row for key in ("queryMs", "routingMs", "clusterDownloadMs", "clusterDecodeMs", "scoringMs")):
        raise ValueError("Accuracy-only checks must omit query timing")
    if values["probes"] > values["clusters"]:
        raise ValueError("probes cannot exceed clusters")
    trials = row.get("trials")
    if not isinstance(trials, list) or len(trials) != values["queries"] * values["repeats"]:
        raise ValueError("trials must contain queries × repeats observations")
    seen = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("Each query trial must be an object")
        repeat, query = trial.get("repeat"), trial.get("query")
        if (type(repeat) is not int or not 0 <= repeat < values["repeats"]
                or type(query) is not int or not 0 <= query < values["queries"]
                or (repeat, query) in seen):
            raise ValueError("Invalid or duplicate query/repeat in trials")
        seen.add((repeat, query))
        finite_number(trial.get("recall"), "trial recall", maximum=1)
        if accuracy_only:
            if any(key in trial for key in ("queryMs", "routingMs", "downloadMs", "decodeMs", "scoringMs")):
                raise ValueError("Accuracy-only trials must omit query timing")
        else:
            finite_number(trial.get("queryMs"), "queryMs", positive=True)
    return {**values, "codec": codec, "metric": row.get("metric"), "accuracy_only": accuracy_only, "trials": trials,
            **validate_centroid_samples(row)}


def centroid_downloads(rows: Iterable[CentroidSamples]) -> list[CentroidTrial]:
    shared = {}
    for row in rows:
        for trial in row["centroidTrials"]:
            key = (row["centroid_setup_id"], trial["sample"])
            if key in shared and shared[key] != trial:
                raise ValueError("Conflicting measurements for a shared centroid setup")
            shared[key] = trial
    return list(shared.values())
