"""Fixed corpus and experiment settings shared by preparation and result readers."""

from pathlib import Path

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "data" / "wikipedia"
DIMENSIONS = 384
CODECS = ("f16", "sq8", "sq6", "sq4", "int8", "pq4")
CLUSTER_COUNTS = (8192, 16384, 32768)
DEMO_INDEX_PATHS = ((16384, "manifest.json"), (8192, "c8192/manifest.json"),
                    (32768, "c32768/manifest.json"))
CLUSTER_PROBES = (1, 4, 8)
PROBE_CLUSTERS = (16384,)
PROBE_COUNTS = (1, 2, 3, 4, 6, 8, 10)
QUANTIZATION_CLUSTERS = 16384
QUANTIZATION_PROBES = 4
CORPUS_COUNT = 35_167_920
TRAINING_SAMPLE_SIZE = 2_097_152
QUERY_COUNT = 10_000
NOISE_LEVELS = (0.05, 0.15, 0.4)
NOISE_LABELS = ("small", "medium", "large")


def comparison_settings() -> dict[str, object]:
    return {"cluster_counts": list(CLUSTER_COUNTS), "cluster_probe_counts": list(CLUSTER_PROBES),
            "probe_cluster_counts": list(PROBE_CLUSTERS), "probe_counts": list(PROBE_COUNTS),
            "quantization_clusters": QUANTIZATION_CLUSTERS, "quantization_probes": QUANTIZATION_PROBES,
            "codecs": list(CODECS), "k": 1, "query_count": QUERY_COUNT, "repeats": 1}


def comparison_memberships() -> dict[str, set[tuple[int, int, str, bool]]]:
    return {
        "counts": {(c, p, "f16", False) for c in CLUSTER_COUNTS for p in CLUSTER_PROBES},
        "probes": {(c, p, "f16", False) for c in PROBE_CLUSTERS for p in PROBE_COUNTS},
        "codecs": {(QUANTIZATION_CLUSTERS, QUANTIZATION_PROBES, codec, True) for codec in CODECS},
    }
