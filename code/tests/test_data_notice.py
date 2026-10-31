"""Check that text exports and hosting packages retain Wikipedia attribution."""

import gzip
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from fixtures import write_json
import prepare_demo
import prepare_demo_pages


class DataNoticeChecks(unittest.TestCase):
    def test_gzip_export_keeps_text_and_adds_notice(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "passages.parquet"
            pq.write_table(pa.table({"title": ["Sunlight"], "text": ["Archived passage"]}), source)
            indexes = [(name, {"count": 1}) for name in
                       ("manifest.json", "c8192/manifest.json", "c32768/manifest.json")]
            with patch.object(prepare_demo, "load_indexes", return_value=indexes), \
                    patch.object(prepare_demo, "metadata_source", return_value=[
                        {"path": source, "offset": 0, "count": 1}]):
                prepare_demo.prepare_demo(root / "export")
            notice = (root / "export/DATA-NOTICE.txt").read_text()
            self.assertIn("Wikipedia contributors", notice)
            self.assertIn("https://creativecommons.org/licenses/by-sa/3.0/", notice)
            self.assertIn("https://en.wikipedia.org/?curid=WIKI_ID", notice)
            raw = gzip.decompress((root / "export/metadata/meta_0.bin.gz").read_bytes())
            self.assertEqual(raw, struct.pack("<4sIIII", b"WMD1", 0, 1, 8, 16)
                             + b"SunlightArchived passage")

    def test_pages_package_includes_notice_in_client_and_text_deployments(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            (bundle / "metadata").mkdir(parents=True)
            payload = b"unchanged compressed text"
            (bundle / "metadata/meta_0.bin.br").write_bytes(payload)
            for clusters, relative in [(16384, "manifest.json"), (8192, "c8192/manifest.json"),
                                       (32768, "c32768/manifest.json")]:
                target = bundle / relative
                write_json(target, {"count": 35167920, "dimensions": 384,
                    "clusters": clusters, "source": {}, "source_shards": [],
                    "metadata": {"format": "binary-http-v1", "count": 35167920, "batch_size": 500}})
            # Exercise packaging with one text file instead of the absent full corpus.
            with patch.object(prepare_demo_pages.math, "ceil", return_value=1):
                hosting = prepare_demo_pages.prepare_demo_pages(bundle, root / "pages", "notice-test")
            expected = (prepare_demo_pages.CODE / "client_wiki/DATA-NOTICE.txt").read_bytes()
            for relative in ["DATA-NOTICE.txt", "root/client_wiki/DATA-NOTICE.txt",
                             "batches/brotli-batch-0/DATA-NOTICE.txt"]:
                self.assertEqual((root / "pages" / relative).read_bytes(), expected)
            batch = root / "pages/batches/brotli-batch-0"
            self.assertEqual((batch / "meta_0.bin.br").read_bytes(), payload)
            self.assertEqual(hosting["batches"][0]["file_count"], len(list(batch.iterdir())))
            self.assertEqual(hosting["batches"][0]["bytes"], len(payload) + len(expected))
            self.assertIn('rel="license"', (batch / "_headers").read_text())
            for name in ["index.html", "benchmark.html"]:
                self.assertIn('href="DATA-NOTICE.txt" download',
                              (root / "pages/root/client_wiki" / name).read_text())


if __name__ == "__main__":
    unittest.main()
