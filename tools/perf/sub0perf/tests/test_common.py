"""Atomic no-clobber publication is required for durable evidence."""
import tempfile
import unittest
from pathlib import Path
from ..common import atomic_write_text


class PublicationTest(unittest.TestCase):
    def test_publish_refuses_existing_evidence_and_cleans_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            atomic_write_text(path, "original", overwrite=False)
            with self.assertRaises(FileExistsError):
                atomic_write_text(path, "replacement", overwrite=False)
            self.assertEqual(path.read_text(), "original")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_legacy_replace_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            atomic_write_text(path, "first")
            atomic_write_text(path, "second")
            self.assertEqual(path.read_text(), "second")

    def test_concurrent_publishers_have_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "record.json"
            def publish(text):
                try:
                    atomic_write_text(path, text, overwrite=False)
                    return True
                except FileExistsError:
                    return False
            with ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(publish, ("first", "second")))
            self.assertEqual(sum(results), 1)
            self.assertIn(path.read_text(), ("first", "second"))
            self.assertEqual(list(Path(directory).iterdir()), [path])
