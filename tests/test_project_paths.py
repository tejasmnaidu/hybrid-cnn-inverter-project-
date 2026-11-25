from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from models.model_factory import create_model
from project_paths import persist_uploaded_dataset, resolve_project_path


class ProjectPathTests(unittest.TestCase):
    def test_relative_paths_are_resolved_from_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = resolve_project_path("data/raw/dataset.csv", root)
            self.assertEqual(resolved, (root / "data" / "raw" / "dataset.csv").resolve())

    def test_absolute_paths_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            absolute = (Path(temporary) / "checkpoint.pt").resolve()
            self.assertEqual(resolve_project_path(absolute, Path.cwd()), absolute)

    def test_uploaded_dataset_is_cached_by_content_outside_project(self) -> None:
        contents = b"test dataset archive"
        with tempfile.TemporaryDirectory() as temporary, patch(
            "project_paths.tempfile.gettempdir",
            return_value=temporary,
        ):
            first_path = persist_uploaded_dataset(contents)
            second_path = persist_uploaded_dataset(contents)

            self.assertEqual(first_path, second_path)
            self.assertEqual(first_path.suffix, ".zip")
            self.assertEqual(first_path.read_bytes(), contents)
            self.assertNotIn("Hybrid_CNN_Inverter_Optimization", str(first_path))

    def test_empty_uploaded_dataset_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            persist_uploaded_dataset(b"")

    def test_model_factory_resolves_relative_config_from_project_root(self) -> None:
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as temporary:
            try:
                os.chdir(temporary)
                model = create_model("configs/config.yaml")
            finally:
                os.chdir(original_directory)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
