from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg", force=True)
from streamlit.testing.v1 import AppTest


class DatasetStartupTests(unittest.TestCase):
    def test_missing_dataset_shows_upload_instructions_without_traceback(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        with tempfile.TemporaryDirectory() as temporary:
            missing_dataset = Path(temporary) / "missing_dataset.zip"
            app = AppTest.from_file(str(app_path), default_timeout=60).run()
            self.assertFalse(app.exception)

            next(
                item
                for item in app.text_input
                if item.label == "Dataset ZIP or extracted CSV directory"
            ).set_value(str(missing_dataset))
            app.run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "Dataset not found. Upload the project ZIP" in item.value
                for item in app.error
            )
        )
        self.assertTrue(
            any(
                item.label == "Upload project dataset ZIP"
                for item in app.file_uploader
            )
        )


if __name__ == "__main__":
    unittest.main()
