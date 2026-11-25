from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import matplotlib
import torch

matplotlib.use("Agg", force=True)
from streamlit.testing.v1 import AppTest

from evaluation.live_simulation import run_live_simulation
from models.checkpoint_loader import load_checkpoint_for_inference


class DatasetStartupTests(unittest.TestCase):
    def test_checkpoint_and_live_inference_load_without_dataset(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        checkpoint_path = (
            project_root
            / "colab_runs"
            / "baseline_20261120_080836"
            / "checkpoints"
            / "best.pt"
        )
        self.assertTrue(checkpoint_path.is_file())

        model, normalization, metadata, checkpoint = load_checkpoint_for_inference(
            checkpoint_path,
            torch.device("cpu"),
        )
        self.assertTrue(checkpoint)
        self.assertEqual(metadata["operating_feature_names"][0], "modulationIndex")
        self.assertEqual(model.parameter_count(), 201601)

        result = run_live_simulation(
            model,
            normalization,
            modulation_index=0.75,
            dc_voltage_per_cell_v=100.0,
            load_resistance_ohm=30.0,
            load_inductance_h=0.03,
            phase_angle_degrees=0.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(result.target_reference_voltage.shape, (400,))
        self.assertEqual(result.cnn_prediction_voltage.shape, (400,))
        self.assertTrue(all(torch.isfinite(torch.as_tensor(result.cnn_prediction_voltage))))

    def test_missing_optional_dataset_allows_live_cnn_simulation(self) -> None:
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
                    "No upload is needed for live inverter simulation or CNN inference"
                    in item.value
                    for item in app.info
                )
            )
            self.assertFalse(
                any(
                    "selected dataset could not be loaded" in item.value.casefold()
                    or "dataset not found" in item.value.casefold()
                    for item in app.error
                )
            )

            next(
                item
                for item in app.button
                if item.label == "Run New Simulation"
            ).click()
            app.run()

        self.assertFalse(app.exception)
        self.assertIn("live_demo_result", app.session_state)
        self.assertTrue(
            any(
                item.label == "Upload project dataset ZIP"
                for item in app.file_uploader
            )
        )


if __name__ == "__main__":
    unittest.main()
