"""Unit tests for YAML configuration parsing."""

from pathlib import Path
import unittest

from c12_thermal_detection.training.training_utils import load_yaml_config


class TestConfig(unittest.TestCase):

    def test_load_app_config(self):
        config_path = Path("c12_thermal_detection/config/app.yaml")
        if not config_path.exists():
            self.skipTest("app.yaml not found at expected path")

        cfg = load_yaml_config(config_path)
        self.assertIn("app", cfg)
        self.assertIn("storage", cfg)
        self.assertIn("performance", cfg)

    def test_load_detection_config(self):
        config_path = Path("c12_thermal_detection/config/detection.yaml")
        if not config_path.exists():
            self.skipTest("detection.yaml not found at expected path")

        cfg = load_yaml_config(config_path)
        self.assertIn("model", cfg)
        self.assertIn("inference", cfg)
        self.assertGreater(cfg["inference"]["conf_threshold"], 0.0)
        self.assertEqual(cfg["inference"]["target_classes"], [0])

    def test_load_training_config(self):
        config_path = Path("c12_thermal_detection/config/training.yaml")
        if not config_path.exists():
            self.skipTest("training.yaml not found at expected path")

        cfg = load_yaml_config(config_path)
        self.assertIn("training", cfg)
        self.assertIn("hyperparameters", cfg)
        self.assertIn("augmentation", cfg)
        self.assertGreater(cfg["training"]["batch_size"], 0)


if __name__ == "__main__":
    unittest.main()

