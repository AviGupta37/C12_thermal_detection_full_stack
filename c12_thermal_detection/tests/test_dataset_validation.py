"""Unit tests for dataset validation and YOLO label format verification."""

from pathlib import Path
import tempfile
import unittest

from c12_thermal_detection.dataset_tools.validate_dataset import (
    validate_yolo_label_file,
)


class TestDatasetValidation(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_valid_yolo_label_file(self):
        label_file = self.tmp_path / "valid.txt"
        label_file.write_text("0 0.512 0.438 0.125 0.350\n0 0.720 0.520 0.080 0.220\n", encoding="utf-8")
        
        is_valid, count, errors = validate_yolo_label_file(label_file, expected_class_ids={0})
        self.assertTrue(is_valid)
        self.assertEqual(count, 2)
        self.assertEqual(len(errors), 0)

    def test_empty_yolo_label_file_is_valid_negative(self):
        empty_label = self.tmp_path / "empty_negative.txt"
        empty_label.write_text("", encoding="utf-8")
        
        is_valid, count, errors = validate_yolo_label_file(empty_label, expected_class_ids={0})
        self.assertTrue(is_valid)
        self.assertEqual(count, 0)
        self.assertEqual(len(errors), 0)

    def test_invalid_coordinates(self):
        label_file = self.tmp_path / "invalid_coords.txt"
        label_file.write_text("0 1.5 0.5 -0.1 0.4\n", encoding="utf-8")
        
        is_valid, count, errors = validate_yolo_label_file(label_file, expected_class_ids={0})
        self.assertFalse(is_valid)
        self.assertTrue(any("outside [0.0, 1.0]" in e for e in errors))
        self.assertTrue(any("must be positive" in e for e in errors))

    def test_invalid_class_id(self):
        label_file = self.tmp_path / "wrong_class.txt"
        label_file.write_text("5 0.5 0.5 0.2 0.3\n", encoding="utf-8")
        
        is_valid, count, errors = validate_yolo_label_file(label_file, expected_class_ids={0})
        self.assertFalse(is_valid)
        self.assertTrue(any("not in expected classes" in e for e in errors))

    def test_malformed_syntax(self):
        label_file = self.tmp_path / "syntax_error.txt"
        label_file.write_text("0 0.5 0.5\n", encoding="utf-8")
        
        is_valid, count, errors = validate_yolo_label_file(label_file, expected_class_ids={0})
        self.assertFalse(is_valid)
        self.assertTrue(any("expected 5 elements" in e for e in errors))


if __name__ == "__main__":
    unittest.main()

