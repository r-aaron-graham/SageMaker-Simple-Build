import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from inference import input_fn, model_fn, output_fn, predict_fn


class InferenceSmokeTest(unittest.TestCase):
    def test_train_reload_and_predict(self):
        trainer = Path(__file__).with_name("SageMaker SKLearn estimator.py")
        with tempfile.TemporaryDirectory() as model_dir:
            subprocess.run(
                [sys.executable, str(trainer), "--model-dir", model_dir],
                check=True,
            )
            model = model_fn(model_dir)
            rows = input_fn("[[0.5, 0.2, 0.8]]", "application/json")
            body, content_type = output_fn(predict_fn(rows, model), "application/json")
        self.assertEqual(content_type, "application/json")
        predictions = json.loads(body)
        self.assertEqual(len(predictions), 1)
        self.assertAlmostEqual(predictions[0], 3.28, delta=0.05)

    def test_rejects_invalid_requests(self):
        for body in ("[]", "[[1, 2]]", "[[1, 2, NaN]]"):
            with self.subTest(body=body), self.assertRaises(ValueError):
                input_fn(body, "application/json")
        with self.assertRaises(ValueError):
            input_fn("[[1, 2, 3]]", "text/plain")


if __name__ == "__main__":
    unittest.main()
