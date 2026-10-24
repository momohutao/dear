"""CPU tests for BEIR checkpoint selection without loading model weights."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pointwise_reranker"))
from trainer import DLAverageEarlyStoppingCallback


class BeirSelectionTest(unittest.TestCase):
    def test_beir_mean_best_link_and_patience(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("trainer.os.path.isfile", return_value=True), patch("trainer.os.path.getsize", return_value=1):
                callback = DLAverageEarlyStoppingCallback(
                    tokenizer=SimpleNamespace(bos_token_id=1), eval_steps=500,
                    patience=3, min_delta=0, batch_size=32, q_max_len=32, p_max_len=196,
                    dl19_path="unused", dl20_path="unused", dl19_qrels="unused", dl20_qrels="unused",
                    metric="beir8", beir_data_dir=directory, beir_qrels_dir=directory,
                )
            self.assertEqual(list(callback.dataset_paths),
                             ["covid", "nfc", "touche", "dbpedia", "scifact", "signal", "news", "robust04"])
            args = SimpleNamespace(output_dir=directory, device=torch.device("cpu"))
            state = SimpleNamespace(global_step=499)
            control = SimpleNamespace(should_training_stop=False)
            model = SimpleNamespace(student=SimpleNamespace(save_pretrained=lambda *a, **k: None), train=lambda: None)
            scores = dict(zip(callback.dataset_paths, [.8, .3, .4, .5, .7, .2, .6, .5]))
            with patch("trainer.is_deepspeed_zero3_enabled", return_value=False), \
                 patch("trainer._DLEvalDataset", return_value=[]), \
                 patch.object(callback, "_score_dataset", return_value=[]) as scorer, \
                 patch.object(callback, "_evaluate_with_trec", side_effect=lambda name, *a: scores[name]):
                callback.on_step_end(args, state, control, model)
                scorer.assert_not_called()
                for step in [500, 1000, 1500, 2000]:
                    state.global_step = step
                    callback.on_step_end(args, state, control, model)
                    self.assertEqual(control.should_training_stop, step == 2000)
                self.assertEqual(scorer.call_count, 32)
            self.assertAlmostEqual(state.best_metric, .5)
            self.assertEqual((root / "best-adapter").resolve(), root / "adapter-step-500")
            record = json.loads((root / "early_stop_state.json").read_text())
            self.assertEqual(record["latest"]["metric"], "beir8")
            self.assertEqual(record["latest"]["bad_evaluations"], 3)
            self.assertEqual(len((root / "early-stop-eval/step-500/beir8_summary.tsv").read_text().splitlines()), 9)


if __name__ == "__main__":
    unittest.main()
