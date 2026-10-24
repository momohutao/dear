"""Run with DEAR_TEST_TOKENIZER pointing to a local tokenizer directory."""

import os
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pointwise_reranker"))

from eval_text import format_eval_document


class DocumentFormattingTest(unittest.TestCase):
    def test_historical_inference_whitespace(self):
        self.assertEqual(format_eval_document("", "hello"), "document:  hello")
        self.assertEqual(format_eval_document("title", " text "), "document: title  text ")
        self.assertEqual(format_eval_document("title", "text", "|"), "document: title|text")


@unittest.skipUnless(os.environ.get("DEAR_TEST_TOKENIZER"), "Set DEAR_TEST_TOKENIZER for token parity tests")
class EvaluationTokenParityTest(unittest.TestCase):
    def test_early_stop_matches_final_inference(self):
        from transformers import AutoTokenizer
        from data import RerankerInferencePreProcessor, RerankerInferenceDataset, RerankerInferenceCollator
        from trainer import _DLEvalDataset, _DLEvalCollator

        tokenizer = AutoTokenizer.from_pretrained(os.environ["DEAR_TEST_TOKENIZER"], local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        rows = [
            ("", "5. Cancel."),
            ("", " leading and trailing whitespace "),
            ("Title", "ordinary passage"),
            (" Title ", " passage with whitespace "),
            ("", "long passage " * 400),
        ]
        early = _DLEvalDataset.__new__(_DLEvalDataset)
        early.rows = [("1", str(i), "example query", title, text) for i, (title, text) in enumerate(rows)]
        early.tokenizer = tokenizer
        early.q_max_len, early.p_max_len = 32, 196
        preprocess = RerankerInferencePreProcessor(tokenizer, tokenizer, 32, 196)
        processed = [preprocess(dict(query_id=q, docid=d, query=query, title=title, text=text))
                     for q, d, query, title, text in early.rows]
        final = RerankerInferenceDataset(processed, tokenizer, tokenizer, 32, 196)
        early_qids, early_docs, early_batch = _DLEvalCollator(tokenizer, 228)([early[i] for i in range(len(rows))])
        final_qids, final_docs, final_batch, _ = RerankerInferenceCollator(tokenizer, tokenizer)([final[i] for i in range(len(rows))])
        self.assertEqual(early_qids, final_qids)
        self.assertEqual(early_docs, final_docs)
        self.assertEqual(set(early_batch), set(final_batch))
        for key in early_batch:
            self.assertTrue(early_batch[key].equal(final_batch[key]), key)


if __name__ == "__main__":
    unittest.main()
