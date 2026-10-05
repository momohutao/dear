import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mine_intent_hard_negatives import classify, make_context, mine_row


def doc(i, text):
    return {"docid": str(i), "title": "", "text": text}


class MiningTests(unittest.TestCase):
    def test_constraint_and_synonym(self):
        ctx = make_context("right pelvic pain causes", [doc(1, "Right pelvic pain due to infection")])
        self.assertIn("keyword_constraint_missing", classify(doc(2, "Pelvic pain causes include infection"), ctx)["labels"])
        ctx = make_context("female pelvic pain", [doc(1, "Pelvic pain in women")])
        self.assertNotIn("keyword_constraint_missing", classify(doc(2, "Pelvic pain in women"), ctx)["labels"])

    def test_domain(self):
        ctx = make_context("definition of mechanical ventilation", [doc(1, "Medical mechanical ventilation helps patient breathing and lungs")])
        self.assertIn("domain_confusion", classify(doc(2, "Mechanical ventilation in HVAC building ducts and architecture"), ctx)["labels"])

    def test_intent(self):
        ctx = make_context("what types of food can you cook sous vide", [doc(1, "Sous vide foods include eggs, beef, and fish")])
        self.assertIn("intent_mismatch", classify(doc(2, "Sous vide is a method to cook food in a water bath"), ctx)["labels"])
        self.assertNotIn("intent_mismatch", classify(doc(3, "Sous vide foods include eggs, beef, fish"), ctx)["labels"])

    def test_conservative_rule_guards(self):
        ctx = make_context("medical cell biology", [doc(1, "Medical biology examines cells")])
        self.assertNotIn("domain_confusion", classify(doc(2, "Medical biology cells tissue organisms species genetic bacteria"), ctx)["labels"])
        ctx = make_context("where can static members be used", [doc(1, "Static members share values across instances")])
        self.assertIsNone(ctx[-1])

    def example(self):
        return {"query_id": "q1", "query": "right pelvic pain causes", "positive_passages": [doc(1, "Right pelvic pain caused by injury"), doc(2, "Right pelvic pain caused by infection")],
                "negative_passages": [doc(1, "conflicting positive id"), doc(100, "Right pelvic pain caused by infection")]
                + [doc(i + 3, f"Pelvic pain causes include disorder {i}") for i in range(10)]
                + [doc(50, "irrelevant unique text"), doc(51, "irrelevant unique text")]}

    def test_size_exclusions_and_repeatability(self):
        row = self.example()
        a = mine_row(row)
        self.assertEqual(a, mine_row(row))
        self.assertEqual(len(a[0]["positive_passages"]), 1)
        neg = a[0]["negative_passages"]
        self.assertEqual(len(neg), 7)
        self.assertEqual(len({p["docid"] for p in neg}), 7)
        self.assertTrue({p["docid"] for p in neg}.isdisjoint({"1", "2", "100"}))
        self.assertEqual(a[-1]["known_positive_or_positive_copy"], 2)
        self.assertEqual(a[-1]["duplicate_negative"], 1)
        b = mine_row(row, hard_slots=0)
        self.assertEqual(set(b[3]), set(b[4]))

    def test_insufficient_pool_fails(self):
        row = self.example()
        row["negative_passages"] = row["negative_passages"][:4]
        with self.assertRaisesRegex(ValueError, "unique eligible negatives"):
            mine_row(row, short_pool="error")
        result = mine_row(row)
        self.assertEqual(len(result[0]["negative_passages"]), 7)
        self.assertEqual(sorted(result[3]), sorted(result[4]))
        row["negative_passages"] = []
        with self.assertRaises(ValueError):
            mine_row(row)

    def test_cli_gzip_stats_and_failure_atomicity(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "input.jsonl.gz"
            with gzip.open(source, "wt") as f:
                f.write(json.dumps(self.example()) + "\n")
            outputs = []
            for name in ("a.gz", "b.gz"):
                dest = tmp / name
                cmd = [sys.executable, str(Path(__file__).with_name("mine_intent_hard_negatives.py")), "--input", str(source), "--output", str(dest)]
                subprocess.run(cmd, check=True, capture_output=True)
                outputs.append(dest.read_bytes())
                stats = json.loads(Path(str(dest) + ".stats.json").read_text())
                for label in ("before", "after"):
                    self.assertEqual(sum(x["count"] for x in stats["primary_distributions"][label].values()), 7)
                self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(*outputs)
            with gzip.open(source, "at") as f:
                f.write("invalid json\n")
            dest = tmp / "failed.gz"
            cmd[-1] = str(dest)
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertFalse(dest.exists())
            self.assertFalse(Path(str(dest) + ".stats.json").exists())


if __name__ == "__main__":
    unittest.main()
