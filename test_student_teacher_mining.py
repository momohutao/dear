import unittest
import random
import json
import tempfile
from pathlib import Path
from mine_student_teacher_hard_negatives import choose_top1, make_record
from pointwise_reranker.runtime_hard_negative import candidate_indices, inject_one, injection_decision, load_candidates, sha256


def doc(did, text):
    return {'docid': did, 'title': '', 'text': text}


class StudentTeacherMiningTests(unittest.TestCase):
    def test_completed_sidecar_uses_teacher_approval_only(self):
        row = self.row()
        record = make_record(row, [10., 9.], [9., 3.], [3., 3.], 2., .5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / 'train.jsonl'
            train.write_text(json.dumps(row)+'\n')
            manifest = {'format': 'student_top1_teacher_margin_v1',
                        'settings': {'q_max_len': 32, 'p_max_len': 196},
                        'input_sha256': sha256(train)}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            def write_record():
                (root / 'candidates.jsonl').write_text(json.dumps(record)+'\n')
                (root / 'COMPLETE.json').write_text(json.dumps({
                    'complete': True, 'full_dataset': True,
                    'manifest_sha256': sha256(root / 'manifest.json'),
                    'candidates_sha256': sha256(root / 'candidates.jsonl'),
                    'counts': {'queries_with_candidate': 1}}))
            write_record()
            self.assertEqual(candidate_indices(row, load_candidates(root, [train], 32, 196)), [0, 0])
            record['accepted_positives'][1]['student_positive_score'] = 11.
            write_record()
            self.assertEqual(candidate_indices(row, load_candidates(root, [train], 32, 196)), [0, 0])
            del record['accepted_positives'][1]['student_positive_score']
            write_record()
            self.assertEqual(candidate_indices(row, load_candidates(root, [train], 32, 196)), [0, 0])
            result, slot, reason = injection_decision([1] * 7, 0, 1., random.Random(42))
            self.assertEqual(reason, 'replaced')
            self.assertEqual(result[slot], 0)
            record['accepted_positives'][1]['student_positive_score'] = 9.
            record['accepted_positives'][1]['teacher_positive_score'] = 2.5
            record['accepted_positives'][1]['teacher_margin'] = .5
            write_record()
            with self.assertRaises(ValueError):
                load_candidates(root, [train], 32, 196)

    def test_training_injection_reasons_and_aligned_draws(self):
        baseline = [1, 2, 3, 4, 5, 6, 7]
        states = []
        for candidate, probability, expected in [
            (-1, 1., 'missing_candidate'), (0, 0., 'probability_missed'),
            (2, 1., 'already_present'), (0, 1., 'replaced'),
        ]:
            rng = random.Random(42)
            result, slot, reason = injection_decision(baseline, candidate, probability, rng)
            self.assertEqual(reason, expected)
            if expected == 'replaced':
                self.assertEqual(result[slot], candidate)
                self.assertEqual(sum(a != b for a, b in zip(result, baseline)), 1)
            else:
                self.assertEqual((result, slot), (baseline, -1))
            states.append(rng.getstate())
        self.assertTrue(all(state == states[0] for state in states))

    def row(self):
        return {'query_id': 'q', 'query': 'right pelvic pain causes',
                'positive_passages': [doc('p1', 'Right pelvic pain caused by infection'),
                                      doc('p2', 'Right pelvic pain due to injury')],
                'negative_passages': [doc('n1', 'Pelvic pain due to disorder'),
                                      doc('n2', 'Pain elsewhere')]}

    def test_teacher_gate_is_strict_and_positive_specific(self):
        r = make_record(self.row(), [8., 10.], [9., 3.], [2.5, 2.5001], 2., .5)
        self.assertEqual(r['hard_docid'], 'n1')
        self.assertEqual([p['positive_docid'] for p in r['accepted_positives']], ['p2'])
        self.assertEqual([p['positive_docid'] for p in r['rejected_positives']], ['p1'])
        self.assertFalse(r['accepted_positives'][0]['student_negative_above_positive'])

    def test_no_top2_fallback_if_top1_rejected(self):
        r = make_record(self.row(), [0., 0.], [9., 3.], [1., 1.], 2., .5)
        self.assertEqual(r['hard_docid'], 'n1')
        self.assertFalse(r['accepted_positives'])

    def test_excludes_known_positive_copy_even_with_teacher_approval(self):
        row = self.row()
        row['negative_passages'][0]['text'] = row['positive_passages'][1]['text'].upper()
        r = make_record(row, [0., 0.], [9., 3.], [10., 10.], 0., .5)
        self.assertFalse(r['accepted_positives'])
        self.assertTrue(all(p['reason'] == 'known_positive_or_copy' for p in r['rejected_positives']))

    def test_tie_break_and_score_alignment(self):
        row = self.row()
        self.assertEqual(choose_top1(row, [9., 9.]), 0)
        self.assertEqual(choose_top1(row, [3., 9.]), 1)
        with self.assertRaises(ValueError):
            choose_top1(row, [9.])
        with self.assertRaises(ValueError):
            choose_top1(row, [float('nan'), 3.])
        with self.assertRaises(ValueError):
            make_record(row, [0., 0.], [9., 3.], [10., 10.], float('nan'), .5)

    def test_runtime_approval_follows_selected_positive(self):
        row = self.row()
        record = make_record(row, [0., 0.], [9., 3.], [2.5, 3.], 2., .5)
        candidates = {'q': (record['source_row_sha256'], 0, 'n1', ['p2'])}
        self.assertEqual(candidate_indices(row, candidates), [-1, 0])
        row['negative_passages'].reverse()
        with self.assertRaises(ValueError):
            candidate_indices(row, candidates)

    def test_runtime_only_changes_one_slot_and_preserves_disabled_rng(self):
        baseline = [1, 2, 3, 4, 5, 6, 7]
        rng = random.Random(42)
        state = rng.getstate()
        self.assertEqual(inject_one(baseline, 0, 0, rng), (baseline, -1))
        self.assertEqual(rng.getstate(), state)
        self.assertEqual(inject_one(baseline, -1, 1, rng), (baseline, -1))
        self.assertEqual(inject_one(baseline, 2, 1, rng), (baseline, -1))
        for _ in range(100):
            result, slot = inject_one(baseline, 0, 1, rng)
            self.assertEqual(result[slot], 0)
            self.assertEqual(sum(a != b for a, b in zip(result, baseline)), 1)

    def test_candidate_validation_inside_dataset_map(self):
        try:
            from datasets import Dataset
        except ImportError:
            self.skipTest('requires training environment datasets')
        row = self.row()
        record = make_record(row, [0., 0.], [9., 3.], [2.5, 3.], 2., .5)
        candidates = {'q': (record['source_row_sha256'], 0, 'n1', ['p2'])}
        types_seen = []

        def preprocess(example):
            types_seen.append(type(example).__name__)
            return {'approved': candidate_indices(example, candidates)}

        result = Dataset.from_list([row]).map(preprocess, load_from_cache_file=False)
        self.assertEqual(types_seen, ['LazyRow'])
        self.assertEqual(result[0]['approved'], [-1, 0])
        row['negative_passages'][0]['text'] = 'changed content'
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            Dataset.from_list([row]).map(preprocess, load_from_cache_file=False)


if __name__ == '__main__':
    unittest.main()
