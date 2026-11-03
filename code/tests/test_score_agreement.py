import unittest

from score_agreement import score


class AgreementTest(unittest.TestCase):
    def test_scores_saved_winners_in_query_order(self):
        reference = dict(fixture_id="fixture", noise_level=0.4, queries=3, passages=10,
                         exact_top1_ids=[1, 7, 3], source_ids=[1, 2, 3])
        trials = [
            dict(query=2, repeat=0, targetId=3, resultIds=[4]),
            dict(query=0, repeat=0, targetId=1, resultIds=[1]),
            dict(query=1, repeat=0, targetId=2, resultIds=[7]),
        ]
        result = dict(fixture_id="fixture", noise={"level": 0.4}, queries=3, k=1, repeats=1,
                      trials=trials, clusters=32768, probes=4, codec="sq6", accuracy_only=False)
        scored = score(reference, result)
        self.assertEqual(scored["top1_agreement"], 2 / 3)
        self.assertEqual(scored["source_matches"], 1)
        self.assertEqual(scored["exact_source_matches"], 2)
        with self.assertRaisesRegex(ValueError, "same top-1 queries"):
            score(reference, {**result, "fixture_id": "other"})
        with self.assertRaisesRegex(ValueError, "mismatched browser trial"):
            score(reference, {**result, "trials": [trials[0], trials[0], trials[1]]})
