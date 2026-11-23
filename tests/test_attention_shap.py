"""Independent games exercise Shapley interactions, efficiency, and display grouping."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location(
    "attention_shap", Path(__file__).resolve().parents[1] / "scripts/export_attention_shap.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExactShapleyTests(unittest.TestCase):
    def test_additive_game_with_nonzero_baseline(self):
        scores = [7 + 2*bool(s&1) - 3*bool(s&2) for s in range(4)]
        self.assertEqual(module.exact_shapley(scores, 2), [2, -3])

    def test_interaction_is_shared_and_dummy_gets_zero(self):
        # Only players 0 and 2 jointly create six units; player 1 does nothing.
        scores = [4 + (6 if s&1 and s&4 else 0) for s in range(8)]
        self.assertEqual(module.exact_shapley(scores, 3), [3, 0, 3])

    def test_three_way_interaction(self):
        self.assertEqual(module.exact_shapley([0]*7 + [9], 3), [3, 3, 3])

    def test_subwords_are_summed_without_losing_punctuation(self):
        text = "Jest wasn't good."
        offsets = [[0,0],[0,2],[2,4],[5,9],[9,10],[10,11],[12,16],[16,17],[0,0]]
        words = module.display_words(text, offsets, [0,1,2,3,4,5,6,7,0])
        self.assertEqual([w["text"] for w in words], ["Jest", "wasn't", "good", "."])
        self.assertEqual([w["shap_value"] for w in words], [3,12,6,7])
        self.assertEqual([w["token_positions"] for w in words], [[1,2],[3,4,5],[6],[7]])


if __name__ == "__main__":
    unittest.main()
