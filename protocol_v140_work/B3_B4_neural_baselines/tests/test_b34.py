import inspect
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT.parent / "shared"))

from b34.data import CharacterCodec, codec_for
from b34.decoding import beam_search, canonical_topk
from b34.engine import _evaluate_checkpoint, validate_config
from b34.io import read
from benchmark_protocol_v140 import FORMAL_TEST_SUITES
from b34.model import BiLSTMMLM, build_model, model_from_spec, model_spec


DATA = PROJECT.parent / "shared/data/v1.5.0"


class BaselineProtocolTests(unittest.TestCase):
    def config(self, arm):
        return read(PROJECT / f"configs/{arm}.json")

    def test_frozen_configs(self):
        for arm in ("B3", "B4"):
            validate_config(self.config(arm))

    def test_protocol_identity(self):
        for arm in ("B3", "B4"):
            cfg = self.config(arm)
            self.assertEqual(cfg["dataset_version"], "v1.5.0")
            self.assertEqual(cfg["prediction_space"], "shared_4116_targets")
            self.assertTrue(cfg["training_boundary_masks_enabled"])
            self.assertTrue(cfg["training_square_bracket_restorations_maskable"])
            self.assertEqual(
                cfg["protected_target_types"], ["damage", "ligature", "annotation_suffix", "non_main"]
            )
            self.assertEqual(cfg["test_suites"], list(FORMAL_TEST_SUITES))

    def test_character_codec_has_distinct_mask_and_damage(self):
        codec = CharacterCodec(DATA, self.config("B3"))
        self.assertEqual(codec.vocab_size, 4550)
        self.assertEqual(len(codec.units), 4116)
        self.assertNotEqual(codec.mask_id, codec.unit_to_id["[LACUNA_1]"])
        question = {
            "gap_length": 1,
            "prefix_units": ["王", "[LACUNA_1]"],
            "suffix_units": ["[LACUNA_N]"],
            "gold_units": [codec.units[0]],
        }
        encoded = codec.encode(question, supervised=True)
        self.assertEqual(len(encoded["positions"]), 1)
        self.assertEqual(encoded["targets"], [0])
        self.assertIn(codec.unit_to_id["[LACUNA_1]"], encoded["input_ids"])

    def test_annotation_control_alone_is_not_source_context(self):
        for arm in ('B3', 'B4'):
            codec = codec_for(DATA, self.config(arm))
            with self.assertRaises(ValueError):
                codec.encode({'gap_length': 1, 'prefix_units': [],
                              'suffix_units': ['<ANNOTATION_BOUNDARY>']})

    def test_b3_is_two_layer_bidirectional_lstm(self):
        cfg = self.config("B3")
        codec = codec_for(DATA, cfg)
        model = build_model(codec, cfg)
        self.assertIsInstance(model, BiLSTMMLM)
        self.assertEqual(model.encoder.num_layers, 2)
        self.assertTrue(model.encoder.bidirectional)
        self.assertEqual(model.classifier.out_features, 4116)
        spec = model_spec(model, cfg)
        restored = model_from_spec(spec)
        self.assertEqual(restored.model_spec(), spec)

    def test_b4_matches_e1_capacity_and_is_random_init(self):
        cfg = self.config("B4")
        self.assertEqual(
            (cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"], cfg["intermediate_size"]),
            (12, 768, 12, 3072),
        )
        self.assertEqual(cfg["learning_rate"], 5e-5)
        from b34 import model as model_module

        self.assertNotIn("from_pretrained", inspect.getsource(model_module))

    def test_stable_ties_and_gold_free_beam(self):
        self.assertEqual(canonical_topk([1.0, 1.0, 0.0], 2).tolist(), [0, 1])
        encoded = [{"input_ids": [2, 1, 3], "positions": [1]}]

        def scores(inputs, positions):
            self.assertEqual(inputs, [[2, 1, 3]])
            self.assertEqual(positions, [1])
            import numpy as np

            return np.array([[0.0, -1.0, -2.0]])

        result = beam_search(encoded, [10, 11, 12], scores, beam_width=3, returns=2)
        self.assertEqual(result[0]["paths"][0], [0])
        self.assertEqual(result[0]["single_ranking"], [0, 1, 2])

    def test_test_predictions_precede_gold_open(self):
        source = inspect.getsource(_evaluate_checkpoint)
        first_prediction = source.index("predict_questions")
        first_gold = source.index("include_answers=True")
        self.assertLess(first_prediction, first_gold)


if __name__ == "__main__":
    unittest.main()
