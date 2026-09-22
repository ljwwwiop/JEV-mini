import json
import math
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from jev_mini.data import distill_request, iter_records, materialize
from jev_mini.evaluate import evaluate_records


class DistillTests(unittest.TestCase):
    def row(self, kind, options, target):
        return {'id': 'row-1', 'state': 'context', 'question': 'question',
                'kind': kind, 'options': options, 'target': target, 'source': 'teacher'}

    def test_soft_targets_and_option_order(self):
        r = self.row('choice', ['duplicate', 'duplicate', 'third'], [0.2, 0.3, 0.5001])
        q = materialize(distill_request(r))['questions'][0]
        self.assertEqual(q['label'], 2)
        self.assertEqual(len(q['options']), 3)
        for actual, expected in zip(q['target'], r['target']):
            self.assertAlmostEqual(actual, expected / 1.0001)

    def test_noul_score_and_invalid_targets(self):
        for kind, options in [('noul', ['false', 'true']), ('score', ['low', 'high'])]:
            q = materialize(distill_request(self.row(kind, options, [0.1, 0.9])))['questions'][0]
            self.assertEqual(q['label'], 1)
            self.assertEqual(q['target'], [0.1, 0.9])
        for target in ([0, 0], [-1, 2], [math.nan, 1], [1]):
            with self.assertRaises(ValueError):
                distill_request(self.row('noul', ['false', 'true'], target))

    def test_loader_includes_line_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'rows.jsonl'
            p.write_text(json.dumps(self.row('noul', ['false', 'true'], [0.3, 0.7])) + '\n{}\n')
            rows = iter_records(p)
            self.assertEqual(next(rows)['_meta']['id'], 'row-1')
            with self.assertRaisesRegex(ValueError, ':2:'):
                next(rows)

    def test_evaluation_batches_and_restores_training_mode(self):
        class FakeModel(torch.nn.Module):
            def encode(self, tokenizer, rec, strict, **kwargs):
                return {'ids': [1]}
            def forward_batch(self, encs):
                return [[torch.tensor([0.25, 0.75]).log()] for _ in encs]
        model = FakeModel()
        rows = [distill_request(self.row('noul', ['false', 'true'], [0.25, 0.75])) for _ in range(3)]
        # Only the context constant is needed; no Transformers/model weights in this test.
        module = types.ModuleType('jev_mini.model'); module.MAX_PACKED = 2048
        with patch.dict(sys.modules, {'jev_mini.model': module}):
            result = evaluate_records(model, None, rows, batch=2)
        self.assertEqual(result['records'], 3)
        self.assertEqual(result['teacher_argmax_accuracy'], 1)
        self.assertAlmostEqual(result['brier_to_target'], 0)
        self.assertAlmostEqual(result['soft_target_cross_entropy'], -0.25*math.log(0.25)-0.75*math.log(0.75), places=6)
        self.assertTrue(model.training)


if __name__ == '__main__':
    unittest.main()
