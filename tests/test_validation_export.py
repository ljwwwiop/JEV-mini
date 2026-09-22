import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from jev_mini.context import Context
from jev_mini.evaluate import evaluate_to_json


class FakeModel(torch.nn.Module):
    def encode(self, tokenizer, rec, **kwargs):
        if rec['state'] == 'too long':
            raise ValueError('state exceeds 384 tokens: 500')
        return {'ids': [1], 'questions': len(rec['questions'])}

    def forward_batch(self, encs):
        assert not self.training
        assert not torch.is_grad_enabled()
        return [[torch.tensor([0.25, 0.75]).log() for _ in range(e['questions'])] for e in encs]


class ValidationExportTests(unittest.TestCase):
    def test_raw_rows_order_skips_and_probabilities(self):
        flat = {'state': 'context', 'question': 'pick', 'kind': 'choice',
                'options': ['first', 'second'], 'target': [1, 3], 'extra': {'keep': True}}
        skipped = {**flat, 'state': 'too long'}
        native = {'state': 'context', 'questions': {
            'a': {'type': 'noul', 'instructions': 'yes?', 'label': True},
            'b': {'type': 'score', 'instructions': 'level?', 'criteria': ['low', 'high'], 'label': 1}}}
        rows = [flat, skipped, native]
        with tempfile.TemporaryDirectory() as tmp:
            data, output = Path(tmp) / 'val.jsonl', Path(tmp) / 'validation' / 'step_00000500.json'
            data.write_text('\n'.join(map(json.dumps, rows)))
            model = FakeModel()
            metrics = evaluate_to_json(model, None, data, output, batch=2, context=Context(overlength='skip'))
            exported = json.loads(output.read_text())
            self.assertEqual(len(exported), 3)
            for original, result in zip(rows, exported):
                prediction = result.pop('model_prediction')
                self.assertEqual(original, result)
            predictions = json.loads(output.read_text())
            pred = predictions[0]['model_prediction']
            self.assertEqual(pred['index'], 1)
            self.assertEqual(pred['option'], 'second')
            for actual, expected in zip(pred['probabilities'], [0.25, 0.75]):
                self.assertAlmostEqual(actual, expected, places=6)
            self.assertEqual(predictions[1]['model_prediction']['status'], 'skipped_overlength')
            self.assertEqual(set(predictions[2]['model_prediction']['questions']), {'a', 'b'})
            self.assertEqual(metrics['records'], 2)
            self.assertEqual(metrics['skipped_overlength'], 1)
            self.assertTrue(model.training)
            self.assertFalse(output.with_suffix('.json.tmp').exists())

    def test_failure_restores_mode_and_does_not_publish_partial_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, output = Path(tmp) / 'val.jsonl', Path(tmp) / 'out.json'
            data.write_text(json.dumps({'state': 'too long', 'kind': 'noul', 'question': '?',
                                       'options': ['false', 'true'], 'target': [0, 1]}))
            model = FakeModel()
            with self.assertRaises(ValueError):
                evaluate_to_json(model, None, data, output)
            self.assertTrue(model.training)
            self.assertFalse(output.exists())
            self.assertFalse(output.with_suffix('.json.tmp').exists())

    def test_sample_limit_does_not_read_remaining_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, output = Path(tmp) / 'val.jsonl', Path(tmp) / 'out.json'
            row = {'state': 'context', 'kind': 'noul', 'question': '?',
                   'options': ['false', 'true'], 'target': [0, 1]}
            data.write_text(json.dumps(row) + '\ninvalid unread JSON\n')
            metrics = evaluate_to_json(FakeModel(), None, data, output, batch=8, max_records=1)
            self.assertEqual(metrics['requested_records'], 1)
            self.assertEqual(metrics['records'], 1)
            self.assertEqual(len(json.loads(output.read_text())), 1)

    def test_tensorboard_scalar_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            with SummaryWriter(tmp) as writer:
                writer.add_scalar('train/loss', 0.8821, 500)
            events = EventAccumulator(tmp).Reload().Scalars('train/loss')
            self.assertEqual(events[0].step, 500)
            self.assertAlmostEqual(events[0].value, 0.8821, places=6)
