import copy
from pathlib import Path
import random
import unittest
import torch
from jev_mini.data import load_records, materialize, augment
from jev_mini.losses import question_loss, permutation_kl

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/train.jsonl'

class CoreTests(unittest.TestCase):
    def test_three_types(self):
        rec = materialize(load_records(EXAMPLE)[0])
        self.assertEqual([q['label'] for q in rec['questions']], [0, 1, 2])
        self.assertEqual([len(q['options']) for q in rec['questions']], [3, 2, 3])

    def test_permutation_preserves_label(self):
        request = load_records(EXAMPLE)[0]
        for seed in range(10):
            rec = materialize(augment(request, random.Random(seed), 0, 0, 0))
            self.assertEqual(rec['questions'][0]['keys'][rec['questions'][0]['label']], 'billing')

    def test_invalid_labels_and_targets(self):
        request = load_records(EXAMPLE)[0]
        for value in [-1, 3, 1.5, True]:
            r = copy.deepcopy(request); r['questions']['urgency']['label'] = value
            with self.assertRaises(ValueError): materialize(r)
        request['questions']['team']['target'] = {'billing': -1, 'technical': 2}
        with self.assertRaises(ValueError): materialize(request)

    def test_losses_backward(self):
        z = torch.tensor([0.3, 0.7, -0.2], requires_grad=True)
        q = {'label': 1, 'qtype': 'choice'}
        loss = question_loss(z, q, 'cpu', 0)
        torch.testing.assert_close(loss, -z.log_softmax(-1)[1])
        loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        q['target'] = [0.2, 0.5, 0.3]
        torch.testing.assert_close(question_loss(z, q, 'cpu', 0), -(torch.tensor(q['target']) * z.log_softmax(-1)).sum())

    def test_permutation_kl_alignment(self):
        z = torch.tensor([0.2, 1.3, -0.4])
        perm = [2, 0, 1]
        self.assertAlmostEqual(permutation_kl(z, z[perm], perm, 'cpu').item(), 0, places=6)

if __name__ == '__main__':
    unittest.main()
