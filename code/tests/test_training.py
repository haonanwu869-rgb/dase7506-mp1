import contextlib
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import train
from student import build_model


class TrainingTests(unittest.TestCase):
    def test_dropout_and_schedule(self):
        config = dict(vocab=2048, context=256, width=32, heads=4, depth=1)
        self.assertEqual(build_model(config).blocks[0].drop.p, .1)
        self.assertEqual(build_model(config | {'dropout': .23}).blocks[0].drop.p, .23)
        self.assertAlmostEqual(train.learning_rate(0, 2000, 1e-4, 1e-5, 0), 1e-4)
        self.assertAlmostEqual(train.learning_rate(1999, 2000, 1e-4, 1e-5, 0), 1e-5)

    def test_rng_roundtrip(self):
        g = torch.Generator().manual_seed(123)
        state = train.rng_state(g)
        def draw():
            return (torch.rand(4), torch.rand(4, generator=g), random.random(), np.random.rand())
        first = draw()
        train.restore_rng(state, g)
        second = draw()
        for a, b in zip(first[:2], second[:2]):
            self.assertTrue(torch.equal(a, b))
        self.assertEqual(first[2:], second[2:])

    def test_resume_matches_uninterrupted_with_ema_and_dropout(self):
        # Synthetic data for engineering correctness only, never used in experiments.
        data = {'train': (torch.arange(1100) % 2048, 2200),
                'validation': (torch.arange(270) % 2048, 540)}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = root/'config.json'
            config.write_text(json.dumps(dict(vocab=2048, context=256, width=32, heads=4, depth=1)))
            common = ['train.py', '--config', str(config), '--steps', '4', '--batch-size', '2',
                      '--threads', '2', '--ema-decay', '.9', '--eval-every', '2', '--warmup', '0']
            def run(name, extra):
                with patch.object(sys, 'argv', common+['--run-dir', str(root/name)]+extra), \
                     patch.object(train, 'load_data', return_value=data), contextlib.redirect_stdout(io.StringIO()):
                    train.main()
                return torch.load(root/name/'training_state.pt', weights_only=True)
            whole = run('whole', [])
            part = run('part', ['--stop-after', '2'])
            resumed = run('resumed', ['--resume', str(root/'part/training_state.pt')])
            for key in ('model', 'ema'):
                for name in whole[key]:
                    self.assertTrue(torch.equal(whole[key][name], resumed[key][name]), name)
            self.assertEqual(whole['train_tokens'], resumed['train_tokens'])
            self.assertEqual(whole['selection'], resumed['selection'])
            for index, state in whole['optimizer']['state'].items():
                for name, value in state.items():
                    self.assertTrue(torch.equal(value, resumed['optimizer']['state'][index][name]))
            self.assertEqual(part['stage_step'], 2)
            self.assertTrue(torch.equal(whole['rng']['torch'], resumed['rng']['torch']))
            shortened = run('shortened', ['--resume', str(root/'part/training_state.pt'), '--resume-horizon', '3'])
            self.assertEqual(shortened['stage_step'], 3)
            self.assertEqual(shortened['schedule_horizon'], 3)
            self.assertEqual(shortened['optimizer']['param_groups'][0]['lr'], 1e-4)


if __name__ == '__main__':
    unittest.main()
