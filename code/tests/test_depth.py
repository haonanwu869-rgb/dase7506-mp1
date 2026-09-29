import unittest
import torch
from torch.nn import functional as F

from hybrid import Hybrid
from prepare_depth import deepen
from train import training_component


class DepthTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(53)
        config = dict(vocab=2048, context=256, width=32, heads=4, depth=2,
                      ngram_shapes=[dict(rows=0, edges=0, order=2)],
                      ngram_weight=.2, cache_weight=.05, copy_weight=.4,
                      ngram_gate=.5, copy_agreement=.5, dropout=0.)
        self.model = Hybrid(config).eval()
        self.x = torch.randint(0, 2048, (2, 24))

    def test_growth_preserves_predictions_and_inherited_tensors(self):
        grown = deepen(self.model, 4).eval()
        with torch.no_grad():
            torch.testing.assert_close(self.model.predict_log_probs(self.x), grown.predict_log_probs(self.x), atol=0, rtol=0)
            for key, tensor in self.model.state_dict().items():
                torch.testing.assert_close(tensor, grown.state_dict()[key], atol=0, rtol=0)
            first = grown.predict_log_probs(self.x)
            changed = self.x.clone()
            changed[:, 12:] += 1
            changed %= 2048
            torch.testing.assert_close(first[:, :12], grown.predict_log_probs(changed)[:, :12], atol=2e-6, rtol=1e-6)

    def test_zero_output_branches_can_learn_with_neural_only_loss(self):
        grown = deepen(self.model, 4).train()
        net = training_component(grown, 'neural')
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3)
        targets = (self.x+17) % 2048
        static = {k:v.clone() for k,v in grown.ngram.state_dict().items()}
        # Hybrid forward must never be called for the neural-only loss.
        grown.forward = lambda ids: self.fail('Training used the hybrid loss')
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(net(self.x).flatten(0, 1), targets.flatten())
            loss.backward()
            for block in net.blocks[2:]:
                self.assertGreater(block.proj.weight.grad.abs().sum().item(), 0.)
                self.assertGreater(block.mlp.down.weight.grad.abs().sum().item(), 0.)
                if step == 1:
                    self.assertGreater(block.qkv.weight.grad.abs().sum().item(), 0.)
                    self.assertGreater(block.mlp.up.weight.grad.abs().sum().item(), 0.)
            optimizer.step()
        for key, tensor in static.items():
            torch.testing.assert_close(tensor, grown.ngram.state_dict()[key], atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
