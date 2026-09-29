import contextlib
import io
import unittest
import numpy as np
import torch
from fit_hybrid import training_statistics
from hybrid import Hybrid
from refine_hybrid import phrase_statistics


class HybridContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(91)
        # Temporary test-only counts, not used by the real experiment.
        sequence = np.tile(np.asarray([10, 20, 30, 10, 20, 40, 50, 60]), 80)
        with contextlib.redirect_stdout(io.StringIO()):
            shapes, cls.counts, _ = training_statistics(sequence)
        cls.config = dict(vocab=2048, width=32, heads=4, depth=2, context=256,
                          ngram_shapes=shapes, ngram_weight=.2, cache_weight=.1, cache_theta=10., copy_weight=.3)

    def setUp(self):
        self.model = Hybrid(self.config).eval()
        self.model.ngram.load_state_dict(self.counts)
        self.ids = torch.tensor([[10, 20, 30, 10, 20, 40, 50, 60, 10, 20, 70, 80],
                                 [30, 20, 10, 30, 20, 40, 50, 60, 30, 20, 70, 80]])

    def test_no_future_tokens_and_no_cross_window_state(self):
        with torch.no_grad():
            first = self.model.predict_log_probs(self.ids)
            changed = self.ids.clone()
            changed[:, 7:] = (changed[:, 7:]+123) % 2048
            second = self.model.predict_log_probs(changed)
            torch.testing.assert_close(first[:, :7], second[:, :7], atol=1e-6, rtol=1e-6)
            self.model.predict_log_probs((self.ids+17) % 2048)
            torch.testing.assert_close(first, self.model.predict_log_probs(self.ids), atol=0, rtol=0)
            # Prefixes passed alone produce the same prediction as full windows.
            for length in (1, 2, 3, 7):
                prefix = self.model.predict_log_probs(self.ids[:, :length])
                torch.testing.assert_close(first[:, :length], prefix, atol=2e-6, rtol=1e-6)

    def test_normalized_independent_and_finite(self):
        with torch.no_grad():
            logp = self.model.predict_log_probs(self.ids)
            self.assertTrue(torch.isfinite(logp).all())
            torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(self.ids.shape), atol=1e-6, rtol=0)
            torch.testing.assert_close(logp[:1], self.model.predict_log_probs(self.ids[:1]), atol=2e-6, rtol=1e-6)
            ng = self.model.ngram(self.ids)
            torch.testing.assert_close(ng.sum(-1), torch.ones(self.ids.shape), atol=1e-6, rtol=0)

    def test_cache_contains_only_observed_prefix_tokens(self):
        with torch.no_grad():
            cache = self.model.cache_probabilities(self.model.neural.features(self.ids), self.ids, 10.)
        self.assertEqual(cache[:, 0].count_nonzero().item(), 0)
        for row in range(len(self.ids)):
            for position in range(1, self.ids.shape[1]):
                observed = set(self.ids[row, 1:position+1].tolist())
                supported = set(cache[row, position].nonzero().flatten().tolist())
                self.assertTrue(supported <= observed)
                self.assertAlmostEqual(cache[row, position].sum().item(), 1., places=6)

    def test_reload_invalidates_derived_bigram_cache(self):
        first = self.model.ngram(self.ids).clone()
        self.assertIsNotNone(self.model.ngram._bigram)
        self.model.ngram.load_state_dict(self.counts)
        self.assertIsNone(self.model.ngram._bigram)
        torch.testing.assert_close(first, self.model.ngram(self.ids), atol=0, rtol=0)

    def test_adaptive_mixture_is_causal_normalized_and_window_local(self):
        self.model.ngram_gate = 1.
        self.model.copy_half_life = 64.
        self.model.copy_agreement = .5
        with torch.no_grad():
            for cache_weight in (.05, .79):
                # Also exercise the reserved-cache bound near the valid limit.
                self.model.cache_weight = cache_weight
                full = self.model.predict_log_probs(self.ids)
                self.assertTrue(torch.isfinite(full).all())
                torch.testing.assert_close(full.logsumexp(-1), torch.zeros(self.ids.shape), atol=1e-6, rtol=0)
                changed = self.ids.clone()
                changed[:, 7:] += 123
                torch.testing.assert_close(full[:, :7], self.model.predict_log_probs(changed)[:, :7], atol=2e-6, rtol=1e-6)
                for length in (1, 2, 3, 7):
                    torch.testing.assert_close(full[:, :length], self.model.predict_log_probs(self.ids[:, :length]), atol=2e-6, rtol=1e-6)
                self.model.predict_log_probs(changed.flip(0))
                torch.testing.assert_close(full, self.model.predict_log_probs(self.ids), atol=0, rtol=0)
                torch.testing.assert_close(full[:1], self.model.predict_log_probs(self.ids[:1]), atol=2e-6, rtol=1e-6)

    def test_copy_recency_and_conflicting_continuations(self):
        x = torch.tensor([[10, 20, 30, 91, 10, 20, 40, 92, 10, 20]])
        plain, _ = self.model.phrase_copy(x)
        recent, confidence = self.model.phrase_copy(x, half_life=2., agreement=.5)
        self.assertEqual(plain[0, -1, 30].item(), .5)
        self.assertEqual(plain[0, -1, 40].item(), .5)
        self.assertAlmostEqual(recent[0, -1, 40].item(), .8, places=6)
        self.assertLess(confidence[0, -1].item(), 1/3)
        torch.testing.assert_close(recent[0, -1].sum(), torch.tensor(1.), atol=1e-6, rtol=0)
        for length in range(1, x.shape[1]):
            pp, cc = self.model.phrase_copy(x[:, :length], half_life=2., agreement=.5)
            torch.testing.assert_close(pp, recent[:, :length], atol=0, rtol=0)
            torch.testing.assert_close(cc, confidence[:, :length], atol=0, rtol=0)
        # Extremely fast decay still normalizes around the most recent match.
        fast, _ = self.model.phrase_copy(x, half_life=.01)
        self.assertEqual(fast[0, -1, 40].item(), 1.)

    def test_phrase_copy_uses_observed_continuations(self):
        x = torch.tensor([[10, 20, 30, 40, 10, 20]])
        probabilities, confidence = self.model.phrase_copy(x)
        self.assertEqual(probabilities[0, 5, 30].item(), 1.)
        self.assertAlmostEqual(confidence[0, 5].item(), 1/3, places=6)
        for position in range(x.shape[1]):
            support = set(probabilities[0, position].nonzero().flatten().tolist())
            self.assertTrue(support <= set(x[0, :position+1].tolist()))
        for length in range(1, x.shape[1]):
            prefix_p, prefix_c = self.model.phrase_copy(x[:, :length])
            torch.testing.assert_close(prefix_p, probabilities[:, :length], atol=0, rtol=0)
            torch.testing.assert_close(prefix_c, confidence[:, :length], atol=0, rtol=0)

    def test_long_phrase_exact_check_and_causality(self):
        sequence = np.tile(np.asarray([10, 20, 30, 10, 20, 40, 50, 60]), 80)
        with contextlib.redirect_stdout(io.StringIO()):
            shapes, state, details = phrase_statistics(sequence)
        model = Hybrid(self.config | {'ngram_shapes': self.config['ngram_shapes']+shapes}).eval()
        buffers = {'ngram.'+k:v for k,v in self.counts.items()} | state
        model.ngram.load_state_dict({k.removeprefix('ngram.'):v for k,v in buffers.items()})
        self.assertTrue(all(row['hash_collisions'] == 0 for row in details))
        x = torch.from_numpy(sequence[:48].reshape(2, 24))
        with torch.no_grad():
            first = model.predict_log_probs(x)
            changed = x.clone()
            changed[:, 13:] += 123
            torch.testing.assert_close(first[:, :13], model.predict_log_probs(changed)[:, :13], atol=2e-6, rtol=1e-6)
            torch.testing.assert_close(first.logsumexp(-1), torch.zeros(2, 24), atol=1e-6, rtol=0)
        layer = model.ngram.orders[-1]
        probabilities = torch.full((1, 12, 2048), 1/2048)
        # Force a hash hit with a different exact prefix; it must leave the
        # lower-order distribution untouched instead of borrowing phrase counts.
        fake_ids = torch.zeros(1, 12, dtype=torch.long)
        fake_keys = torch.full_like(fake_ids, layer.keys[0].item())
        eligible = torch.arange(12)[None, :] >= layer.order-2
        actual = layer.add_distribution(probabilities.clone(), fake_keys, eligible, fake_ids)
        torch.testing.assert_close(probabilities, actual, atol=0, rtol=0)


if __name__ == '__main__':
    unittest.main()
