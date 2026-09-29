"""Train-derived Kneser-Ney probabilities and a strictly window-local neural cache.

No evaluation targets, files, network access or cross-window input state are used.
References and the bounded validation search are documented in fit_hybrid.py.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
import student


class SparseOrder(nn.Module):
    def __init__(self, rows, edges, order, hashed=False):
        super().__init__()
        self.order = order
        self.hashed = hashed
        self.register_buffer('keys', torch.zeros(rows, dtype=torch.int64))
        self.register_buffer('pointers', torch.zeros(rows+1, dtype=torch.int32))
        self.register_buffer('backoff', torch.ones(rows))
        self.register_buffer('targets', torch.zeros(edges, dtype=torch.int16))
        self.register_buffer('mass', torch.zeros(edges))
        if hashed:
            self.register_buffer('context_tokens', torch.zeros(rows, order-1, dtype=torch.int16))

    def add_distribution(self, probabilities, contexts, eligible, ids=None):
        if not self.keys.numel():
            return probabilities
        flat = probabilities.reshape(-1, probabilities.shape[-1])
        packed = contexts.reshape(-1)
        rows = torch.searchsorted(self.keys, packed).clamp_max(self.keys.numel()-1)
        matched = (self.keys[rows] == packed) & eligible.reshape(-1)
        queries = matched.nonzero().flatten()
        rows = rows[queries]
        if self.hashed:
            # Hash lookup is followed by exact token comparison. A collision
            # therefore falls back and cannot borrow another phrase's counts.
            length = ids.shape[1]
            positions = queries % length
            offsets = torch.arange(self.order-2, -1, -1, device=ids.device)
            actual = ids[(queries//length)[:, None], positions[:, None]-offsets]
            exact = (actual == self.context_tokens[rows]).all(-1)
            queries, rows = queries[exact], rows[exact]
        flat[queries] *= self.backoff[rows, None]
        starts = self.pointers[rows].long()
        lengths = (self.pointers[rows+1]-self.pointers[rows]).long()
        ends = lengths.cumsum(0)
        offsets = torch.repeat_interleave(starts-ends+lengths, lengths)
        offsets += torch.arange(offsets.numel(), device=offsets.device)
        destination = torch.repeat_interleave(queries, lengths)*flat.shape[-1]
        destination += self.targets[offsets].long()
        flat.view(-1).index_add_(0, destination, self.mass[offsets])
        return probabilities


class KneserNey(nn.Module):
    def __init__(self, vocab, shapes):
        super().__init__()
        self.vocab = vocab
        self.register_buffer('unigram', torch.full((vocab,), 1/vocab))
        self.orders = nn.ModuleList([SparseOrder(**shape) for shape in shapes])
        # This cache is derived solely from fixed training statistics. It never
        # contains input-window information, and is excluded from checkpoint assets.
        self.register_buffer('_bigram', None, persistent=False)
        self.register_load_state_dict_post_hook(lambda module, incompatible: module.clear_cache())

    def clear_cache(self):
        self._bigram = None

    def forward(self, ids):
        if self._bigram is None:
            table = self.unigram.expand(self.vocab, self.vocab).clone()
            contexts = torch.arange(self.vocab, device=ids.device)
            self._bigram = self.orders[0].add_distribution(table, contexts, torch.ones_like(contexts, dtype=torch.bool))
        batch, length = ids.shape
        probabilities = self._bigram[ids].contiguous()
        contexts = ids.clone()
        hashed_contexts = ids.clone()
        for layer in self.orders[1:]:
            size = layer.order-1
            # Pack only tokens through the current position. Leading incomplete
            # contexts are excluded, never filled with tokens from another window.
            contexts = torch.cat((torch.zeros_like(contexts[:, :1]), contexts[:, :-1]), dim=1)*self.vocab+ids
            hashed_contexts = torch.cat((torch.zeros_like(hashed_contexts[:, :1]), hashed_contexts[:, :-1]), dim=1)*2053+ids
            eligible = (torch.arange(length, device=ids.device) >= size-1).expand(batch, -1)
            layer.add_distribution(probabilities, hashed_contexts if layer.hashed else contexts, eligible, ids)
        return probabilities


class Hybrid(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.context = config['context']
        self.neural = student.build_model(config)
        self.ngram = KneserNey(config['vocab'], config['ngram_shapes'])
        self.ngram_weight = float(config.get('ngram_weight', 0.))
        self.cache_weight = float(config.get('cache_weight', 0.))
        self.cache_theta = float(config.get('cache_theta', 10.))
        self.temperature = float(config.get('neural_temperature', 1.))
        self.copy_weight = float(config.get('copy_weight', 0.))
        # Zero defaults reproduce all existing checkpoints. These are scalar
        # validation-selected settings, never parameters fitted on validation.
        self.ngram_gate = float(config.get('ngram_gate', 0.))
        self.copy_half_life = float(config.get('copy_half_life', 0.))
        self.copy_agreement = float(config.get('copy_agreement', 0.))
        if min(self.ngram_weight, self.cache_weight) < 0 or self.ngram_weight+self.cache_weight >= 1:
            raise ValueError('Mixture weights must be nonnegative and sum to less than one.')
        if self.temperature <= 0 or self.cache_theta <= 0:
            raise ValueError('Temperature and cache sharpness must be positive.')
        if not 0 <= self.copy_weight < 1:
            raise ValueError('Phrase-copy weight must be in [0, 1).')
        if not 0 <= self.ngram_gate <= 2 or not 0 <= self.copy_agreement <= 2:
            raise ValueError('Adaptive exponents must be finite and in [0, 2].')
        if not math.isfinite(self.copy_half_life) or self.copy_half_life < 0:
            raise ValueError('Copy half-life must be finite and nonnegative; 0 disables decay.')

    @staticmethod
    def cache_probabilities(features, ids, theta, vocab=2048):
        batch, length = ids.shape
        probabilities = features.new_zeros(batch, length, vocab, dtype=torch.float32)
        if length <= 1:
            return probabilities
        normalized = F.normalize(features.float(), dim=-1)
        similarities = normalized @ normalized[:, :-1].transpose(-1, -2)
        # Memory entry s stores (h_s, observed token x_{s+1}); allow s < t,
        # hence every consulted token x_{s+1} is already inside the prefix x_0..x_t.
        mask = torch.arange(length-1, device=ids.device)[None, :] < torch.arange(length, device=ids.device)[:, None]
        weights = (theta*similarities).masked_fill(~mask, -1e9).softmax(-1)*mask
        probabilities.scatter_add_(2, ids[:, None, 1:].expand(batch, length, length-1), weights)
        return probabilities

    @staticmethod
    def phrase_copy(ids, vocab=2048, half_life=0., agreement=0.):
        """Suffix matches within this call only; each copied token is already seen."""
        batch, length = ids.shape
        probabilities = torch.zeros(batch, length, vocab, device=ids.device)
        confidence = torch.zeros(batch, length, device=ids.device)
        if length <= 1:
            return probabilities, confidence
        equal = ids[:, :, None] == ids[:, None, :]
        # s is the end of a previous matching context; its continuation x[s+1]
        # is available exactly when s < t. No x[t+1] entry can be consulted.
        allowed = torch.arange(length, device=ids.device)[None, :] < torch.arange(length, device=ids.device)[:, None]
        matched = equal & allowed
        best = torch.zeros_like(matched)
        span = torch.zeros_like(ids)
        for size in range(1, min(8, length)+1):
            if size > 1:
                shifted = torch.zeros_like(equal)
                shifted[:, size-1:, size-1:] = equal[:, :length-size+1, :length-size+1]
                matched = matched & shifted
            present = matched.any(-1)
            best = torch.where(present[:, :, None], matched, best)
            span = torch.where(present, size, span)
        weights = best[:, :, :-1].float()
        if half_life:
            # A function of relative distance only: passing a prefix alone
            # must give the same prediction as that prefix in a longer window.
            positions = torch.arange(length, device=ids.device)
            distance = (positions[:, None]-positions[None, :-1]).clamp_min(0)
            scores = (-distance.float()/half_life)[None, :, :].expand_as(weights)
            scores = scores.masked_fill(~best[:, :, :-1], -torch.inf)
            peak = scores.amax(-1, keepdim=True)
            peak = peak.masked_fill(~torch.isfinite(peak), 0.)
            weights = torch.exp2(scores-peak)
        weights /= weights.sum(-1, keepdim=True).clamp_min(1e-30)
        probabilities.scatter_add_(2, ids[:, None, 1:].expand(batch, length, length-1), weights)
        confidence = ((span.float()-1)/3).clamp(0, 1)
        if agreement:
            # Consensus is computed across observed continuations, without
            # knowing which token the evaluator will ask us to predict.
            confidence *= probabilities.amax(-1).pow(agreement)
        return probabilities, confidence

    @staticmethod
    def statistics_weight(neural_top, ngram_top, base, strength):
        """Bounded odds adjustment from prefix-derived component confidence."""
        if not strength or not base:
            return torch.full_like(neural_top, base)
        ratio = (ngram_top.clamp_min(1e-12)/neural_top.clamp_min(1e-12)).clamp(.25, 4.)
        odds = (base/(1-base))*ratio.pow(strength)
        # Limit the heuristic's effect; the caller also reserves the cache share.
        return (odds/(1+odds)).clamp(max=min(.6, base+.3*(1-base)))

    def predict_log_probs(self, ids):
        features = self.neural.features(ids)
        logits = self.neural.head(features).float()/self.temperature
        if self.ngram_weight == 0 and self.cache_weight == 0 and self.copy_weight == 0:
            return F.log_softmax(logits, dim=-1)
        neural = logits.softmax(-1)
        cache_weight = self.cache_weight*(torch.arange(ids.shape[1], device=ids.device) > 0)[None, :, None]
        ngram = self.ngram(ids) if self.ngram_weight else None
        ngram_weight = self.ngram_weight
        if self.ngram_gate and ngram is not None:
            ngram_weight = self.statistics_weight(neural.amax(-1), ngram.amax(-1),
                                                   self.ngram_weight, self.ngram_gate)[:, :, None]
            # Reserve the fixed cache fraction even for nonstandard configs.
            ngram_weight = torch.minimum(ngram_weight, (1-cache_weight)*.99)
        mixture = neural*(1-ngram_weight-cache_weight)
        if self.ngram_weight:
            mixture = mixture+ngram_weight*ngram
        if self.cache_weight:
            mixture = mixture+cache_weight*self.cache_probabilities(features, ids, self.cache_theta)
        if self.copy_weight:
            copied, confidence = self.phrase_copy(ids, half_life=self.copy_half_life,
                                                  agreement=self.copy_agreement)
            amount = self.copy_weight*confidence[:, :, None]
            mixture = (1-amount)*mixture+amount*copied
        return mixture.clamp_min(1e-30).log()

    def forward(self, ids):
        return self.predict_log_probs(ids)


def build_model(config):
    return Hybrid(config)
