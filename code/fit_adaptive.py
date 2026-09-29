"""Fixed 12-choice validation comparison; no gradient updates or table refitting.

All gates depend solely on probabilities/observed tokens in the current prefix.
Candidate settings are declared before scoring. Only aggregate validation losses
are retained, never validation target probabilities or answer lookup tables.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys
import time

import torch

from common import ROOT, PROTOCOL, setup, sha, windows
from fit_hybrid import read_split
from hybrid import Hybrid


def candidates():
    return [dict(ngram_gate=gate, copy_half_life=half_life, copy_agreement=agreement)
            for gate, (half_life, agreement) in itertools.product(
                (0., .5, 1.), ((0., 0.), (0., .5), (64., 0.), (64., .5)))]


def apply_settings(model, settings):
    for name, value in settings.items():
        setattr(model, name, value)


@torch.no_grad()
def target_probabilities(model, x, y, choices):
    """Reuse components; gather y only after all input-dependent gates are made."""
    features = model.neural.features(x)
    neural = (model.neural.head(features).float()/model.temperature).softmax(-1)
    ngram = model.ngram(x)
    neural_top, ngram_top = neural.amax(-1), ngram.amax(-1)
    cache = model.cache_probabilities(features, x, model.cache_theta)
    target = y.clamp_min(0).unsqueeze(-1)
    pn, pg, pc = [p.gather(-1, target).squeeze(-1) for p in (neural, ngram, cache)]
    del neural, ngram, cache, features
    cache_weight = model.cache_weight*(torch.arange(x.shape[1], device=x.device) > 0)[None, :]
    copied = {}
    for half_life in sorted({row['copy_half_life'] for row in choices}):
        distribution, confidence = model.phrase_copy(x, half_life=half_life)
        copied[half_life] = (distribution.gather(-1, target).squeeze(-1),
                             confidence, distribution.amax(-1))
    result = []
    for row in choices:
        ng_weight = model.statistics_weight(neural_top, ngram_top, model.ngram_weight, row['ngram_gate'])
        if row['ngram_gate']:
            ng_weight = torch.minimum(ng_weight, (1-cache_weight)*.99)
        base = pn*(1-ng_weight-cache_weight)+pg*ng_weight+pc*cache_weight
        copy_p, confidence, agreement = copied[row['copy_half_life']]
        if row['copy_agreement']:
            confidence = confidence*agreement.pow(row['copy_agreement'])
        amount = model.copy_weight*confidence
        result.append((1-amount)*base+amount*copy_p)
    return result


@torch.no_grad()
def smoke_check(model, choices):
    # Real training prefix, without weight updates. No test data are requested.
    train, _ = read_split('train')
    batch = torch.from_numpy(train[:257]).unsqueeze(0)
    x, y = batch[:, :-1], batch[:, 1:]
    shared = target_probabilities(model, x, y, choices)
    maximum_normalization_error = 0.
    maximum_probability_error = 0.
    for settings, probability in zip(choices, shared):
        apply_settings(model, settings)
        logp = model.predict_log_probs(x)
        maximum_normalization_error = max(maximum_normalization_error, logp.logsumexp(-1).abs().max().item())
        # FP32 sums through multiple discounted tables can differ by a few
        # ulps; this is still much stricter than the fixed evaluator's 1e-3.
        torch.testing.assert_close(logp.logsumexp(-1), torch.zeros_like(x, dtype=torch.float32),
                                   atol=3e-6, rtol=0)
        direct = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1).exp()
        maximum_probability_error = max(maximum_probability_error, (direct-probability).abs().max().item())
        torch.testing.assert_close(direct, probability, atol=2e-7, rtol=3e-6)
    return dict(train_targets_checked=256, settings_checked=len(choices), no_updates=True,
                maximum_normalization_error=maximum_normalization_error,
                maximum_probability_error=maximum_probability_error)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/hybrid-selected/checkpoint.pt')
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/hybrid-adaptive')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        parser.error('Use a new empty run directory; existing results are preserved')
    setup('cpu', 'fp32', args.threads)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL:
        raise ValueError('Wrong protocol')
    model = Hybrid(checkpoint['config']).eval()
    model.load_state_dict(checkpoint['model'])
    choices = candidates()
    plan = dict(command=sys.argv, source=str(args.source), source_sha256=sha(args.source),
                grid=choices, maximum_candidates=12, neural_updates=0, statistic_refits=0,
                split='validation', device='cpu', precision='fp32', threads=args.threads,
                targets_for_selection='all validation targets; training prefix used for smoke only')
    (args.run_dir/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    smoke_started = time.perf_counter()
    smoke = smoke_check(model, choices)
    smoke['seconds'] = time.perf_counter()-smoke_started
    print(json.dumps(dict(smoke=smoke)), flush=True)
    ids, byte_count = read_split('validation')
    totals = [0.]*len(choices)
    count = 0
    validation_started = time.perf_counter()
    for batch_index, (x, y) in enumerate(windows(torch.from_numpy(ids))):
        valid = y != -100
        for index, probability in enumerate(target_probabilities(model, x, y, choices)):
            totals[index] += -probability[valid].double().clamp_min(1e-30).log().sum().item()
        count += valid.sum().item()
        if (batch_index+1) % 10 == 0:
            print(json.dumps(dict(validation_batches=batch_index+1, targets=count,
                                 seconds=time.perf_counter()-validation_started)), flush=True)
    if count != len(ids)-1:
        raise AssertionError('Validation incomplete')
    grid = [row | dict(bpb=nll/math.log(2)/byte_count) for row,nll in zip(choices, totals)]
    selected = min(grid, key=lambda row: row['bpb'])
    settings = {key: selected[key] for key in choices[0]}
    checkpoint['config'] = checkpoint['config'] | settings
    checkpoint['validation_bpb'] = selected['bpb']
    # Every state tensor is inherited untouched, including all training tables.
    torch.save(checkpoint, args.run_dir/'checkpoint.pt')
    metrics = dict(**plan, smoke=smoke, results=grid, selected=selected,
                   reference_bpb=grid[0]['bpb'], delta_bpb=selected['bpb']-grid[0]['bpb'],
                   validation_targets=count, utf8_bytes=byte_count,
                   additional_training_targets=0,
                   inherited_neural_training_targets=checkpoint.get('train_tokens'),
                   validation_seconds=time.perf_counter()-validation_started,
                   process_seconds=time.perf_counter()-started)
    (args.run_dir/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
