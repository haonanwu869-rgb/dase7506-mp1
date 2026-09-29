"""Fit compact statistics on train only, then a fixed 20-point validation grid.

Modified interpolated Kneser-Ney: Chen & Goodman (1999).
Window-local continuous cache adapted from Grave et al. (2017), using cosine
features and resetting all prefix memory for every independent input window.
No target-probability arrays from validation are saved or served by the model.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from tokenizers import Tokenizer
from common import ROOT, PROTOCOL, sha, setup, windows
from hybrid import Hybrid


def training_statistics(ids, vocab=2048, order=5):
    """Unique predecessor counts for lower orders; ordinary counts at top order."""
    packed = np.asarray(ids, dtype=np.int64).copy()
    counts_by_order = {}
    observations = {}
    for n in range(2, order+1):
        packed = packed[:-1]*vocab+ids[n-1:]
        keys, counts = np.unique(packed, return_counts=True)
        suffix, continuation = np.unique(keys % (vocab**(n-1)), return_counts=True)
        counts_by_order[n-1] = (suffix, continuation)
        observations[n] = len(packed)
        print(json.dumps({'counted_order': n, 'unique_ngrams': len(keys)}), flush=True)
    counts_by_order[order] = (keys, counts)
    unigram = np.full(vocab, 1e-3, dtype=np.float64)
    keys, counts = counts_by_order[1]
    unigram[keys] += counts
    unigram /= unigram.sum()
    state = {'unigram': torch.from_numpy(unigram.astype(np.float32))}
    shapes, details = [], []
    for n in range(2, order+1):
        keys, counts = counts_by_order[n]
        frequency = [int((counts == k).sum()) for k in range(1, 5)]
        n1, n2, n3, n4 = frequency
        y = n1/max(n1+2*n2, 1)
        discounts = [np.clip(1-2*y*n2/max(n1, 1), .05, .99),
                     np.clip(2-3*y*n3/max(n2, 1), .05, 1.99),
                     np.clip(3-4*y*n4/max(n3, 1), .05, 2.99)]
        contexts = keys//vocab
        all_rows, starts = np.unique(contexts, return_index=True)
        totals = np.add.reduceat(counts, starts)
        # Retain all bigrams/trigrams, repeated higher-order grams only.
        keep = counts >= (2 if n >= 4 else 1)
        keys, counts, contexts = keys[keep], counts[keep], contexts[keep]
        rows, starts = np.unique(contexts, return_index=True)
        denominators = totals[np.searchsorted(all_rows, contexts)]
        discounted = np.maximum(counts-np.asarray(discounts)[np.minimum(counts, 3)-1], 0)
        mass = (discounted/denominators).astype(np.float32)
        backoff = (1-np.add.reduceat(mass.astype(np.float64), starts)).astype(np.float32)
        prefix = f'orders.{n-2}.'
        state.update({prefix+'keys': torch.from_numpy(rows),
                      prefix+'pointers': torch.from_numpy(np.r_[starts, len(keys)].astype(np.int32)),
                      prefix+'backoff': torch.from_numpy(backoff),
                      prefix+'targets': torch.from_numpy((keys % vocab).astype(np.int16)),
                      prefix+'mass': torch.from_numpy(mass)})
        shapes.append(dict(rows=len(rows), edges=len(keys), order=n))
        details.append(dict(order=n, rows=len(rows), edges=len(keys), discounts=discounts,
                            counts_of_counts=frequency, minimum_count=2 if n >= 4 else 1))
    return shapes, state, dict(orders=details, ngram_observations=observations,
                              train_targets=len(ids)-1)


def read_split(split):
    manifest = json.loads((ROOT/'data/manifest.json').read_text())
    for name in ('tokenizer.json', f'wikitext_{split}.txt'):
        if sha(ROOT/'data'/name) != manifest['sha256'][name]:
            raise ValueError(f'Protected benchmark file changed: {name}')
    raw = (ROOT/'data'/f'wikitext_{split}.txt').read_bytes()
    tokenizer = Tokenizer.from_file(str(ROOT/'data/tokenizer.json'))
    return np.asarray(tokenizer.encode(raw.decode('utf-8')).ids, dtype=np.int64), len(raw)


@torch.no_grad()
def select_mixture(model, tokens, bytes_count):
    grid = []
    for ngram in (0., .1, .2, .3):
        grid.append(dict(ngram_weight=ngram, cache_weight=0., cache_theta=10.))
        for cache in (.05, .1):
            for theta in (10., 20.):
                grid.append(dict(ngram_weight=ngram, cache_weight=cache, cache_theta=theta))
    totals = np.zeros(len(grid), dtype=np.float64)
    components = dict(neural=0., ngram=0.)
    seen = 0
    model.eval()
    tick = time.perf_counter()
    for batch_index, (x, y) in enumerate(windows(torch.from_numpy(tokens))):
        features = model.neural.features(x)
        logits = model.neural.head(features).float()
        neural = logits.log_softmax(-1).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1).exp()
        ngram = model.ngram(x).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        cache = {theta: model.cache_probabilities(features, x, theta).gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
                 for theta in (10., 20.)}
        valid = y != -100
        nonfirst = (torch.arange(x.shape[1]) > 0).float()[None, :]
        for index, entry in enumerate(grid):
            a, b = entry['ngram_weight'], entry['cache_weight']*nonfirst
            probability = (1-a-b)*neural+a*ngram+b*cache[entry['cache_theta']]
            totals[index] += -probability[valid].double().log().sum().item()
        components['neural'] += -neural[valid].double().log().sum().item()
        components['ngram'] += -ngram[valid].double().clamp_min(1e-30).log().sum().item()
        seen += valid.sum().item()
        if (batch_index+1) % 10 == 0:
            print(json.dumps(dict(validation_batches=batch_index+1, targets=seen, seconds=time.perf_counter()-tick)), flush=True)
    if seen != len(tokens)-1:
        raise AssertionError('Incomplete validation')
    for entry, total in zip(grid, totals):
        entry['bpb'] = float(total/math.log(2)/bytes_count)
    return min(grid, key=lambda row: row['bpb']), dict(grid=grid, targets=seen, utf8_bytes=bytes_count,
             component_bpb={k:v/math.log(2)/bytes_count for k,v in components.items()},
             seconds=time.perf_counter()-tick)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'runs/rope-v4b-continuation-resumed/checkpoint.pt')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/hybrid-v1')
    p.add_argument('--threads', type=int, default=3)
    args = p.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Use a new empty output directory')
    setup('cpu', 'fp32', args.threads)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    ids, train_bytes = read_split('train')
    shapes, state, statistics = training_statistics(ids)
    config = source['config'] | dict(ngram_shapes=shapes, ngram_weight=0., cache_weight=0., cache_theta=10.)
    model = Hybrid(config)
    model.neural.load_state_dict(source['model'])
    model.ngram.load_state_dict(state)
    model.eval()
    model.ngram_weight, model.cache_weight = .2, .1
    with torch.no_grad():
        smoke = torch.from_numpy(ids[:64].reshape(2, 32))
        before = model.predict_log_probs(smoke)
        changed = smoke.clone()
        changed[:, 16:] = (changed[:, 16:]+19) % 2048
        after = model.predict_log_probs(changed)
        torch.testing.assert_close(before[:, :16], after[:, :16], atol=2e-6, rtol=1e-6)
        torch.testing.assert_close(before.logsumexp(-1), torch.zeros(2, 32), atol=1e-6, rtol=0)
        if not torch.isfinite(before).all():
            raise RuntimeError('Nonfinite real-data smoke predictions')
    model.ngram_weight, model.cache_weight = 0., 0.
    print(json.dumps(dict(real_train_prefix_smoke='passed', learning_updates=0)), flush=True)
    fitted_seconds = time.perf_counter()-started
    # Preserve the train-derived artifact before looking at validation, for audit.
    checkpoint = dict(protocol=PROTOCOL, implementation='hybrid', config=config, model=model.state_dict(),
                      seed=source['seed'], train_tokens=source['train_tokens'],
                      statistical_train_targets=len(ids)-1, ancestor_sha256=sha(args.source))
    torch.save(checkpoint, args.run_dir/'unselected.pt')
    inferred_bytes = (args.run_dir/'unselected.pt').stat().st_size + (ROOT/'hybrid.py').stat().st_size + (ROOT/'student.py').stat().st_size
    if inferred_bytes > 64*2**20:
        raise RuntimeError(f'Inference assets exceed budget: {inferred_bytes} bytes')
    print(json.dumps(dict(fitted_seconds=fitted_seconds, inference_bytes=inferred_bytes, statistics=statistics)), flush=True)
    ids, validation_bytes = read_split('validation')
    selected, selection = select_mixture(model, ids, validation_bytes)
    config.update({k:v for k,v in selected.items() if k != 'bpb'})
    checkpoint.update(config=config, validation_bpb=selected['bpb'])
    torch.save(checkpoint, args.run_dir/'checkpoint.pt')
    metrics = dict(command=sys.argv, source=str(args.source), source_sha256=sha(args.source),
                   train_sha256=sha(ROOT/'data/wikitext_train.txt'), tokenizer_sha256=sha(ROOT/'data/tokenizer.json'),
                   source_train_targets=source['train_tokens'], train_statistics_targets=statistics['train_targets'],
                   fit_seconds=fitted_seconds, process_seconds=time.perf_counter()-started,
                   threads=args.threads, precision='fp32', statistics=statistics, selection=selection,
                   selected=selected, inference_bytes=inferred_bytes, architecture='fixed neural + modified Kneser-Ney + causal window cache')
    (args.run_dir/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(dict(selected=selected, seconds=metrics['process_seconds'])), flush=True)


if __name__ == '__main__':
    main()
