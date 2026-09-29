"""One bounded extension: repeated train phrases up to order 10, then calibration.

Use exact context verification after a 64-bit hash lookup. Training collisions
are explicitly detected and dropped. No evaluation prefix is persisted.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import torch
from common import ROOT, sha, setup, windows
from fit_hybrid import read_split
from hybrid import Hybrid


def phrase_statistics(ids, start_order=6, max_order=10):
    hashed = ids.copy()
    state, shapes, details = {}, [], []
    for width in range(2, max_order):
        with np.errstate(over='ignore'):
            hashed = hashed[:-1]*np.int64(2053)+ids[width-1:]
        order = width+1
        if order < start_order:
            continue
        tick = time.perf_counter()
        contexts = hashed[:-1]
        row_keys, first, inverse, totals = np.unique(contexts, return_index=True, return_inverse=True, return_counts=True)
        representative = first[inverse]
        conflict = np.zeros(len(contexts), dtype=bool)
        for offset in range(width):
            conflict |= ids[offset:offset+len(contexts)] != ids[representative+offset]
        bad_keys = np.unique(contexts[conflict])
        pairs = np.empty(len(contexts), dtype=[('context', '<i8'), ('target', '<i2')])
        pairs['context'] = contexts
        pairs['target'] = ids[width:]
        grams, counts = np.unique(pairs, return_counts=True)
        n1, n2, n3, n4 = [int((counts == k).sum()) for k in range(1, 5)]
        y = n1/max(n1+2*n2, 1)
        discounts = np.asarray([np.clip(1-2*y*n2/max(n1, 1), .05, .99),
                                np.clip(2-3*y*n3/max(n2, 1), .05, 1.99),
                                np.clip(3-4*y*n4/max(n3, 1), .05, 2.99)])
        keep = (counts >= 2) & ~np.isin(grams['context'], bad_keys)
        grams, counts = grams[keep], counts[keep]
        rows, starts = np.unique(grams['context'], return_index=True)
        lookup = np.searchsorted(row_keys, grams['context'])
        mass = ((counts-discounts[np.minimum(counts, 3)-1])/totals[lookup]).astype(np.float32)
        backoff = (1-np.add.reduceat(mass.astype(np.float64), starts)).astype(np.float32)
        examples = first[np.searchsorted(row_keys, rows)]
        exact_contexts = ids[examples[:, None]+np.arange(width)].astype(np.int16)
        prefix = f'ngram.orders.{order-2}.'
        state.update({prefix+'keys': torch.from_numpy(rows.copy()),
                      prefix+'pointers': torch.from_numpy(np.r_[starts, len(grams)].astype(np.int32)),
                      prefix+'backoff': torch.from_numpy(backoff),
                      prefix+'targets': torch.from_numpy(grams['target'].copy()),
                      prefix+'mass': torch.from_numpy(mass),
                      prefix+'context_tokens': torch.from_numpy(exact_contexts)})
        shapes.append(dict(order=order, rows=len(rows), edges=len(grams), hashed=True))
        detail = dict(order=order, rows=len(rows), edges=len(grams), hash_collisions=len(bad_keys),
                      train_observations=len(contexts), discounts=discounts.tolist(), seconds=time.perf_counter()-tick)
        details.append(detail)
        print(json.dumps(detail), flush=True)
    return shapes, state, details


@torch.no_grad()
def calibrate(model, ids, byte_count):
    # A fixed 3 x 3 x 2 grid: no additional adaptive search or fitted gates.
    grid = [dict(neural_temperature=t, ngram_weight=a, cache_weight=b, cache_theta=10.)
            for t in (.9, 1., 1.1) for a in (.15, .2, .25) for b in (.05, .1)]
    totals = np.zeros(len(grid), dtype=np.float64)
    count = 0
    model.eval()
    started = time.perf_counter()
    for index, (x, y) in enumerate(windows(torch.from_numpy(ids))):
        features = model.neural.features(x)
        logits = model.neural.head(features).float()
        targets = y.clamp_min(0).unsqueeze(-1)
        neural = {t:(logits/t).log_softmax(-1).gather(-1, targets).squeeze(-1).exp() for t in (.9, 1., 1.1)}
        ng = model.ngram(x).gather(-1, targets).squeeze(-1)
        cache = model.cache_probabilities(features, x, 10.).gather(-1, targets).squeeze(-1)
        nonfirst = (torch.arange(x.shape[1]) > 0).float()[None, :]
        valid = y != -100
        count += valid.sum().item()
        for i, row in enumerate(grid):
            a, b = row['ngram_weight'], row['cache_weight']*nonfirst
            probability = (1-a-b)*neural[row['neural_temperature']]+a*ng+b*cache
            totals[i] += -probability[valid].double().log().sum().item()
        if (index+1) % 10 == 0:
            print(json.dumps(dict(validation_batches=index+1, targets=count, seconds=time.perf_counter()-started)), flush=True)
    if count != len(ids)-1:
        raise AssertionError('Incomplete validation')
    for row, total in zip(grid, totals):
        row['bpb'] = float(total/math.log(2)/byte_count)
    return min(grid, key=lambda row: row['bpb']), dict(grid=grid, targets=count, utf8_bytes=byte_count, seconds=time.perf_counter()-started)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'runs/hybrid-v1/checkpoint.pt')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/hybrid-v2-long')
    p.add_argument('--threads', type=int, default=3)
    args = p.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Use a new empty output directory')
    setup('cpu', 'fp32', args.threads)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    ids, _ = read_split('train')
    shapes, state, details = phrase_statistics(ids)
    config = source['config'] | dict(ngram_shapes=source['config']['ngram_shapes']+shapes)
    model = Hybrid(config).eval()
    model.load_state_dict(source['model'] | state)
    with torch.no_grad():
        x = torch.from_numpy(ids[:128].reshape(2, 64))
        logp = model.predict_log_probs(x)
        torch.testing.assert_close(logp.logsumexp(-1), torch.zeros(2, 64), atol=1e-6, rtol=0)
        changed = x.clone()
        changed[:, 32:] = (changed[:, 32:]+13) % 2048
        torch.testing.assert_close(logp[:, :32], model.predict_log_probs(changed)[:, :32], atol=2e-6, rtol=1e-6)
    checkpoint = {k:v for k,v in source.items() if k != 'validation_bpb'}
    checkpoint.update(config=config, model=model.state_dict(), long_phrase_train_targets=len(ids)-1)
    torch.save(checkpoint, args.run_dir/'unselected.pt')
    asset_bytes = (args.run_dir/'unselected.pt').stat().st_size + sum((ROOT/name).stat().st_size for name in ('student.py', 'hybrid.py'))
    if asset_bytes > 64*2**20:
        raise RuntimeError(f'Asset budget exceeded: {asset_bytes} bytes')
    fit_seconds = time.perf_counter()-started
    print(json.dumps(dict(smoke='passed', fit_seconds=fit_seconds, asset_bytes=asset_bytes)), flush=True)
    validation, byte_count = read_split('validation')
    selected, selection = calibrate(model, validation, byte_count)
    config.update({k:v for k,v in selected.items() if k != 'bpb'})
    checkpoint.update(config=config, validation_bpb=selected['bpb'])
    torch.save(checkpoint, args.run_dir/'checkpoint.pt')
    metrics = dict(command=sys.argv, source=str(args.source), source_sha256=sha(args.source),
                   fit_seconds=fit_seconds, process_seconds=time.perf_counter()-started, train_targets=len(ids)-1,
                   long_orders=details, selection=selection, selected=selected, asset_bytes=asset_bytes,
                   inherited_statistics_targets=source.get('statistical_train_targets', 0),
                   inherited_neural_train_targets=source['train_tokens'])
    (args.run_dir/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(dict(selected=selected, seconds=metrics['process_seconds'])), flush=True)


if __name__ == '__main__':
    main()
