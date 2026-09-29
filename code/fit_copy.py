"""Final bounded six-point check of causal, window-local phrase continuation."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch
from common import ROOT, setup, sha, windows
from fit_hybrid import read_split
from hybrid import Hybrid


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'runs/hybrid-v2-long/checkpoint.pt')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/hybrid-v3-copy')
    p.add_argument('--threads', type=int, default=3)
    args = p.parse_args()
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Use a new empty directory')
    setup('cpu', 'fp32', args.threads)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=True)
    model = Hybrid(checkpoint['config']).eval()
    model.load_state_dict(checkpoint['model'])
    model.copy_weight = 0.
    ids, byte_count = read_split('validation')
    choices = (0., .1, .2, .3, .4, .6)
    totals = [0.]*len(choices)
    count = 0
    started = time.perf_counter()
    for batch_index, (x, y) in enumerate(windows(torch.from_numpy(ids))):
        targets = y.clamp_min(0).unsqueeze(-1)
        base = model.predict_log_probs(x).gather(-1, targets).squeeze(-1).exp()
        copied, confidence = model.phrase_copy(x)
        copied = copied.gather(-1, targets).squeeze(-1)
        valid = y != -100
        for index, strength in enumerate(choices):
            amount = strength*confidence
            probability = (1-amount)*base+amount*copied
            totals[index] += -probability[valid].double().log().sum().item()
        count += valid.sum().item()
        if (batch_index+1) % 10 == 0:
            print(json.dumps(dict(validation_batches=batch_index+1, targets=count, seconds=time.perf_counter()-started)), flush=True)
    if count != len(ids)-1:
        raise AssertionError('Incomplete validation')
    grid = [dict(copy_weight=w, bpb=nll/math.log(2)/byte_count) for w,nll in zip(choices, totals)]
    selected = min(grid, key=lambda row: row['bpb'])
    checkpoint['config'] = checkpoint['config'] | dict(copy_weight=selected['copy_weight'])
    checkpoint['validation_bpb'] = selected['bpb']
    torch.save(checkpoint, args.run_dir/'checkpoint.pt')
    metrics = dict(command=sys.argv, source=str(args.source), source_sha256=sha(args.source),
                   additional_training_targets=0, validation_targets=count, utf8_bytes=byte_count,
                   grid=grid, selected=selected, seconds=time.perf_counter()-started)
    (args.run_dir/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
