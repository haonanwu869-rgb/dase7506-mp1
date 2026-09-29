"""Full validation controls with unchanged neural weights and scorer."""
import argparse
import json
from pathlib import Path
import sys
import time
import torch
from common import ROOT, setup, sha
from evaluate import score
from fit_hybrid import read_split
from hybrid import Hybrid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--threads', type=int, default=3)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output already exists')
    device, _ = setup('cpu', 'fp32', args.threads)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = Hybrid(checkpoint['config']).eval()
    model.load_state_dict(checkpoint['model'])
    original = torch.load(ROOT/'runs/rope-v4b-continuation-resumed/checkpoint.pt', map_location='cpu', weights_only=True)
    unchanged = all(torch.equal(tensor, model.neural.state_dict()[key]) for key, tensor in original['model'].items())
    if not unchanged:
        raise AssertionError('Neural backbone weights changed unexpectedly')
    ids, byte_count = read_split('validation')
    ids = torch.from_numpy(ids)
    config = checkpoint['config']
    variants = [
        ('neural_only', dict(ngram_weight=0., cache_weight=0., copy_weight=0.)),
        ('without_train_statistics', dict(ngram_weight=0.)),
        ('without_neural_cache', dict(cache_weight=0.)),
        ('without_phrase_copy', dict(copy_weight=0.)),
    ]
    results = []
    for name, changes in variants:
        for key in ('ngram_weight', 'cache_weight', 'copy_weight'):
            setattr(model, key, float(changes.get(key, config.get(key, 0.))))
        result = score(model, ids, byte_count, device, 'fp32')
        result.pop('window_nll_nats')
        results.append(dict(name=name, changes=changes, **result))
        print(json.dumps(results[-1]), flush=True)
        # Write after every completed control so an interruption loses no scores.
        args.output.write_text(json.dumps(dict(command=sys.argv, checkpoint_sha256=sha(args.checkpoint),
                                              neural_backbone_bit_identical=unchanged, results=results), indent=2)+'\n')


if __name__ == '__main__':
    main()
