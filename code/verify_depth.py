"""Verify a trained depth branch kept all statistical tables/settings frozen."""
import argparse
import json
from pathlib import Path
import torch
from common import ROOT, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('Output exists')
    original = torch.load(ROOT/'runs/hybrid-adaptive-selected/checkpoint.pt', map_location='cpu', weights_only=True)
    current = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    frozen = all(torch.equal(value, current['model'][key]) for key,value in original['model'].items() if key.startswith('ngram.'))
    settings = all(value == current['config'][key] for key,value in original['config'].items() if key != 'depth')
    changed = any(not torch.equal(value, current['model'][key]) for key,value in original['model'].items() if key.startswith('neural.'))
    new_live = all(current['model'][f'neural.blocks.{i}.{name}'].count_nonzero().item() > 0
                   for i in range(8, current['config']['depth']) for name in ('proj.weight','mlp.down.weight'))
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
                  frozen_statistics_bit_identical=frozen, fixed_settings_unchanged=settings,
                  neural_weights_updated=changed, added_output_branches_learned=new_live,
                  stage_step=current['stage_step'], train_tokens=current['train_tokens'])
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    if not all((frozen, settings, changed, new_live)):
        raise RuntimeError('Training branch verification failed')


if __name__ == '__main__':
    main()
