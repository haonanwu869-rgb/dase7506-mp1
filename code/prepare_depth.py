"""Prepare matched 8/10-layer initial states without learning or changing inference code."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

import torch

from common import ROOT, PROTOCOL, setup, sha
from fit_hybrid import read_split
from hybrid import Hybrid


def deepen(model, depth):
    old_depth = model.config['depth']
    if depth <= old_depth:
        raise ValueError('New depth must exceed the source depth')
    grown = Hybrid(model.config | dict(depth=depth))
    old_state = model.state_dict()
    missing = grown.load_state_dict(old_state, strict=False)
    expected = {name for name in grown.state_dict() if any(
        name.startswith(f'neural.blocks.{i}.') for i in range(old_depth, depth))}
    if set(missing.missing_keys) != expected or missing.unexpected_keys:
        raise ValueError('Unexpected state mismatch during depth expansion')
    for block in grown.neural.blocks[old_depth:]:
        # Keep attention/MLP inputs random, zero only their output projections.
        # Each new block starts as x -> x and receives useful output gradients.
        torch.nn.init.zeros_(block.proj.weight)
        torch.nn.init.zeros_(block.mlp.down.weight)
    grown.train(model.training)
    return grown


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'runs/hybrid-adaptive-selected/checkpoint.pt')
    parser.add_argument('--audit-dir', type=Path, default=ROOT/'runs/depth_audit')
    parser.add_argument('--prefix', default='depth')
    parser.add_argument('--seed', type=int, default=53)
    args = parser.parse_args()
    setup('cpu', 'fp32', 3)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    checkpoint = torch.load(args.source, map_location='cpu', weights_only=True)
    if checkpoint['protocol'] != PROTOCOL or checkpoint['config']['depth'] != 8:
        raise ValueError('This fixed experiment expects the eight-layer MP1 hybrid')
    paths = {depth: ROOT/f'runs/{args.prefix}{depth}-init' for depth in (8, 10)}
    if any(path.exists() for path in paths.values()):
        parser.error('Initialization outputs already exist')
    previous = json.loads((ROOT/'runs/adaptive_audit/summary.json').read_text())
    ancestry = dict(neural_training_targets=previous['neural_training_targets'],
                    neural_train_seconds=previous['inherited_neural_train_seconds'],
                    statistical_fit_targets=previous['inherited_statistical_fit_targets'],
                    statistical_fit_seconds=previous['inherited_statistical_fit_seconds'],
                    prior_mixture_candidates=previous['inherited_selection_candidates']+previous['selection_candidates'],
                    historical_audits=['runs/continuation_audit', 'runs/hybrid_audit', 'runs/adaptive_audit'])
    models = {8: Hybrid(checkpoint['config']).eval()}
    models[8].load_state_dict(checkpoint['model'])
    models[10] = deepen(models[8], 10).eval()
    train, _ = read_split('train')
    x = torch.from_numpy(train[:514]).reshape(2, 257)[:, :-1]
    original = models[8].predict_log_probs(x)
    expanded = models[10].predict_log_probs(x)
    torch.testing.assert_close(original, expanded, atol=0, rtol=0)
    changed = x.clone()
    changed[:, 173:] = (changed[:, 173:]+71) % 2048
    torch.testing.assert_close(expanded[:, :173], models[10].predict_log_probs(changed)[:, :173], atol=2e-6, rtol=1e-6)
    results = []
    for depth, model in models.items():
        path = paths[depth]
        path.mkdir(parents=True)
        state = model.state_dict()
        inherited_identical = all(torch.equal(tensor, state[key]) for key, tensor in checkpoint['model'].items())
        if not inherited_identical:
            raise AssertionError('An inherited parameter/statistic changed')
        output = dict(checkpoint, config=model.config, model=state,
                      cumulative_train_seconds=ancestry['neural_train_seconds'], ancestry=ancestry,
                      initialization=dict(source_sha256=sha(args.source), depth=depth, seed=args.seed,
                                          new_branches_zero_initialized=depth == 10))
        torch.save(output, path/'checkpoint.pt')
        for name in ('hybrid.py', 'student.py'):
            shutil.copy2(ROOT/name, path/name)
        (path/'README.txt').write_text(
            f'MP1 {depth}-layer hybrid initialization; supplied-training-only ancestry.\n'
            'Inference assets: checkpoint.pt, hybrid.py, student.py, README.txt.\n'
            'Use the unchanged evaluator/common.py/data/tokenizer, CPU FP32, 3 threads.\n'
            'metrics.json is an experimental cost record, not an inference dependency.\n', encoding='utf-8')
        assets = sum((path/name).stat().st_size for name in ('checkpoint.pt','hybrid.py','student.py','README.txt'))
        if assets > 64*2**20:
            raise AssertionError('Initial inference assets exceed 64 MiB')
        metrics = dict(command=sys.argv, config=model.config, source=str(args.source), source_sha256=sha(args.source),
                       parameters=sum(p.numel() for p in model.parameters()), asset_bytes=assets,
                       train_tokens=checkpoint['train_tokens'], train_seconds=0.,
                       cumulative_train_seconds=ancestry['neural_train_seconds'], ancestry=ancestry,
                       optimizer_mode='not_created; next stage rebuilds AdamW',
                       inherited_tensors_bit_identical=inherited_identical,
                       initial_predictions_bit_identical=True)
        (path/'metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
        results.append(dict(depth=depth, checkpoint=str(path/'checkpoint.pt'), **metrics))
    record = dict(results=results, smoke_train_prefix_positions=512, additional_training_targets=0,
                  init_seconds=time.perf_counter()-started, model_implementation_sha256=sha(ROOT/'hybrid.py'),
                  neural_implementation_sha256=sha(ROOT/'student.py'))
    (args.audit_dir/'initialization.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(seconds=record['init_seconds'], initial_predictions_bit_identical=True,
                         models=[{k:r[k] for k in ('depth','parameters','asset_bytes')} for r in results])), flush=True)


if __name__ == '__main__':
    main()
