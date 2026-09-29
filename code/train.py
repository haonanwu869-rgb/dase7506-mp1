"""Training with validation selection, restartable state and optional EMA."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import PROTOCOL, ROOT, autocast, load_data, make_model, setup, sha
from evaluate import score


def save_atomic(obj, path):
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def rng_state(generator):
    ns = np.random.get_state()
    return dict(torch=torch.get_rng_state(), sampler=generator.get_state(),
                python=random.getstate(), numpy=(ns[0], ns[1].tolist(), *ns[2:]),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state, generator):
    torch.set_rng_state(state['torch'])
    generator.set_state(state['sampler'])
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.array(ns[1], dtype=np.uint32), *ns[2:]))
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def learning_rate(step, steps, lr, min_lr, warmup):
    progress = max(0, step-warmup) / max(1, steps-warmup-1)
    value = min_lr + (lr-min_lr)*0.5*(1+math.cos(math.pi*progress))
    return value * min(1., (step+1)/max(1, warmup))


def training_component(model, component):
    """Select the loss-producing module; evaluation always uses the full model."""
    if component == 'model':
        return model
    if component == 'neural' and hasattr(model, 'neural'):
        return model.neural
    raise ValueError('The neural training component requires a hybrid model')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--implementation', default='student')
    p.add_argument('--train-component', choices=['model', 'neural'], default='model',
                   help='neural trains hybrid.neural with ordinary cross-entropy; validates the full hybrid.')
    p.add_argument('--config', type=Path, default=ROOT/'configs/baseline.json')
    p.add_argument('--run-dir', type=Path, default=ROOT/'runs/baseline-s17')
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['auto', 'fp32', 'bf16'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--steps', type=int, default=1200, help='Stage horizon, including restored updates.')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--eval-every', type=int, default=0)
    p.add_argument('--checkpoint-every', type=int, default=100,
                   help='Save latest complete training state between validations; 0 disables.')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--min-lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=100)
    p.add_argument('--dropout', type=float)
    p.add_argument('--ema-decay', type=float, default=0., help='0 disables EMA.')
    p.add_argument('--patience', type=int, default=0, help='Validation events without improvement before stopping; 0 disables.')
    p.add_argument('--resume', type=Path, help='Restore training_state.pt into a NEW output directory.')
    p.add_argument('--resume-horizon', type=int,
                   help='Explicitly shorten a resumed stage and replan cosine decay to its new endpoint; optimizer/RNG stay restored.')
    p.add_argument('--init-from', type=Path, help='Initialize a new stage from weights; rebuild optimizer.')
    p.add_argument('--stop-after', type=int, help='Save and stop for restart checks; preserve schedule horizon.')
    args = p.parse_args()
    if args.resume and args.init_from:
        p.error('--resume and --init-from are mutually exclusive')
    if args.resume_horizon is not None and not args.resume:
        p.error('--resume-horizon requires --resume')
    if args.steps < 1 or args.batch_size < 1 or args.warmup < 0 or args.eval_every < 0 or args.patience < 0 or args.checkpoint_every < 0:
        p.error('Invalid step, batch, warmup, evaluation or patience value')
    if not 0 <= args.ema_decay < 1 or not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        p.error('Invalid EMA decay or learning rates')
    if args.stop_after is not None and not 1 <= args.stop_after <= args.steps:
        p.error('--stop-after must be within the stage')
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        p.error('Use a new, empty --run-dir; existing results are never overwritten.')
    device, precision = setup(args.device, args.precision, args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    source_path = args.resume or args.init_from
    source = torch.load(source_path, map_location='cpu', weights_only=True) if source_path else None
    if source and source['protocol'] != PROTOCOL:
        p.error('Checkpoint protocol mismatch')
    recipe = dict(steps=args.steps, batch_size=args.batch_size, lr=args.lr, min_lr=args.min_lr,
                  warmup=args.warmup, ema_decay=args.ema_decay, eval_every=args.eval_every,
                  patience=args.patience, seed=args.seed, precision=precision, threads=args.threads,
                  device=str(device))
    if args.train_component != 'model':
        recipe['train_component'] = args.train_component
    if args.resume and (source.get('recipe') != recipe or 'optimizer' not in source):
        p.error('Exact resume needs training_state.pt and the same recipe/device; use --init-from for a new stage.')
    horizon = source.get('schedule_horizon', args.steps) if args.resume else args.steps
    if args.resume_horizon is not None:
        if not source['stage_step'] < args.resume_horizon <= horizon:
            p.error('The replanned horizon must exceed the restored step and cannot extend training.')
        horizon = args.resume_horizon
    config = dict(source['config']) if source else json.loads(args.config.read_text())
    if args.dropout is not None:
        if args.resume and args.dropout != config.get('dropout', .1):
            p.error('Cannot change dropout during exact resume')
        config['dropout'] = args.dropout
    implementation = source['implementation'] if source else args.implementation
    model, implementation_sha = make_model(implementation, config, device)
    if args.resume and source['implementation_sha256'] != implementation_sha:
        p.error('Implementation changed; exact resume rejected')
    if source:
        model.load_state_dict(source['model'])
    loss_model = training_component(model, args.train_component)
    optimizer = torch.optim.AdamW(loss_model.parameters(), lr=args.lr, weight_decay=.1)
    ema = copy.deepcopy(model).eval().requires_grad_(False) if args.ema_decay else None
    ema_loss_model = training_component(ema, args.train_component) if ema else None
    generator = torch.Generator().manual_seed(args.seed)
    stage_step = 0
    ancestor_targets = source.get('train_tokens', 0) if source else 0
    ancestor_seconds = source.get('cumulative_train_seconds', 0.) if source else 0.
    if source_path and (source_path.parent/'metrics.json').exists():
        prior = json.loads((source_path.parent/'metrics.json').read_text())
        ancestor_seconds = prior.get('cumulative_train_seconds', prior.get('train_seconds', ancestor_seconds))
    ancestry = source.get('ancestry', {}) if source else {}
    sampling_digest = hashlib.sha256(b'').hexdigest()
    sampling_digest_start = 0
    history, validations = [], []
    best_bpb, best_kind, best_step, bad = float('inf'), None, 0, 0
    train_seconds = validation_seconds = 0.
    best_states = {}
    if args.resume:
        optimizer.load_state_dict(source['optimizer'])
        if ema:
            ema.load_state_dict(source['ema'])
        stage_step = source['stage_step']
        ancestor_targets = source['ancestor_targets']
        ancestor_seconds = source['ancestor_train_seconds']
        history, validations = source['history'], source['validation_history']
        best_bpb, best_kind, best_step, bad = source['selection']
        best_states = source['best_states']
        train_seconds, validation_seconds = source['train_seconds'], source['validation_seconds']
        sampling_digest = source.get('sampling_digest', sampling_digest)
        sampling_digest_start = source.get('sampling_digest_start', stage_step)
        restore_rng(source['rng'], generator)
    data = load_data()
    tokens = data['train'][0].to(device)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    def inference(weights, step, kind, bpb):
        return dict(protocol=PROTOCOL, implementation=implementation, config=config,
                    model=weights, seed=args.seed,
                    train_tokens=ancestor_targets+step*args.batch_size*256,
                    stage_step=step, weight_kind=kind, validation_bpb=bpb,
                    cumulative_train_seconds=ancestor_seconds+train_seconds, ancestry=ancestry)

    def cpu_state(net):
        # Preserve tied storage in inference artifacts.
        return copy.deepcopy(net).cpu().state_dict()

    def validate(step):
        nonlocal best_bpb, best_kind, best_step, bad, validation_seconds
        improved = False
        for kind, net in [('raw', model)] + ([('ema', ema)] if ema else []):
            result = score(net, *data['validation'], device, 'fp32')
            result.pop('window_nll_nats')
            validation_seconds += result['seconds']
            validations.append(dict(step=step, kind=kind, **result))
            print(json.dumps({'validation': validations[-1]}), flush=True)
            old = best_states.get(kind)
            if old is None or result['bpb'] < old['validation_bpb']:
                best_states[kind] = inference(cpu_state(net), step, kind, result['bpb'])
                save_atomic(best_states[kind], args.run_dir/f'best_{kind}.pt')
            if result['bpb'] < best_bpb:
                best_bpb, best_kind, best_step = result['bpb'], kind, step
                save_atomic(best_states[kind], args.run_dir/'checkpoint.pt')
                improved = True
        bad = 0 if improved else bad+1

    def persist(reason):
        total_targets = ancestor_targets+stage_step*args.batch_size*256
        state = dict(protocol=PROTOCOL, implementation=implementation, config=config,
                     model=cpu_state(model), optimizer=optimizer.state_dict(),
                     ema=cpu_state(ema) if ema else None, rng=rng_state(generator),
                     seed=args.seed, recipe=recipe, schedule_horizon=horizon,
                     stage_step=stage_step, train_tokens=total_targets,
                     ancestor_targets=ancestor_targets, ancestor_train_seconds=ancestor_seconds,
                     ancestry=ancestry, sampling_digest=sampling_digest,
                     sampling_digest_start=sampling_digest_start,
                     train_seconds=train_seconds, validation_seconds=validation_seconds,
                     history=history, validation_history=validations, best_states=best_states,
                     selection=(best_bpb, best_kind, best_step, bad), implementation_sha256=implementation_sha)
        save_atomic(state, args.run_dir/'training_state.pt')
        save_atomic(inference(cpu_state(model), stage_step, 'raw', None), args.run_dir/'last_raw.pt')
        if ema:
            save_atomic(inference(cpu_state(ema), stage_step, 'ema', None), args.run_dir/'last_ema.pt')
        result = {k: v for k, v in state.items() if k not in ('model','optimizer','ema','rng','best_states')}
        result.update(command=sys.argv, source=str(source_path) if source_path else None,
                      source_sha256=sha(source_path) if source_path else None,
                      optimizer_mode='restored' if args.resume else 'rebuilt_new_stage' if source else 'fresh',
                      schedule_replanned=args.resume_horizon is not None,
                      checkpoint_every=args.checkpoint_every,
                      cumulative_train_seconds=ancestor_seconds+train_seconds,
                      stage_train_targets=stage_step*args.batch_size*256,
                      process_seconds=time.perf_counter()-started, stop_reason=reason,
                      torch_version=str(torch.__version__), best_validation_bpb=best_bpb)
        (args.run_dir/'metrics.json').write_text(json.dumps(result, indent=2)+'\n')

    if args.resume:
        for kind, checkpoint in best_states.items():
            save_atomic(checkpoint, args.run_dir/f'best_{kind}.pt')
        save_atomic(best_states[best_kind], args.run_dir/'checkpoint.pt')
    else:
        validate(0)  # Keep ancestor eligible even when every update harms validation.
        persist('initialized')
    model.train()
    reason = 'completed'
    for step in range(stage_step, horizon):
        if args.patience and bad >= args.patience:
            reason = 'early_stopping'
            break
        tick = time.perf_counter()
        starts_cpu = torch.randint(len(tokens)-256, (args.batch_size,), generator=generator)
        sampling_digest = hashlib.sha256(bytes.fromhex(sampling_digest)+starts_cpu.numpy().tobytes()).hexdigest()
        starts = starts_cpu.to(device)
        batch = tokens[starts[:, None]+torch.arange(257, device=device)]
        lr = learning_rate(step, horizon, args.lr, args.min_lr, args.warmup)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(device, precision):
            loss = F.cross_entropy(loss_model(batch[:, :-1]).flatten(0, 1).float(), batch[:, 1:].flatten())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss; previous checkpoint remains intact')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(loss_model.parameters(), 1.)
        optimizer.step()
        if ema:
            with torch.no_grad():
                for average, parameter in zip(ema_loss_model.parameters(), loss_model.parameters()):
                    average.lerp_(parameter, 1-args.ema_decay)
                # Fixed n-gram tables need no per-update copy when only the
                # neural component learns; they remain identical in raw/EMA.
                for average, buffer in zip(ema_loss_model.buffers(), loss_model.buffers()):
                    average.copy_(buffer)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter()-tick
        stage_step = step+1
        if stage_step % 20 == 0 or stage_step == horizon:
            row = dict(step=stage_step, loss=loss.item(), lr=lr, train_seconds=train_seconds)
            history.append(row)
            print(json.dumps(row), flush=True)
        stopping = args.stop_after is not None and stage_step >= args.stop_after
        if (args.eval_every and stage_step % args.eval_every == 0) or stage_step == horizon:
            validate(stage_step)
            persist('running')
        elif args.checkpoint_every and stage_step % args.checkpoint_every == 0:
            persist('running')
        if stopping:
            reason = 'requested_stop'
            break
    persist(reason)
    print(json.dumps(dict(best_bpb=best_bpb, best_kind=best_kind, best_step=best_step,
                          final_step=stage_step, stop_reason=reason)), flush=True)


if __name__ == '__main__':
    main()
