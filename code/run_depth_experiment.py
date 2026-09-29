"""Bounded, serial, equal-target 8-vs-10-layer continuation experiment.

No inference/data/scorer changes or mixture searches. Refuses existing outputs.
The two real smoke updates count toward each branch's 1000-update maximum.
"""
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT/'runs/depth_audit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    started = time.perf_counter()
    if (AUDIT/'plan.json').exists():
        raise RuntimeError('Experiment already started; do not overwrite its outputs')
    recipe = ['--steps','1000','--batch-size','32','--lr','5e-5','--min-lr','5e-6','--warmup','50',
              '--ema-decay','0.99','--eval-every','400','--patience','2','--checkpoint-every','20',
              '--threads','3','--device','cpu','--precision','fp32','--seed','31','--train-component','neural']
    plan = dict(source='runs/hybrid-adaptive-selected/checkpoint.pt', depths=[8,10],
                stages=[2,400,800,1000], maximum_updates_per_depth=1000,
                maximum_total_updates=2000, maximum_total_training_targets=16384000,
                recipe=recipe, objective='ordinary neural-only cross entropy',
                validation='full hybrid, fixed mixture/table, raw and EMA separately',
                early_stop='stop both branches at the shared step if either has two failed validation events',
                sampler='same seed, batch size, training data, chained SHA256 of sampled starts',
                optimizer='new AdamW stage for both; restore optimizer/RNG across stage boundaries',
                preflight='three alternating baseline/expanded full validation processes, CPU FP32 threads=3')
    (AUDIT/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    commands, measures, stages = [], [], []

    def run(command, label):
        log = AUDIT/f'{label}.log'
        if log.exists():
            raise RuntimeError(f'Log already exists: {log}')
        command = [sys.executable]+command
        print(json.dumps(dict(event='start', label=label, command=command)), flush=True)
        tick = time.perf_counter()
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        row = dict(label=label, command=command, returncode=result.returncode,
                   process_seconds=time.perf_counter()-tick, log=str(log.relative_to(ROOT)))
        commands.append(row)
        (AUDIT/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        print(json.dumps(dict(event='finish', **row)), flush=True)
        if result.returncode:
            raise RuntimeError(f'{label} failed; see {log}')

    def measurements(prefix, checkpoint, asset_bytes):
        rows = []
        for trial in range(1, 4):
            for kind in (['baseline','candidate'] if trial % 2 else ['candidate','baseline']):
                cp = ROOT/'runs/baseline/checkpoint.pt' if kind == 'baseline' else checkpoint
                label = f'{prefix}_{kind}_{trial}'
                output = AUDIT/f'{label}.json'
                run(['measure_eval.py','--checkpoint',str(cp),'--split','validation','--device','cpu',
                     '--precision','fp32','--threads','3','--output',str(output)], label)
                row = dict(name=kind, trial=trial, phase=prefix, **read(output),
                           resources=read(output.with_suffix('.resources.json')))
                if row['targets'] != 376599 or row['utf8_bytes'] != 1148007:
                    raise RuntimeError('Incomplete validation')
                rows.append(row)
                measures.append(row)
                (AUDIT/'measurements.json').write_text(json.dumps(measures, indent=2)+'\n')
        base = statistics.median(row['seconds'] for row in rows if row['name'] == 'baseline')
        candidate = [row for row in rows if row['name'] == 'candidate']
        timing = statistics.median(row['seconds'] for row in candidate)
        peak = max(row['resources']['peak_working_set_bytes'] for row in candidate)
        if max(row['bpb'] for row in candidate)-min(row['bpb'] for row in candidate) > 1e-10:
            raise RuntimeError('Repeated BPB differs')
        result = dict(bpb=candidate[0]['bpb'], baseline_seconds=base, candidate_seconds=timing,
                      time_ratio=timing/base, peak_working_set_bytes=peak,
                      peak_commit_bytes=max(row['resources']['peak_pagefile_bytes'] for row in candidate),
                      asset_bytes=asset_bytes, passes=timing/base <= 5 and peak <= 4*2**30 and asset_bytes <= 64*2**20)
        (AUDIT/f'{prefix}.json').write_text(json.dumps(result, indent=2)+'\n')
        return result

    initial = read(AUDIT/'initialization.json')
    previous = read(ROOT/'runs/adaptive_audit/summary.json')
    expanded = next(row for row in initial['results'] if row['depth'] == 10)
    preflight = measurements('preflight', Path(expanded['checkpoint']), expanded['asset_bytes'])
    if abs(preflight['bpb']-previous['after_bpb']) > 1e-9:
        raise RuntimeError('Expanded initialization changed full validation BPB')
    if not preflight['passes']:
        (AUDIT/'outcome.json').write_text(json.dumps(dict(status='preflight_resource_failure',
            retained_bundle=previous['recommended_bundle'], preflight=preflight, new_training_targets=0), indent=2)+'\n')
        return
    sources = {depth: ROOT/f'runs/depth{depth}-init/checkpoint.pt' for depth in (8,10)}
    final_step, stop_reason = 0, 'completed'
    for end in plan['stages']:
        pair = []
        for depth in (8,10):
            run_dir = ROOT/f'runs/depth{depth}-step{end}'
            if run_dir.exists():
                raise RuntimeError(f'Training output exists: {run_dir}')
            mode = '--init-from' if end == 2 else '--resume'
            command = ['train.py',mode,str(sources[depth]),'--run-dir',str(run_dir),'--stop-after',str(end)]+recipe
            run(command, f'train_depth{depth}_step{end}')
            metrics = read(run_dir/'metrics.json')
            if metrics['stage_step'] != end:
                raise RuntimeError('Branch stopped before the shared boundary')
            pair.append(dict(depth=depth, run_dir=str(run_dir.relative_to(ROOT)),
                             step=end, train_tokens=metrics['train_tokens'],
                             stage_train_targets=metrics['stage_train_targets'],
                             sampling_digest=metrics['sampling_digest'],
                             sampling_digest_start=metrics['sampling_digest_start'],
                             best_validation_bpb=metrics['best_validation_bpb'], selection=metrics['selection'],
                             train_seconds=metrics['train_seconds'], validation_seconds=metrics['validation_seconds']))
            sources[depth] = run_dir/'training_state.pt'
            if end == 2:
                run(['verify_depth.py','--checkpoint',str(run_dir/'last_raw.pt'),
                     '--output',str(AUDIT/f'smoke_depth{depth}.json')], f'verify_smoke_depth{depth}')
        matched = pair[0]['sampling_digest'] == pair[1]['sampling_digest'] and all(row['sampling_digest_start'] == 0 for row in pair)
        if not matched:
            raise RuntimeError('Training windows did not match between branches')
        stages.append(dict(step=end, matched_training_samples=matched, branches=pair))
        (AUDIT/'stages.json').write_text(json.dumps(stages, indent=2)+'\n')
        print(json.dumps(dict(event='paired_boundary', **stages[-1])), flush=True)
        final_step = end
        if any(row['selection'][3] >= 2 for row in pair):
            stop_reason = 'paired_early_stop'
            break
    finalists = stages[-1]['branches']
    winner = min(finalists, key=lambda row: row['best_validation_bpb'])
    for row in finalists:
        run(['verify_depth.py','--checkpoint',str(ROOT/row['run_dir']/'last_raw.pt'),
             '--output',str(AUDIT/f'final_weights_depth{row["depth"]}.json')], f'verify_final_depth{row["depth"]}')
    bundle = ROOT/'runs/depth-selected'
    if bundle.exists():
        raise RuntimeError('Final bundle exists')
    bundle.mkdir()
    shutil.copy2(ROOT/winner['run_dir']/'checkpoint.pt', bundle/'checkpoint.pt')
    for name in ('hybrid.py','student.py'):
        shutil.copy2(ROOT/name, bundle/name)
    (bundle/'README.txt').write_text(
        'MP1 bounded depth experiment inference bundle.\n'
        'Use these four files with the unchanged common.py/evaluate.py/data/tokenizer.\n'
        '.venv/Scripts/python.exe evaluate.py --checkpoint runs/depth-selected/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3\n'
        'Python 3.12, torch 2.7.1, CPU FP32. All fixed statistics are inside checkpoint.pt.\n', encoding='utf-8')
    asset_bytes = sum(path.stat().st_size for path in bundle.iterdir() if path.is_file())
    final = measurements('final', bundle/'checkpoint.pt', asset_bytes)
    if abs(final['bpb']-winner['best_validation_bpb']) > 1e-9:
        raise RuntimeError('Exported winner disagrees with training validation')
    accepted = final['passes'] and final['bpb'] < previous['after_bpb']
    result = dict(status='complete', before_bpb=previous['after_bpb'], after_bpb=final['bpb'],
                  delta_bpb=final['bpb']-previous['after_bpb'], accepted=accepted,
                  recommended_bundle='runs/depth-selected' if accepted else previous['recommended_bundle'],
                  selected_depth=winner['depth'], selected_weights=winner['selection'][1],
                  selected_step=winner['selection'][2], final_step_per_branch=final_step,
                  stopping_reason=stop_reason, preflight=preflight, resources=final,
                  total_new_updates=2*final_step, total_new_training_targets=2*final_step*32*256,
                  inherited_neural_training_targets=previous['neural_training_targets'],
                  inherited_neural_train_seconds=previous['inherited_neural_train_seconds'],
                  new_train_seconds=sum(row['train_seconds'] for row in finalists),
                  new_validation_seconds=sum(row['validation_seconds'] for row in finalists),
                  experiment_process_seconds=time.perf_counter()-started,
                  commands=commands, stages=stages)
    (AUDIT/'outcome.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('commands','stages')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
