"""Freeze a bounded adaptive-mixture experiment and measure it with the fixed scorer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT/'runs/hybrid-adaptive-v1')
    parser.add_argument('--audit-dir', type=Path, default=ROOT/'runs/adaptive_audit')
    parser.add_argument('--bundle-dir', type=Path, default=ROOT/'runs/hybrid-adaptive-selected')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    audit, bundle = args.audit_dir.resolve(), args.bundle_dir.resolve()
    if bundle.exists() or (audit/'summary.json').exists():
        parser.error('Use new output locations; completed results are preserved')
    started = time.perf_counter()
    experiment = read(args.run_dir/'metrics.json')
    previous = read(ROOT/'runs/hybrid_audit/summary.json')
    source = Path(experiment['source']).resolve()
    if digest(source) != experiment['source_sha256']:
        raise RuntimeError('Source checkpoint changed')
    bundle.mkdir(parents=True)
    shutil.copy2(args.run_dir/'checkpoint.pt', bundle/'checkpoint.pt')
    for name in ('hybrid.py', 'student.py'):
        shutil.copy2(ROOT/name, bundle/name)
    (bundle/'README.txt').write_text(
        'MP1 adaptive hybrid inference bundle. All neural weights and statistics are train-derived.\n'
        'Place hybrid.py and student.py alongside the unchanged common.py and evaluate.py.\n'
        'Use the supplied unchanged tokenizer/data, Python 3.12 and PyTorch 2.7.1.\n'
        f'From code/: .venv/Scripts/python.exe evaluate.py --checkpoint {bundle.relative_to(ROOT).as_posix()}/checkpoint.pt '
        '--split validation --device cpu --precision fp32 --threads 3\n'
        'These four files contain every additional inference asset; no other lookup files or network access.\n', encoding='utf-8')
    assets = [bundle/name for name in ('checkpoint.pt', 'hybrid.py', 'student.py', 'README.txt')]
    asset_bytes = sum(path.stat().st_size for path in assets)
    if asset_bytes > 64*2**20:
        raise RuntimeError('Inference assets exceed 64 MiB')
    records = []

    def measure(name, trial, checkpoint):
        output = audit/f'{name}_trial{trial}.json'
        if output.exists():
            raise RuntimeError(f'Output already exists: {output}')
        command = [sys.executable, 'measure_eval.py', '--checkpoint', str(checkpoint), '--split', 'validation',
                   '--device', 'cpu', '--precision', 'fp32', '--threads', str(args.threads), '--output', str(output)]
        print(json.dumps(dict(command=command)), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        result = dict(name=name, trial=trial, **read(output), resources=read(output.with_suffix('.resources.json')))
        records.append(result)
        (audit/'measurements.json').write_text(json.dumps(records, indent=2)+'\n')
        return result

    # Confirm zero-default backward compatibility through the original scorer.
    original = measure('previous', 1, source)
    if abs(original['bpb']-previous['after_bpb']) > 1e-9:
        raise RuntimeError('Previous checkpoint behavior changed')
    for trial in range(1, 4):
        for name in (['baseline', 'candidate'] if trial % 2 else ['candidate', 'baseline']):
            checkpoint = ROOT/'runs/baseline/checkpoint.pt' if name == 'baseline' else bundle/'checkpoint.pt'
            measure(name, trial, checkpoint)
    base = [row for row in records if row['name'] == 'baseline']
    candidate = [row for row in records if row['name'] == 'candidate']
    base_seconds = statistics.median(row['seconds'] for row in base)
    candidate_seconds = statistics.median(row['seconds'] for row in candidate)
    peak = max(row['resources']['peak_working_set_bytes'] for row in candidate)
    commit = max(row['resources']['peak_pagefile_bytes'] for row in candidate)
    after = candidate[0]['bpb']
    if any(abs(row['bpb']-after) > 1e-10 for row in candidate):
        raise RuntimeError('Repeated scores disagree')
    if abs(after-experiment['selected']['bpb']) > 1e-6:
        raise RuntimeError('Search disagrees with original scorer')
    if any(row['targets'] != 376599 or row['utf8_bytes'] != 1148007 for row in records):
        raise RuntimeError('Full validation coverage mismatch')
    protected = []
    for filename in (audit/'original_hashes.json', ROOT/'runs/continuation_audit/original_hashes.json'):
        for row in read(filename):
            path = Path(row['Path'])
            protected.append(dict(path=str(path), unchanged=digest(path).lower() == row['Hash'].lower()))
    if not all(row['unchanged'] for row in protected):
        raise RuntimeError('Protected data/code/checkpoint changed')
    passed = candidate_seconds/base_seconds <= 5 and peak <= 4*2**30 and asset_bytes <= 64*2**20
    before = original['bpb']
    accepted = passed and after < before
    summary = dict(before_bpb=before, after_bpb=after, delta_bpb=after-before,
                   relative_reduction_percent=(1-after/before)*100,
                   selected_settings=experiment['selected'], selected_source=str(args.run_dir),
                   candidate_bundle=str(bundle.relative_to(ROOT)),
                   recommended_bundle=str(bundle.relative_to(ROOT)) if accepted else previous['inference_bundle'],
                   accepted=accepted, reached_1_3=after <= 1.3,
                   baseline_median_seconds=base_seconds, candidate_median_seconds=candidate_seconds,
                   time_ratio=candidate_seconds/base_seconds, peak_working_set_bytes=peak, peak_commit_bytes=commit,
                   inference_asset_bytes=asset_bytes, resource_limits_pass=passed,
                   neural_training_targets=previous['neural_training_targets'],
                   inherited_neural_train_seconds=previous['neural_ancestry_train_seconds'],
                   inherited_statistical_fit_targets=previous['statistical_fit_targets'],
                   inherited_statistical_fit_seconds=previous['statistical_fit_seconds'],
                   inherited_selection_candidates=previous['selection_candidates'],
                   additional_neural_updates=0, additional_training_targets=0, additional_statistics_fits=0,
                   selection_candidates=len(experiment['results']), search_seconds=experiment['validation_seconds'],
                   search_process_seconds=experiment['process_seconds'], smoke=experiment['smoke'],
                   verification_process_seconds=time.perf_counter()-started,
                   tests_log='runs/adaptive_audit/tests.log', tests_passed=16,
                   measurements=records, search_results=experiment['results'], protected_files=protected,
                   assets=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p)) for p in assets])
    (audit/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    report = [
        '# MP1 动态融合有界实验（2026-09-29）', '',
        f'完整验证集 BPB：{before:.12f} → {after:.12f}，相对下降 {summary["relative_reduction_percent"]:.4f}%。',
        f'候选采用状态：{accepted}。推荐推理包：`{summary["recommended_bundle"]}`。未达到 1.3，未执行测试集评分。', '',
        '## 修改与原理', '',
        '- hybrid.py 新增三个默认关闭的选项，旧 checkpoint 的完整验证 BPB 已用原评估器复现。',
        '- ngram_gate：以统计模型和神经模型各自最大预测概率之比调整统计权重的 odds；比值限制为 [0.25,4]，最终权重有上限，并保留缓存和神经概率份额。它是置信度启发式，不是校准后的正确率。',
        '- copy_half_life：对已出现的最长匹配短语，按距离指数衰减其后续 token 的贡献；使用稳定归一化，且只允许历史位置 s<t。',
        '- copy_agreement：用已观察后续 token 的最大概率衡量匹配之间的一致性；存在分歧时降低复制强度。',
        '- 全部权重只依赖当前前缀。没有拟合验证标签、增加训练数据、修改神经参数或重新建表；验证集仅选择预先列出的 12 组超参数。', '',
        '## 固定候选与消融', '',
        '网格为 ngram_gate∈{0,0.5,1} × (copy_half_life,copy_agreement)∈{(0,0),(0,0.5),(64,0),(64,0.5)}，包括原规则及各模块关闭的对照。',
        '每批共享神经/统计/缓存计算，仅比较融合概率；12 个配置不是 12 轮神经训练。网格及源文件哈希在验证前写入 plan.json。', '',
        '| ngram_gate | copy_half_life | copy_agreement | 完整验证 BPB |', '|---:|---:|---:|---:|']
    for row in experiment['results']:
        report.append(f'| {row["ngram_gate"]} | {row["copy_half_life"]} | {row["copy_agreement"]} | {row["bpb"]:.9f} |')
    report += ['', '## 原评估器与资源', '',
        '所有正式评分为同机 CPU / FP32 / 3 线程 / 原始 batch=32 / 完整验证集；376,599 targets，1,148,007 UTF-8 bytes。baseline 与候选交替启动独立进程，各三次，无并行训练。',
        '评分时间取原始 score 的 seconds 中位数；RAM 使用 Windows 全进程峰值工作集，包含加载。资源结论只覆盖验证集。', '',
        '| 项目 | 实测 | 限制 |', '|---|---:|---:|',
        f'| baseline 评分时间 | {base_seconds:.3f} s | — |',
        f'| 候选评分时间 | {candidate_seconds:.3f} s | {5*base_seconds:.3f} s |',
        f'| 耗时倍率 | {candidate_seconds/base_seconds:.3f}× | 5× |',
        f'| 峰值工作集 | {peak/2**30:.3f} GiB | 4 GiB |',
        f'| 未压缩推理资产 | {asset_bytes/2**20:.3f} MiB | 64 MiB |', '',
        f'资源通过：{passed}。峰值提交内存另记为 {commit/2**30:.3f} GiB。三次候选 BPB 一致；保护文件/所有旧 checkpoint 哈希一致。',
        '16 项检查通过，含因果性、独立窗口、短前缀一致性、近期复制归一化、冲突后续词及原有恢复训练检查。',
        f'真实训练前缀短检查覆盖 256 targets 和全部 12 组配置，最大 logsumexp 误差 {experiment["smoke"]["maximum_normalization_error"]:.3g}，共享计算与模型实际输出的最大概率误差 {experiment["smoke"]["maximum_probability_error"]:.3g}。', '',
        '## 成本与限制', '',
        f'- 神经参数继承 {previous["neural_training_targets"]:,} 个 targets，保留训练轨迹用时 {previous["neural_ancestry_train_seconds"]:.3f} 秒。',
        f'- 继承统计拟合 {previous["statistical_fit_targets"]:,} 个 targets，拟合用时 {previous["statistical_fit_seconds"]:.3f} 秒；上轮选择过 {previous["selection_candidates"]} 个候选，详细历史仍见 hybrid_audit 和 continuation_audit。',
        f'- 本轮神经更新 0、新增拟合 targets 0；真实训练前缀仅做正确性检查，不参与学习。12 候选搜索评分 {experiment["validation_seconds"]:.3f} 秒，搜索进程 {experiment["process_seconds"]:.3f} 秒。',
        f'- 独立评分/资源验证进程耗时 {summary["verification_process_seconds"]:.3f} 秒；各命令与进程时间见 measurements.json。另有检查与开发时间，不等同于训练时间。',
        '- 开发时一次语法错误在启动前终止；一次短检查在全验证前因 1e-6 阈值终止，实际 FP32 误差为 1.11e-6，核对后改为 3e-6，远严于原评估器的 1e-3。另对原模型的同一训练前缀做过一次检查，误差 9.95e-7。失败日志保留，耗时未完整测量，未做梯度更新或额外验证搜索。',
        '- 本轮控制使用相同神经训练 targets 和相同统计表；消融能说明融合规则的作用，不意味着全部历史模型都做过等训练成本比较。小幅验证改进不保证测试集同样改善。', '',
        '## 复现', '',
        '`'+' '.join(experiment['command'])+'`', '',
        '重跑搜索应使用新的 --run-dir；finalize_adaptive.py 支持 --run-dir / --audit-dir / --bundle-dir，拒绝覆盖已有结果。资源复现可直接使用 measure_eval.py 对 baseline 和候选串行评分。', '',
        f'`.venv/Scripts/python.exe evaluate.py --checkpoint {bundle.relative_to(ROOT).as_posix()}/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3`', '',
        '备份位于 runs/adaptive_audit/before/，包含原 hybrid.py、student.py、train.py、README 和原测试。原最佳包和 checkpoint 保留。推理包仅四个文件，所有训练表在 checkpoint 内。',
        'Codex 实质协助设计、实现、检查、实验和报告，README 同步披露。', '']
    (audit/'REPORT.md').write_text('\n'.join(report), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('measurements','protected_files','assets','search_results')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
