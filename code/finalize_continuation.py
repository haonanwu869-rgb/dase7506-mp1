"""Bounded post-training validation/resource audit; never evaluates test."""
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT/'runs/continuation_audit'
RUN = ROOT/'runs/rope-v4b-continuation-resumed'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    metrics = read(RUN/'metrics.json')
    if metrics['stop_reason'] not in ('completed', 'early_stopping'):
        raise RuntimeError('Training has not completed')
    records = []
    for trial in range(1, 4):
        order = ['baseline', 'candidate'] if trial % 2 else ['candidate', 'baseline']
        for name in order:
            checkpoint = ROOT/'runs/baseline/checkpoint.pt' if name == 'baseline' else RUN/'checkpoint.pt'
            output = AUDIT/f'{name}_trial{trial}.json'
            if output.exists():
                raise RuntimeError(f'Refusing to overwrite {output}')
            command = [sys.executable, 'measure_eval.py', '--checkpoint', str(checkpoint),
                       '--split', 'validation', '--device', 'cpu', '--precision', 'fp32',
                       '--threads', '3', '--output', str(output)]
            print(json.dumps({'command': command}), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            records.append(dict(name=name, trial=trial, **read(output),
                                resources=read(output.with_suffix('.resources.json'))))
    original = read(AUDIT/'original_hashes.json')
    verification = []
    for row in original:
        path = Path(row['Path'])
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        verification.append(dict(path=str(path), unchanged=actual.lower() == row['Hash'].lower()))
    if not all(row['unchanged'] for row in verification):
        raise RuntimeError('A protected file changed')
    before = read(AUDIT/'rope_v4b_before.json')
    smoke = read(ROOT/'runs/continuation_smoke/metrics.json')
    recovery = read(AUDIT/'recovery.json')
    base = [row for row in records if row['name'] == 'baseline']
    candidate = [row for row in records if row['name'] == 'candidate']
    base_time = statistics.median(row['seconds'] for row in base)
    candidate_time = statistics.median(row['seconds'] for row in candidate)
    peak = max(row['resources']['peak_working_set_bytes'] for row in candidate)
    committed = max(row['resources']['peak_pagefile_bytes'] for row in candidate)
    # Inference bundle: a single chosen weight file plus required Python modules.
    assets = [RUN/'checkpoint.pt', ROOT/'student.py']
    asset_bytes = sum(path.stat().st_size for path in assets)
    selected_bpb = candidate[0]['bpb']
    resource_pass = candidate_time/base_time <= 5 and peak <= 4*2**30 and asset_bytes <= 64*2**20
    summary = dict(before_bpb=before['bpb'], after_bpb=selected_bpb,
                   delta_bpb=selected_bpb-before['bpb'],
                   relative_reduction_percent=(1-selected_bpb/before['bpb'])*100,
                   baseline_median_seconds=base_time, candidate_median_seconds=candidate_time,
                   time_ratio=candidate_time/base_time, candidate_peak_working_set_bytes=peak,
                   candidate_peak_commit_bytes=committed, inference_asset_bytes=asset_bytes,
                   inference_assets=[str(p.relative_to(ROOT)) for p in assets],
                   resource_limits_pass=resource_pass, protected_files=verification,
                   stage_steps=metrics['stage_step'], stage_targets=metrics['stage_train_targets'],
                   cumulative_targets=metrics['train_tokens'],
                   stage_train_seconds=metrics['train_seconds'],
                   cumulative_train_seconds=metrics['cumulative_train_seconds'],
                   smoke_targets=smoke['stage_train_targets'], smoke_train_seconds=smoke['train_seconds'],
                   total_new_targets_lower_bound=metrics['stage_train_targets']+smoke['stage_train_targets']+recovery['discarded_updates_lower_bound']*8192,
                   total_new_targets_upper_bound=metrics['stage_train_targets']+smoke['stage_train_targets']+recovery['discarded_updates_conservative_upper_bound']*8192,
                   discarded_train_seconds_lower_bound=recovery['discarded_train_seconds_lower_bound'],
                   total_new_train_seconds_lower_bound=metrics['train_seconds']+smoke['train_seconds']+recovery['discarded_train_seconds_lower_bound'],
                   recovery=recovery,
                   selection=metrics['selection'], stop_reason=metrics['stop_reason'], measurements=records)
    (AUDIT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = [
        '# MP1 有界续训实验报告', '',
        f'原始 rope-v4b 完整验证 BPB：{before["bpb"]:.12f}；本轮选择结果：{selected_bpb:.12f}。',
        f'变化 {summary["delta_bpb"]:+.12f} BPB，相对下降 {summary["relative_reduction_percent"]:.4f}%。',
        f'选择权重：{metrics["selection"][1]}；阶段步数：{metrics["selection"][2]}；停止原因：{metrics["stop_reason"]}。', '',
        '## 代码和验证', '',
        '- 原始 student.py、train.py、README.md 副本保存在 before/，所有旧 checkpoint、固定评估器及数据哈希复核未变。',
        '- dropout 从 config/CLI 传入模型，未提供时仍为 0.1。模型架构和参数量未改。',
        '- 训练器增加可配置学习率、最低学习率、warmup、阶段步数、最佳模型保存、完整恢复、EMA、早停。',
        '- 保存 optimizer、采样 generator、Python/NumPy/PyTorch CPU/CUDA RNG、EMA、当前步数及验证选择状态。',
        '- 8 项测试通过，包含原始模型契约，以及恢复训练与连续训练的模型/EMA/optimizer 完全一致性。',
        '- 原始 evaluate.py 完整复现旧 BPB；本轮未执行测试集评分。原始 load_data 会加载所有 split，但只用 train 学习、validation 选择。', '',
        '## 固定实验方案', '',
        '复用 .venv：Python 3.12.0 / PyTorch 2.7.1+cpu；AMD Ryzen 7 H 255；无 CUDA；CPU FP32、3 线程。',
        '从原始 rope-v4b 权重开始新阶段，旧文件没有 optimizer 状态，故重建 AdamW，不声称精确恢复旧训练。',
        'seed=23；batch=32；context=256；最多 2000 步；lr=1e-4 余弦下降到 1e-5；warmup=0；weight_decay=0.1；dropout=0.1；EMA decay=0.99。',
        '每 400 步分别完整评估 raw 和 EMA；原始权重参加选择；连续两次验证事件均未刷新全局最佳则停止。',
        '短运行是独立的 2 步流程检查，不用于初始化正式续训。除下述中断后的预算调整外，未追加调参。', '',
        '运行至日志第 640 步时服务重启，Python 进程终止。恢复了最近保存的第 400 步 optimizer、模型、EMA、RNG 和验证选择状态，原中断目录保留。',
        '为把丢失进度的计算也计入 2000 步上限，显式缩短恢复后的余弦终点至阶段第 1600 步，末次学习率仍为 1e-5；这是预算驱动的调度重规划，不声称与原 2000 步学习率轨迹相同。',
        '中断后增加每 100 步保存完整恢复状态；每 400 步完整验证规则不变。恢复后的 8 项测试通过，含相同调度的精确恢复及缩短调度终点检查。', '',
        '| 阶段步数 | 普通权重 BPB | EMA BPB |', '|---:|---:|---:|']
    grouped = {}
    for row in metrics['validation_history']:
        grouped.setdefault(row['step'], {})[row['kind']] = row['bpb']
    for step, values in sorted(grouped.items()):
        lines.append(f'| {step} | {values["raw"]:.9f} | {values["ema"]:.9f} |')
    lines += ['', '## 成本与资源', '',
              f'- 继承成本：4000 步，32,768,000 targets，训练 {metrics["ancestor_train_seconds"]:.3f} 秒。',
              f'- 正式本轮：{metrics["stage_step"]} 步，{metrics["stage_train_targets"]:,} targets，训练 {metrics["train_seconds"]:.3f} 秒；阶段验证 {metrics["validation_seconds"]:.3f} 秒。',
              f'- 保留训练轨迹的累计成本：{metrics["train_tokens"]:,} targets，{metrics["cumulative_train_seconds"]:.3f} 秒；中断丢失进度另计如下，选择较早 checkpoint 也不能抹去后续成本。',
              f'- 额外短运行：{smoke["stage_train_targets"]:,} targets，训练 {smoke["train_seconds"]:.3f} 秒，验证 {smoke["validation_seconds"]:.3f} 秒。',
              f'- 中断丢失进度仍计费：240–260 步（保守范围），1,966,080–2,129,920 targets，已记录训练时间至少 {recovery["discarded_train_seconds_lower_bound"]:.3f} 秒，未记录的末尾耗时无法精确还原。',
              f'- 本轮实际新增训练（含短运行和丢失进度）：{summary["total_new_targets_lower_bound"]:,}–{summary["total_new_targets_upper_bound"]:,} targets，训练至少 {summary["total_new_train_seconds_lower_bound"]:.3f} 秒；不能只按最终 checkpoint 步数计成本。',
              '- 其他历史已完成实验见 historical_runs.json；rope-v4_train.log 另记录一次至少 800 步、1764.511 秒的未完成运行，历史总搜索成本不是只有最佳模型成本。',
              '', '| 项目 | baseline | 候选 |', '|---|---:|---:|',
              f'| 验证 BPB | {base[0]["bpb"]:.9f} | {selected_bpb:.9f} |',
              f'| 3 次评分中位耗时（秒） | {base_time:.3f} | {candidate_time:.3f} |',
              f'| 峰值工作集 GiB | {max(r["resources"]["peak_working_set_bytes"] for r in base)/2**30:.3f} | {peak/2**30:.3f} |',
              '', f'时间倍率 {candidate_time/base_time:.3f}×；候选峰值提交内存 {committed/2**30:.3f} GiB；单份未压缩推理资产 {asset_bytes/2**20:.3f} MiB。',
              f'按本机完整验证评分，三项预算检查：{"通过" if resource_pass else "未通过，不应替换原最佳"}。',
              '时间采用 evaluate.score 的 seconds（不含加载）；内存为独立评估进程生命周期峰值，包含加载。3 轮交替次序、串行运行，线程/精度/批量/数据完全相同。',
              '这里只实测验证集，按用户要求不运行测试集，因而不声称已验证最终测试集时间。',
              'training_state.pt 含训练状态和备选权重，可能超过 64 MiB，不属于推理包；提交只需选中的 checkpoint.pt 和 student.py，固定运行环境/评分器由作业提供。', '',
              '## 关键原理与局限', '',
              '小学习率续训用新训练样本梯度缓慢修正已有模型；更多训练既可能改善泛化，也可能过拟合，因此用完整验证选择，不能只看训练 loss。',
              'EMA 是训练参数的指数滑动平均，降低相邻更新的噪声，但有滞后；同时评估原权重，避免假定平均必然更好。单份 EMA 权重的推理成本与原模型相同。',
              'dropout 仅在训练模式随机屏蔽残差分支，验证模式关闭；正确恢复 RNG 和 optimizer 动量才能让中断续训接上原来的更新轨迹。旧 checkpoint 缺少这些状态，故只能新建优化阶段。',
              'BPB 是验证文本总负对数概率（以 2 为底）除以原始 UTF-8 字节数，越低越好，不能直接当作 token perplexity。',
              '本轮是固定架构的训练阶段实验。raw/EMA 在相同训练 targets 下构成权重平均对照；baseline 与候选训练成本不同，不能据此宣称已完成作业要求的架构等成本对照或架构消融。',
              'Codex 实质协助代码、测试、实验执行和报告，README 已披露。', '',
              '## 复现命令与文件', '',
              '工作目录为 code，全部使用 .venv/Scripts/python.exe。README 给出完整主命令，metrics.json 中 command 保存训练 argv；每个 *.resources.json 保存评估 argv。',
              '- 修改前复现：`evaluate.py --checkpoint runs/rope-v4b/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3 --output runs/continuation_audit/rope_v4b_before.json`',
              '- 流程测试：`-m unittest discover -s tests -v`',
              '- 正式训练：`'+' '.join(metrics['command'])+'`',
              '- 短运行：`'+' '.join(smoke['command'])+'`',
              '- 资源复核：`finalize_continuation.py`，内部仅执行 validation；结果目录存在时拒绝覆盖。',
              '- 最佳模型：`../rope-v4b-continuation-resumed/checkpoint.pt`；恢复状态：`../rope-v4b-continuation-resumed/training_state.pt`。',
              '- 机器可读总表：`summary.json`；逐步日志：`continuation.log`；固定配置：`plan.json`。', '']
    (AUDIT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('measurements', 'protected_files')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
