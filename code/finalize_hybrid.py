"""Freeze one chosen predictor, run serial validation/resource comparisons, report."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT/'runs/hybrid_audit'
BUNDLE = ROOT/'runs/hybrid-selected'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if BUNDLE.exists() or (AUDIT/'summary.json').exists():
        raise RuntimeError('Use fresh output locations; completed results are preserved')
    stages = [(ROOT/f'runs/{name}', read(ROOT/f'runs/{name}/metrics.json')) for name in
              ('hybrid-v1', 'hybrid-v2-long', 'hybrid-v3-copy')]
    source, best = min(stages, key=lambda pair: pair[1]['selected']['bpb'])
    BUNDLE.mkdir()
    shutil.copy2(source/'checkpoint.pt', BUNDLE/'checkpoint.pt')
    for name in ('hybrid.py', 'student.py'):
        shutil.copy2(ROOT/name, BUNDLE/name)
    (BUNDLE/'README.txt').write_text(
        'MP1 inference bundle. Place hybrid.py and student.py alongside the supplied unchanged evaluator/common.py.\n'
        'Use the supplied unchanged data and tokenizer, Python 3.12 and torch 2.7.1.\n'
        'From the project code directory:\n'
        '.venv/Scripts/python.exe evaluate.py --checkpoint runs/hybrid-selected/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3\n'
        'One checkpoint contains neural weights and all train-derived tables. No external index or network access is needed.\n'
        'Only these four files are inference assets. Training states, experimental alternatives and logs are not part of the bundle.\n', encoding='utf-8')
    assets = [BUNDLE/name for name in ('checkpoint.pt', 'hybrid.py', 'student.py', 'README.txt')]
    asset_bytes = sum(p.stat().st_size for p in assets)
    if asset_bytes > 64*2**20:
        raise RuntimeError('Inference asset limit exceeded')
    command = [sys.executable, 'ablate_hybrid.py', '--checkpoint', str(BUNDLE/'checkpoint.pt'),
               '--output', str(AUDIT/'ablations.json'), '--threads', '3']
    print(json.dumps(dict(command=command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    records = []
    for trial in range(1, 4):
        for name in (['baseline', 'candidate'] if trial % 2 else ['candidate', 'baseline']):
            checkpoint = ROOT/'runs/baseline/checkpoint.pt' if name == 'baseline' else BUNDLE/'checkpoint.pt'
            output = AUDIT/f'{name}_trial{trial}.json'
            if output.exists():
                raise RuntimeError(f'Output exists: {output}')
            command = [sys.executable, 'measure_eval.py', '--checkpoint', str(checkpoint), '--split', 'validation',
                       '--device', 'cpu', '--precision', 'fp32', '--threads', '3', '--output', str(output)]
            print(json.dumps(dict(command=command)), flush=True)
            subprocess.run(command, cwd=ROOT, check=True)
            records.append(dict(name=name, trial=trial, **read(output), resources=read(output.with_suffix('.resources.json'))))
    base = [r for r in records if r['name'] == 'baseline']
    candidate = [r for r in records if r['name'] == 'candidate']
    base_seconds = statistics.median(r['seconds'] for r in base)
    candidate_seconds = statistics.median(r['seconds'] for r in candidate)
    peak = max(r['resources']['peak_working_set_bytes'] for r in candidate)
    commit = max(r['resources']['peak_pagefile_bytes'] for r in candidate)
    previous = read(ROOT/'runs/continuation_audit/summary.json')
    before, after = previous['after_bpb'], candidate[0]['bpb']
    if any(abs(r['bpb']-after) > 1e-10 for r in candidate):
        raise RuntimeError('Repeated scores differ')
    if abs(best['selected']['bpb']-after) > 1e-6:
        raise RuntimeError('Selection and original evaluator disagree')
    protected = []
    for filename in (AUDIT/'original_hashes.json', ROOT/'runs/continuation_audit/original_hashes.json'):
        for row in read(filename):
            path = Path(row['Path'])
            protected.append(dict(path=str(path), unchanged=digest(path).lower() == row['Hash'].lower()))
    if not all(row['unchanged'] for row in protected):
        raise RuntimeError('A protected file changed')
    passed = candidate_seconds/base_seconds <= 5 and peak <= 4*2**30 and asset_bytes <= 64*2**20
    ablations = read(AUDIT/'ablations.json')
    summary = dict(before_bpb=before, after_bpb=after, delta_bpb=after-before,
                   relative_reduction_percent=(1-after/before)*100, reached_1_3=after <= 1.3,
                   selected_source=str(source.relative_to(ROOT)), inference_bundle=str(BUNDLE.relative_to(ROOT)),
                   baseline_median_seconds=base_seconds, candidate_median_seconds=candidate_seconds,
                   time_ratio=candidate_seconds/base_seconds, peak_working_set_bytes=peak, peak_commit_bytes=commit,
                   inference_asset_bytes=asset_bytes, resource_limits_pass=passed,
                   parameters_neural_unchanged=ablations['neural_backbone_bit_identical'],
                   neural_training_targets=previous['cumulative_targets'], additional_neural_updates=0,
                   statistical_fit_targets=sum(stages[i][1].get('train_statistics_targets', stages[i][1].get('train_targets', 0)) for i in (0, 1)),
                   neural_ancestry_train_seconds=previous['cumulative_train_seconds'],
                   statistical_fit_seconds=sum(stages[i][1]['fit_seconds'] for i in (0, 1)),
                   experiment_seconds=sum(row.get('process_seconds', row.get('seconds', 0)) for _,row in stages),
                   selection_candidates=44, stages=[dict(name=p.name, selected=m['selected']) for p,m in stages],
                   measurements=records, ablations=ablations['results'], protected_files=protected,
                   assets=[dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=digest(p)) for p in assets])
    (AUDIT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# MP1 混合概率模型实验', '',
             f'完整验证集 BPB 从 {before:.12f} 降为 {after:.12f}，相对下降 {summary["relative_reduction_percent"]:.3f}%。',
             f'本轮{"达到" if after <= 1.3 else "未达到"} 1.3；结果来自原始 evaluate.py，覆盖 376,599 targets / 1,148,007 UTF-8 bytes。未执行测试集评分。', '',
             '## 修改与原理', '',
             '- 保留原有神经网络参数，逐张量核对与上一轮最佳权重完全相同。student.py、train.py、数据、tokenizer、common.py、evaluate.py 保持本轮开始时的内容。',
             '- 新增 hybrid.py：神经语言模型与训练集统计概率线性融合。2–5 阶使用插值 Modified Kneser–Ney；保留全部低阶统计，高阶裁剪单例并把概率质量分配给低阶模型。',
             '- 扩展 6–10 阶重复短语，使用训练计数和折扣回退。64 位哈希仅用于查找；训练时检测冲突，预测时还必须核对完整上下文。它是五阶 KN 之上的长短语扩展，不冒称完整十阶 KN。',
             '- 神经缓存根据当前隐藏向量与同一窗口内过去隐藏向量的相似度，对已观察到的后续 token 分配概率。用余弦归一化特征，温度参数来自有限验证选择。',
             '- 短语复制仅匹配同一窗口中已经出现的最长后缀（最多 8 token），只复制位置 s+1 ≤ t 的已观察 token。匹配长度越长，融合强度越高。',
             '- 所有与输入有关的匹配、隐藏状态和缓存只在本次 forward 内存在，窗口和样本之间不共享。跨调用保留的 bigram 矩阵完全由固定训练统计推导。',
             '- 没有把验证答案、验证专用表或预计算预测放入 checkpoint；模型接口只接收输入 ids。', '',
             '## 有界实验与消融', '',
             '第一阶段 20 组融合参数；第二阶段一次长短语扩展与 18 组温度/融合参数；第三阶段 6 个窗口复制强度。共 44 个预先限定的候选，随后冻结并做消融、评分和资源检查。',
             '温度最终仍为 1.0；神经/训练统计/神经缓存基础权重为 0.75/0.20/0.05。窗口短语复制强度为 0.4 乘以 clamp((匹配长度−1)/3, 0, 1)。', '',
             '| 方案 | 完整验证 BPB |', '|---|---:|', f'| 上一轮最佳神经模型 | {before:.9f} |']
    for path, metrics in stages:
        lines.append(f'| {path.name} | {metrics["selected"]["bpb"]:.9f} |')
    lines += ['', '消融使用同一套神经参数和其余相同系数，不为被移除模块重新调参。', '', '| 消融 | 完整验证 BPB |', '|---|---:|']
    names = dict(neural_only='仅神经模型', without_train_statistics='移除训练集统计',
                 without_neural_cache='移除神经缓存', without_phrase_copy='移除窗口短语复制')
    for row in ablations['results']:
        lines.append(f'| {names[row["name"]]} | {row["bpb"]:.9f} |')
    lines += [f'| 完整候选 | {after:.9f} |', '', '## 资源与验证', '',
              '既有 .venv：Python 3.12.0，PyTorch 2.7.1+cpu，AMD Ryzen 7 H 255；所有正式评分均为 CPU / FP32 / 3 线程 / 原始批量 32 / 全验证集。',
              'baseline 与候选分别启动独立进程，串行交替顺序测量 3 次。时间为原始 score 的 seconds（不含加载）；内存为原生 Windows 进程全生命周期峰值工作集，包含加载。', '',
              '| 约束 | 实测 | 限制 |', '|---|---:|---:|',
              f'| baseline 评分中位秒数 | {base_seconds:.3f} | — |',
              f'| 候选评分中位秒数 | {candidate_seconds:.3f} | {5*base_seconds:.3f} |',
              f'| 耗时倍率 | {candidate_seconds/base_seconds:.3f}× | 5× |',
              f'| 候选峰值工作集 | {peak/2**30:.3f} GiB | 4 GiB |',
              f'| 推理包未压缩大小 | {asset_bytes/2**20:.3f} MiB | 64 MiB |', '',
              f'预算检查：{"通过" if passed else "未通过，保留原有最佳作为可提交模型"}。候选峰值提交内存另为 {commit/2**30:.3f} GiB。',
              '14 项测试通过，涵盖原有模型/恢复检查、完整归一化、未来输入隔离、独立窗口、短前缀一致性、输入缓存仅含已观察 token、哈希冲突精确回退。另完成真实训练前缀短运行检查。',
              '受保护文件和旧 checkpoint 哈希全部复核一致。最终候选的三次原始评分 BPB 一致，统计选择分数与独立评分误差小于 1e-6。',
              '本轮资源结论基于完整验证集；按要求未执行测试集评分或测试集计时。', '',
              '## 成本', '',
              f'- 神经权重继承 {previous["cumulative_targets"]:,} 个训练 targets，沿保留轨迹训练 {previous["cumulative_train_seconds"]:.3f} 秒；上一轮丢失进度、短运行和其他历史搜索成本见 continuation_audit，不能省略。',
              f'- 本轮新增神经梯度更新：0。两次训练集建表合计 {summary["statistical_fit_targets"]:,} 个统计拟合 targets，建表和短检查合计 {summary["statistical_fit_seconds"]:.3f} 秒。',
              '- 每个统计阶数会重数训练序列中的连续片段；各阶观察数在 metrics.json 中分别记录。统计 pass 与 SGD targets 的含义不同，不用一个混合数字掩盖差异。',
              '- 此外做过一次训练集计数规模探查，约 361 万 targets，未计入上述两次正式拟合且没有改变模型。工程测试只用临时合成序列，其参数不进入实验权重。',
              f'- 三阶段拟合与参数选择进程记录合计 {summary["experiment_seconds"]:.3f} 秒；独立复现、消融和 6 次正式资源评分的耗时另逐项保存。',
              '- 所有旧模型和结果保留。未自动发起更多神经训练或无限搜索。', '',
              '## 复现与资产', '',
              '从 code/ 使用 .venv/Scripts/python.exe。再次拟合必须指定新的 --run-dir，防止覆盖已有结果。', '']
    for _, metrics in stages:
        lines.append('`'+' '.join(metrics['command'])+'`\n')
    lines += ['最终评分：', '',
              '`.venv/Scripts/python.exe evaluate.py --checkpoint runs/hybrid-selected/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3`', '',
              '推理资产仅为 runs/hybrid-selected/ 中的 checkpoint.pt、hybrid.py、student.py、README.txt。检查点包含全部训练统计，不依赖其他索引或数据文件；评分仍使用作业提供的固定数据和 tokenizer。',
              '审计：summary.json、ablations.json、各 trial JSON / resources JSON、各阶段 metrics.json、tests_with_copy.log 和各训练/选择日志。修改前副本在 before/，中间代码快照在 v1-code/、v2-code/。', '',
              '## 依据与限制', '',
              '平滑参考 [Chen & Goodman (1999)](https://u.cs.biu.ac.il/~yogo/courses/mt2013/papers/chen-goodman-99.pdf)；神经缓存参考 [Grave 等 (2017)](https://arxiv.org/abs/1612.04426)。代码为本项目实现，缓存被限制为独立评分窗口内，不能直接套用论文的大缓存成绩。',
              '模型互补能降低 BPB，但达不到 1.3 不能靠提高混合权重来保证。验证选择结果不等于测试集结果。与 baseline 的神经训练预算不同；同源权重消融可检验本轮附加机制，却不替代全部历史架构的等成本对照。',
              'Codex 实质协助实现、测试、实验、资源测量和报告；README 已披露。', '']
    (AUDIT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('measurements','protected_files','assets','ablations')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
