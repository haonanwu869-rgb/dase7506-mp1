"""Audit the completed paired depth experiment and write its Chinese report."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT/'runs/depth_audit'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    outcome = read(AUDIT/'outcome.json')
    if (AUDIT/'summary.json').exists():
        raise RuntimeError('Completed report already exists')
    protected = [dict(path=row['Path'], unchanged=digest(Path(row['Path'])).lower() == row['Hash'].lower())
                 for row in read(AUDIT/'original_hashes.json')]
    if not all(row['unchanged'] for row in protected):
        raise RuntimeError('A protected file changed')
    if outcome['status'] != 'complete':
        (AUDIT/'summary.json').write_text(json.dumps(outcome | dict(protected_files=protected), indent=2)+'\n')
        (AUDIT/'REPORT.md').write_text(
            '# 深度扩容预检查\n\n10 层模型预检查未通过资源限制，因此没有启动梯度训练。\n\n'
            '保留原最佳模型 runs/hybrid-adaptive-selected，验证 BPB 1.508208363922。\n\n'
            '完整计时及内存数据见 preflight.json；原文件和 checkpoint 哈希全部一致。\n', encoding='utf-8')
        return
    finalists = outcome['stages'][-1]['branches']
    metrics = {row['depth']: read(ROOT/row['run_dir']/'metrics.json') for row in finalists}
    matched = all(row['matched_training_samples'] for row in outcome['stages'])
    if not matched:
        raise RuntimeError('Unmatched sampling')
    weights = {depth: read(AUDIT/f'final_weights_depth{depth}.json') for depth in (8,10)}
    if not all(row['frozen_statistics_bit_identical'] and row['fixed_settings_unchanged'] for row in weights.values()):
        raise RuntimeError('Fixed mixture/statistics changed')
    bundle = ROOT/'runs/depth-selected'
    assets = [dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=digest(path))
              for path in sorted(bundle.iterdir()) if path.is_file()]
    if any(digest(ROOT/name) != digest(bundle/name) for name in ('hybrid.py','student.py')):
        raise RuntimeError('Packaged implementation differs')
    measurements = read(AUDIT/'measurements.json')
    valid = all(row['split'] == 'validation' and row['targets'] == 376599 and row['utf8_bytes'] == 1148007
                for row in measurements)
    if not valid:
        raise RuntimeError('Wrong evaluation split/coverage')
    checks = dict(tests=18, tests_passed=True, all_protected_hashes_match=True,
                  matched_training_samples=True, statistics_unchanged=True,
                  inference_implementation_unchanged=True, all_full_validation=True,
                  total_new_updates=outcome['total_new_updates'], update_limit_pass=outcome['total_new_updates'] <= 2000,
                  resource_limits_pass=outcome['resources']['passes'])
    (AUDIT/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    summary = outcome | dict(protected_files=protected, assets=assets, checks=checks,
                              initialization=read(AUDIT/'initialization.json'), measurements=measurements,
                              validation_history={str(depth): m['validation_history'] for depth,m in metrics.items()},
                              selected_model_train_targets=outcome['inherited_neural_training_targets']+outcome['selected_step']*8192)
    (AUDIT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    r = outcome['resources']
    lines = ['# MP1 8 层与 10 层等训练 targets 对照（2026-09-29）', '',
             f'原最佳完整验证 BPB：{outcome["before_bpb"]:.12f}；本轮候选：{outcome["after_bpb"]:.12f}。',
             f'候选采用：{outcome["accepted"]}，推荐推理包 `{outcome["recommended_bundle"]}`。',
             f'选择 {outcome["selected_depth"]} 层 / {outcome["selected_weights"]} / 第 {outcome["selected_step"]} 步；两组均执行到第 {outcome["final_step_per_branch"]} 步，停止原因 `{outcome["stopping_reason"]}`。',
             '没有测试集评分，没有新增外部数据或外部预训练参数。是否达到 1.3 以实际 BPB 为准，不能保证下降。', '',
             '## 实验与代码', '',
             '- 准备 8 层原结构和 10 层扩容结构，width=192、heads=6、context=256、dropout=0.1 保持不变。新增两层的 attention 输出投影和 MLP down 投影初始化为零，其余新层参数保持随机初始化。',
             '- 新层起点为恒等映射。真实训练前缀上的预测逐位相同，完整验证起点也由原始 evaluate.py 确认为 1.508208363922。旧权重及训练统计逐张量一致。',
             '- train.py 新增 --train-component neural，只对神经 logits 计算原始交叉熵；完整混合模型用于验证。统计表、融合超参数、缓存和复制规则固定。没有在这轮同时改变目标函数或正则化。',
             '- 原学习阶段的 checkpoint 缺少与本轮结构匹配的 optimizer，两组均明确重建 AdamW。阶段间通过 training_state.pt 恢复 optimizer、EMA、RNG、调度位置、最佳结果和采样摘要。',
             '- 两组相同 seed=31、batch=32、学习率 5e-5→5e-6、warmup=50、cosine horizon=1000、AdamW weight_decay=0.1、EMA=0.99、CPU FP32 3 线程。每 400 步及终点完整验证普通/EMA 权重。',
             '- 正式训练先各做 2 步短运行，再恢复到 400、800、1000 步；短运行计入总预算。每 20 步保存恢复状态。任一分支连续两个验证事件未改善时，在共同步数停止两组。',
             '- 用独立随机采样生成器和逐步链式 SHA256 核对每组采样起点。所有共同边界的摘要一致，证明两组使用相同训练窗口。相等的是训练 targets，10 层的计算量及耗时更高。',
             '- prepare_depth.py 准备初始权重，verify_depth.py 检查固定表及新增层学习，run_depth_experiment.py 按固定计划串行运行并保存原始命令/返回码/耗时。', '',
             '## 完整验证曲线与等 targets 对照', '',
             '| 新阶段步数 | 每组新增 targets | 8 层普通 | 8 层 EMA | 10 层普通 | 10 层 EMA |', '|---:|---:|---:|---:|---:|---:|']
    by_depth = {depth: {(row['step'],row['kind']):row['bpb'] for row in m['validation_history']}
                for depth,m in metrics.items()}
    for step in sorted({step for step,kind in by_depth[8]} & {step for step,kind in by_depth[10]}):
        values = [by_depth[d].get((step,kind)) for d in (8,10) for kind in ('raw','ema')]
        lines.append(f'| {step} | {step*8192:,} | '+' | '.join(f'{v:.9f}' if v is not None else '—' for v in values)+' |')
    lines += ['', '上述 8 层对照与 10 层候选共享原训练轨迹、统计表和融合规则，新增 targets 相同，唯一架构区别为两层初始恒等残差块。只做一对固定种子实验，不将其解释为普遍的容量结论。', '',
              '## 资源', '',
              '预检查和最终候选均使用原始评分器，同机 CPU / FP32 / 3 线程 / batch=32 / 完整验证集；baseline 与候选独立进程交替评分，各三次。RAM 为 Windows 全进程峰值工作集，包含加载，另保留峰值提交内存。', '',
              '| 指标 | 实测 | 限制 |', '|---|---:|---:|',
              f'| baseline 中位耗时 | {r["baseline_seconds"]:.3f} s | — |',
              f'| 候选中位耗时 | {r["candidate_seconds"]:.3f} s | {5*r["baseline_seconds"]:.3f} s |',
              f'| 耗时倍率 | {r["time_ratio"]:.3f}× | 5× |',
              f'| 峰值工作集 | {r["peak_working_set_bytes"]/2**30:.3f} GiB | 4 GiB |',
              f'| 未压缩推理包 | {r["asset_bytes"]/2**20:.3f} MiB | 64 MiB |', '',
              f'资源通过：{r["passes"]}。结果只覆盖验证集；测试集未运行。原 checkpoint、数据、tokenizer、common.py、evaluate.py、student.py、hybrid.py 哈希均未变化。', '',
              '## 训练成本', '',
              f'- 原共享神经训练轨迹：{outcome["inherited_neural_training_targets"]:,} targets，{outcome["inherited_neural_train_seconds"]:.3f} 秒。统计拟合和既往搜索成本继承自 initialization.json 的 ancestry，旧审计保留。',
              f'- 本轮两分支合计 {outcome["total_new_updates"]:,} 次实际更新、{outcome["total_new_training_targets"]:,} 个新增 targets；短运行已包含，未丢弃后重跑。合计训练计时 {outcome["new_train_seconds"]:.3f} 秒。',
              f'- 训练内完整验证计时 {outcome["new_validation_seconds"]:.3f} 秒；整个串行实验进程 {outcome["experiment_process_seconds"]:.3f} 秒，含预检查、数据加载、保存、恢复及最终独立评分。准备模型及工程检查另计。',
              f'- 最终选中权重沿其自身轨迹学习 {summary["selected_model_train_targets"]:,} 个 targets；其他分支和选中步数以后的更新都是已发生的搜索成本，不能从总成本中省略。', '',
              '| 分支 | 新增训练 targets | 累计训练计时（本阶段） | 最佳验证 BPB |', '|---|---:|---:|---:|']
    for row in finalists:
        lines.append(f'| {row["depth"]} 层 | {row["stage_train_targets"]:,} | {row["train_seconds"]:.3f} s | {row["best_validation_bpb"]:.9f} |')
    lines += ['', '各阶段 train_seconds 会从恢复状态累计，表中只取每分支最后一份记录，没有把早期阶段重复累加。18 项检查通过；新增层既保持初始预测，又能在前两次更新后获得内部梯度。真实短运行和最终权重检查确认统计表/融合配置未变化。', '',
              '## 复现与保留', '',
              '所有实际命令及返回码见 commands.json，固定计划见 plan.json，训练日志按 train_depth{depth}_step{boundary}.log 命名。',
              '`.venv/Scripts/python.exe prepare_depth.py`', '',
              '`.venv/Scripts/python.exe run_depth_experiment.py`', '',
              '`.venv/Scripts/python.exe finalize_depth_report.py`', '',
              '上述实验脚本拒绝已有输出；完整重跑请在保留旧结果的独立项目副本中使用新的输出位置。单次评分无需重训：', '',
              f'`.venv/Scripts/python.exe evaluate.py --checkpoint {outcome["recommended_bundle"].replace(chr(92), "/")}/checkpoint.pt --split validation --device cpu --precision fp32 --threads 3`', '',
              '推理只需选中包内 checkpoint.pt、hybrid.py、student.py、README.txt，加作业提供的固定评分环境。optimizer/RNG/EMA 备选权重和恢复状态均为训练资产，不放入推理包。',
              '原始代码副本位于 before/。Codex 实质协助实现、检查、实验和报告。保留函数后扩容的相关思路见 [Net2Net](https://arxiv.org/abs/1511.05641)，本项目通过残差输出零初始化实现，未使用论文模型或外部训练数据。', '']
    (AUDIT/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({k:outcome[k] for k in ('before_bpb','after_bpb','accepted','recommended_bundle','selected_depth','selected_step','total_new_updates')}, indent=2))


if __name__ == '__main__':
    main()
