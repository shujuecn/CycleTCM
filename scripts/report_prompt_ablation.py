"""Audit full-test E5a runs and report paired subject bootstrap effects."""

import argparse
import csv
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from report_reproduction import predictions, paired_bootstrap
from train.data import LABELS
from train.evaluation import metrics
from utils.experiment import sha256, write_json

KEYS = ['syndrome_acc', 'syndrome_f1', 'organ_acc', 'organ_f1']


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def grouped_counts(runs):
    counts = []
    for run in runs:
        rows = [run['predictions'][name] for name in sorted(run['predictions'])]
        y = np.asarray([r['labels'] for r in rows], dtype=bool)
        p = np.asarray([r['probabilities'] for r in rows]) > .5
        assert len({r['subject_id'] for r in rows}) == len(rows) == 895
        counts.append(np.stack([y&p, ~y&~p, ~y&p, y&~p], axis=-1))
    return np.asarray(counts, dtype=np.int32)


def count_metrics(counts):
    tp, tn, fp, fn = np.moveaxis(counts, -1, 0)
    acc = (tp+tn)/(tp+tn+fp+fn)
    f1 = np.divide(2*tp, 2*tp+fp+fn, out=np.zeros_like(tp, dtype=float), where=(2*tp+fp+fn)>0)
    return np.stack([acc[..., :8].mean(-1), f1[..., :8].mean(-1),
                     acc[..., 8:].mean(-1), f1[..., 8:].mean(-1)], axis=-1) * 100


def mean_seed_bootstrap(first, second, iterations=10000):
    """Resample subjects identically across models; average selected seed metrics."""
    a, b = grouped_counts(first), grouped_counts(second)
    rng = np.random.default_rng(20261007)
    differences = []
    for start in range(0, iterations, 100):
        draws = rng.integers(895, size=(min(100, iterations-start), 895))
        ma = count_metrics(a[:, draws].sum(axis=2)).mean(axis=0)
        mb = count_metrics(b[:, draws].sum(axis=2)).mean(axis=0)
        differences.append(mb-ma)
    differences = np.concatenate(differences)
    effect = (count_metrics(b.sum(axis=1))-count_metrics(a.sum(axis=1))).mean(axis=0)
    return {'definition': 'second minus first; paired subject resampling, mean of the selected fixed training seed metrics',
            'iterations': iterations, 'bootstrap_seed': 20261007, 'subjects': 895,
            'seeds': [r['summary']['config']['seed'] for r in first],
            'effect_pp': dict(zip(KEYS, effect.tolist())),
            'ci95_pp': dict(zip(KEYS, np.percentile(differences, [2.5,97.5], axis=0).T.tolist()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--baseline-suite', type=Path, default=ROOT/'outputs/reproduction/20261005_042629_774918_suite')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--report-path', type=Path, default=ROOT/'reports/supplemental_validation_report.md')
    args = parser.parse_args()
    suite = args.suite.resolve()
    state = json.loads((suite/'suite_status.json').read_text())
    selected_seeds = sorted(state.get('seeds') or sorted({entry['seed'] for entry in state['queue']}))
    assert selected_seeds and state['status'] == 'complete'
    assert all(entry['seed'] in selected_seeds for entry in state['queue'])
    variants = ['P0'] + list(state['features'])
    output = args.output_dir.resolve() if args.output_dir else ROOT/'reports/prompt_ablation'/suite.name
    output.mkdir(parents=True, exist_ok=True)
    entries = [{'variant':'P0','seed':json.loads((p/'config.json').read_text())['seed'],'run':str(p)}
               for p in args.baseline_suite.iterdir() if p.is_dir() and (p/'config.json').exists()
               and json.loads((p/'config.json').read_text())['model'] == 'full'
               and json.loads((p/'config.json').read_text())['seed'] in selected_seeds] + state['queue']
    runs, tables, per_class, manifest = [], [], [], []
    reference = reference_data = None
    for entry in entries:
        path = Path(entry['run'])
        summary = json.loads((path/'summary.json').read_text())
        config = summary['config']
        assert not summary['engineering_only'] and config['limit'] == 0 and config['model'] == 'full'
        assert config.get('loss', 'bce') == 'bce' and config.get('mllm_input_dim', 2560) == 2560
        data = json.loads((path/'data_manifest.json').read_text())
        shared_data = {k:v for k,v in data.items() if k != 'mllm_features_sha256'}
        if reference_data is None: reference_data = shared_data
        assert shared_data == reference_data
        assert json.loads((path/'status.json').read_text())['status'] == 'complete'
        indexed = predictions(path/'predictions/test.jsonl')
        identity = {name:(r['subject_id'], r['labels']) for name,r in indexed.items()}
        if reference is None: reference = identity
        assert identity == reference
        assert all(r['label_order'] == LABELS and r['checkpoint_sha256'] == summary['checkpoint_sha256'] for r in indexed.values())
        measured = metrics([r['labels'] for r in indexed.values()], [r['probabilities'] for r in indexed.values()])
        saved = json.loads((path/'metrics/test.json').read_text())
        assert measured['per_class'] == saved['per_class']
        row = {'variant':entry['variant'], 'seed':entry['seed'], 'best_epoch':summary['best_epoch'],
               'stop_reason':summary['stop_reason'], 'test_subjects':895}
        for key in KEYS:
            task, metric = key.split('_')
            assert measured[task][metric] == saved[task][metric] == summary['test'][task][metric]
            row[key+'_percent'] = measured[task][metric]*100
        row['checkpoint_sha256'] = summary['checkpoint_sha256']
        tables.append(row)
        runs.append({'variant':entry['variant'], 'path':path, 'summary':summary, 'metrics':measured, 'predictions':indexed})
        for item in measured['per_class']:
            per_class.append({'variant':entry['variant'],'seed':entry['seed'], **{k:v for k,v in item.items() if not k.startswith('paper_') and not k.startswith('delta_')}})
        with (path/'history.csv').open() as f: history = list(csv.DictReader(f))
        manifest.append({'variant':entry['variant'], 'seed':entry['seed'], 'run':str(path),
                         'config':config, 'checkpoint_sha256':summary['checkpoint_sha256'],
                         'resume':json.loads((path/'resume.json').read_text()) if (path/'resume.json').exists() else None,
                         'epochs':len(history), 'training_seconds_recorded':sum(float(r['seconds_with_checkpoint']) for r in history if r.get('seconds_with_checkpoint')),
                         'epochs_missing_wall_time':sum(not r.get('seconds_with_checkpoint') for r in history),
                         'files_sha256':{name:sha256(path/name) for name in ('summary.json','config.json','data_manifest.json','metrics/test.json','predictions/test.jsonl','history.csv','environment.json')}})
    grouped = {variant:sorted([r for r in runs if r['variant']==variant],key=lambda r:r['summary']['config']['seed']) for variant in variants}
    assert all([r['summary']['config']['seed'] for r in group] == selected_seeds for group in grouped.values())
    # Only feature path may vary; default additions in the shared trainer preserve P0's protocol.
    scientific_keys = ['profile','epochs','batch_size','precision','learning_rate','weight_decay','patience','min_delta','init','normalize','scheduler_monitor','limit','data_dir','feature_file','label_dir']
    assert all({k:r['summary']['config'][k] for k in scientific_keys} == {k:runs[0]['summary']['config'][k] for k in scientific_keys} for r in runs)
    for variant in state['features']:
        path = Path(state['features'][variant])
        assert sha256(path) == state['features_sha256'][variant]
        assert all(json.loads((r['path']/'data_manifest.json').read_text())['mllm_features_sha256'] == state['features_sha256'][variant] for r in grouped[variant])
    stats = {variant:{key:{'mean':float(np.mean([r['summary']['test'][key.split('_')[0]][key.split('_')[1]]*100 for r in group])),
                          'std_sample':float(np.std([r['summary']['test'][key.split('_')[0]][key.split('_')[1]]*100 for r in group],ddof=1)) if len(group) > 1 else 0.0}
                      for key in KEYS} for variant,group in grouped.items()}
    effects = {}
    comparisons = [('A0','A1'),('A1','A2'),('P0','A0'),('A0','A2')]
    if 'A3' in grouped:
        comparisons += [(v, 'A3') for v in ('P0', 'A0', 'A1', 'A2')]
    for first, second in comparisons:
        key = second+'_minus_'+first
        effects[key] = mean_seed_bootstrap(grouped[first], grouped[second])
        effects[key]['per_seed'] = {str(seed):paired_bootstrap(a['path']/'predictions/test.jsonl', b['path']/'predictions/test.jsonl',seed=20261007)
                                    for seed,a,b in zip(selected_seeds, grouped[first], grouped[second])}
    prompts = {v:{'metadata':json.loads((Path(state['features'][v]).parent/'metadata.json').read_text()),
                  'provenance':json.loads((Path(state['features'][v]).parent/'extraction_provenance.json').read_text())}
               for v in state['features']}
    reference_prompt = prompts['A0']
    for prompt in prompts.values():
        assert prompt['provenance']['versions'] == reference_prompt['provenance']['versions']
        assert all(reference_prompt['provenance']['model_files'].get(k) == v
                   for k, v in prompt['provenance']['model_files'].items())
        assert all(prompt['metadata'][k] == reference_prompt['metadata'][k]
                   for k in ('model_dir', 'images_dir', 'pooling', 'add_generation_prompt', 'dtype', 'target'))
    write_csv(output/'main_results.csv', tables)
    write_csv(output/'per_class_results.csv', per_class)
    write_json(output/'multi_seed.json', stats)
    write_json(output/'paired_bootstrap.json', effects)
    write_json(output/'source_manifest.json', manifest)
    write_json(output/'prompt_versions.json', prompts)
    write_json(output/'suite_status.json', state)
    figure_dir = output/'figures'; figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax, task in zip(axes, ('syndrome','organ')):
        values = [stats[v][task+'_f1']['mean'] for v in variants]
        sd = [stats[v][task+'_f1']['std_sample'] for v in variants]
        ax.errorbar(range(len(variants)),values,yerr=sd,fmt='o',capsize=5)
        for i,v in enumerate(variants):
            ax.scatter([i] * len(grouped[v]), [r['summary']['test'][task]['f1']*100 for r in grouped[v]], s=16)
        ax.set_xticks(range(len(variants)),variants); ax.set_ylabel('Macro positive-class F1 (%)'); ax.set_title(task.capitalize()+'; mean ± sample SD' if len(selected_seeds) > 1 else task.capitalize()+f'; seed {selected_seeds[0]}')
    fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(figure_dir/f'prompt_f1.{suffix}',dpi=180)
    plt.close(fig)
    seed_text = ','.join(map(str, selected_seeds))
    names = '/'.join(state['features'])
    lines = ['### E5a：提示词替换', '',
             f'{names} 均完成 5109 张图像的 Qwen3-VL-4B-Instruct 特征抽取和 full 模型训练，训练 seeds={seed_text}；P0 复用既有对应种子的正式复现结果。固定 Qwen3-VL-4B-Instruct 本地权重、BF16 单次前向、最后一层全序列 masked mean、2560 维特征、224×224 图像和原有受试者划分。full 模型使用 FP32、Adam、batch 32、最多 200 epochs、patience 50 和验证集任务平均 Acc 选模；测试为全部 895 位受试者，固定阈值 >0.5。', '',
             'A0/A1/A2 的 system/user 文本逐字取自验证方案附录 B，图片位于 user 消息内且在文本之前。本实验不执行生成 JSON 或显式反思，只衡量提示词对隐状态特征与下游分类的影响。归档 A0 含显式标签清单，A1 以标签定义代替清单且新增目标句，A1/A2 的开头目标句也不同，因此 A1−A0 和 A2−A1 不能完全排除措辞或长度效应；P0−A0 同时变化语言、粒度、格式和提示内容，仅作整体比较。', '']
    if 'A3' in grouped:
        prompt = prompts['A3']['metadata']
        lines += ['A3 是用户指定的无关文本控制，完整替换 system/user 文本，不附加原有 TCM_PRIOR、标签清单或医学知识；图像输入和提取方式保持不变。系统提示词为：', '',
                  '```text', prompt['system_prompt'], '```', '', '用户提示词为：', '', '```text', prompt['user_prompt'], '```', '',
                  'A3 同时改变提示内容和长度。全序列 masked mean 会随文本 token 数量改变图像与文本的池化比例，因此它检验的是整组无关提示版本的效果，不能独立归因于医学语义有无。', '']
        audit_path = output/'A3_feature_audit.json'
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            assert audit['feature_sha256'] == state['features_sha256']['A3']
            lengths = '、'.join(f'{v}={"/".join(audit["sequence_token_counts"][v])}' for v in state['features'])
            lines += [f'全量核查确认 5109 张图像字节与 A0/A1/A2 相同，A3 特征均为有限的 2560 维向量且聚合文件与逐图记录一致。含图像和聊天模板的序列 token 数为 {lengths}（每组所有图像相同），进一步说明长度因素不可忽略。核查记录为 `A3_feature_audit.json`。', '']
    lines += ['| 提示词 | 证候 Acc (%) | 证候 F1 (%) | 脏腑 Acc (%) | 脏腑 F1 (%) |', '| --- | ---: | ---: | ---: | ---: |']
    for v in variants:
        cells = [f'{stats[v][k]["mean"]:.2f}' if len(selected_seeds) == 1 else f'{stats[v][k]["mean"]:.2f} ± {stats[v][k]["std_sample"]:.2f}' for k in KEYS]
        lines.append('| '+v+' | '+' | '.join(cells)+' |')
    if 'A3' in grouped:
        selected = grouped['A3'][0]['summary']
        lines += ['', f'A3 的验证集选中 epoch={selected["best_epoch"]}，停止原因为 `{selected["stop_reason"]}`；测试指标均来自该最佳 checkpoint。']
    summary_note = (f'结果只使用固定 training seed={selected_seeds[0]}；表中没有训练种子间标准差，不能据此估计训练随机性的总体不确定性。' if len(selected_seeds) == 1 else '先逐标签计算正类 F1，再在 8 项证候／5 项脏腑内取平均，最后对固定训练种子取均值和样本标准差。')
    ci_note = '按受试者配对重采样 10,000 次；该 CI 衡量固定训练种子已训练模型的测试样本不确定性，不包含重新训练的种子总体不确定性。'
    figure_path = os.path.relpath(figure_dir/'prompt_f1.png', args.report_path.resolve().parent)
    lines += ['', summary_note,'', f'![E5a 提示词 F1]({figure_path})','',
              ci_note+'逐种子区间保存在 paired_bootstrap.json。以下差值均为后一配置减前一配置；CI 跨零时不宣称有效提升，也不证明两者等效。多项比较未进行校正，显著结果按探索性证据解读。','',
              '| 比较 | 证候 F1 Δ (pp) [95% CI] | 脏腑 F1 Δ (pp) [95% CI] |','| --- | ---: | ---: |']
    for key,value in effects.items():
        cells = []
        for k in ('syndrome_f1','organ_f1'):
            low,high = value['ci95_pp'][k]
            cells.append(f'{value["effect_pp"][k]:+.2f} [{low:+.2f}, {high:+.2f}]')
        lines.append('| '+key.replace('_minus_', '−')+' | '+' | '.join(cells)+' |')
    if 'A3' in grouped:
        lines += ['', 'A3 相对于此前提示版本的结果：', '']
        for first in ('P0', 'A0', 'A1', 'A2'):
            effect = effects['A3_minus_'+first]
            parts = []
            for task, label in [('syndrome', '证候'), ('organ', '脏腑')]:
                key = task+'_f1'
                low, high = effect['ci95_pp'][key]
                verdict = '区间全为正' if low > 0 else '区间全为负' if high < 0 else '区间跨零'
                parts.append(f'{label} F1 {effect["effect_pp"][key]:+.2f} pp（{verdict}）')
            lines.append(f'- A3−{first}：'+ '；'.join(parts)+'。')
        lines += ['', '上述变化只对应当前 seed 的已训练模型；即使无关提示不差于医学提示，也不能据此证明医学知识无效或模型未使用图像。所有配置均保留图像输入，并通过有监督 full 分类器训练。', '']
    class_comparisons = [('A0','A1'),('A1','A2')]
    if 'A3' in grouped:
        class_comparisons.append(('P0', 'A3'))
    for first,second in class_comparisons:
        a = np.mean([[c['f1']*100 for c in r['metrics']['per_class']] for r in grouped[first]],axis=0)
        b = np.mean([[c['f1']*100 for c in r['metrics']['per_class']] for r in grouped[second]],axis=0)
        indices = np.argsort(np.abs(b[:8]-a[:8]))[::-1][:3]
        lines.append(f'{second}−{first} 的证候变化最大的三个标签：'+ '；'.join(f'{LABELS[i]} {b[i]-a[i]:+.2f} pp' for i in indices)+'。逐类 Acc、F1、支持度和混淆矩阵见 per_class_results.csv。')
        effect = effects[second+'_minus_'+first]
        low,high = effect['ci95_pp']['syndrome_f1']
        verdict = '观察到正向差异，但受限于固定训练种子、提示措辞和长度变化及未校正多重比较。' if low > 0 else '观察到负向差异；该提示版本没有提高当前协议下的证候 F1。' if high < 0 else '区间跨零，当前结果不足以宣称该提示版本有效提高证候 F1，也不足以证明二者等效。'
        lines.append(verdict)
        lines.append('')
    spleen = grouped['P0'][0]['metrics']['per_class'][10]
    lines += [f'Spleen 的测试支持度为阳性 {spleen["positive"]}、阴性 {spleen["negative"]}，严重不平衡；其 F1 不应独立用来说明提示词的临床知识价值。', '',
              f'完整 E5a 产物（主结果、逐类结果、配对 bootstrap、prompt 原文与哈希、来源清单和图表）保存在 `{output.relative_to(ROOT)}/`。完整特征、逐图预测和权重保留在本地 data/ 与 outputs/。', '']
    if 'A3' in grouped:
        lines += ['在已有 A0/A1/A2 suite 上补充 A3 的复现命令：', '', '```bash',
                  'uv run --no-sync python scripts/extract_prompt_features.py \\',
                  '  --variant A3 --model-dir /path/to/Qwen3-VL-4B-Instruct \\',
                  '  --images-dir data/processed/CycleTCM/images \\',
                  '  --output-dir data/features/prompt_20261007_A3 --resume',
                  'uv run --no-sync python scripts/run_prompt_ablation.py --seeds 42 --jobs 1 \\',
                  '  --variants A3 --features data/features/prompt_20261007_A3/all_features.json \\',
                  '  --resume-suite '+str(suite.relative_to(ROOT)),
                  'uv run --no-sync python scripts/report_prompt_ablation.py --suite '+str(suite.relative_to(ROOT)),
                  '```', '']
    report = args.report_path.read_text()
    before, heading, after = report.partition('### E5a：提示词替换')
    assert heading and '\n## ' in after, 'Unified report must contain an E5a section followed by another top-level section'
    suffix = after[after.index('\n## '):]
    args.report_path.write_text(before+'\n'.join(lines)+suffix)
    print(f'REPORT {args.report_path}', flush=True)


if __name__ == '__main__':
    main()
