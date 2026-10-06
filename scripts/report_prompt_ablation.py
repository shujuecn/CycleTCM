"""Audit full-test E5a runs and report paired subject bootstrap effects."""

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

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

VARIANTS = ['P0', 'A0', 'A1', 'A2']
KEYS = ['syndrome_acc', 'syndrome_f1', 'organ_acc', 'organ_f1']


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
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
    """Resample subjects identically across models and seeds; average seed metrics."""
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
    return {'definition': 'second minus first; paired subject resampling, mean of seedwise metrics; fixed training seeds',
            'iterations': iterations, 'bootstrap_seed': 20261007, 'subjects': 895,
            'seeds': [r['summary']['config']['seed'] for r in first],
            'effect_pp': dict(zip(KEYS, effect.tolist())),
            'ci95_pp': dict(zip(KEYS, np.percentile(differences, [2.5,97.5], axis=0).T.tolist()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--baseline-suite', type=Path, default=ROOT/'outputs/reproduction/20261005_042629_774918_suite')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    suite = args.suite.resolve()
    state = json.loads((suite/'suite_status.json').read_text())
    assert state['status'] == 'complete' and len(state['queue']) == 9
    output = args.output_dir.resolve() if args.output_dir else ROOT/'reports/prompt_ablation'/suite.name
    output.mkdir(parents=True, exist_ok=True)
    entries = [{'variant':'P0','seed':json.loads((p/'config.json').read_text())['seed'],'run':str(p)}
               for p in args.baseline_suite.iterdir() if p.is_dir() and (p/'config.json').exists()
               and json.loads((p/'config.json').read_text())['model'] == 'full'] + state['queue']
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
                         'epochs':len(history), 'training_seconds_recorded':sum(float(r['seconds_with_checkpoint']) for r in history if r.get('seconds_with_checkpoint')),
                         'epochs_missing_wall_time':sum(not r.get('seconds_with_checkpoint') for r in history),
                         'files_sha256':{name:sha256(path/name) for name in ('summary.json','config.json','data_manifest.json','metrics/test.json','predictions/test.jsonl','history.csv','environment.json')}})
    grouped = {variant:sorted([r for r in runs if r['variant']==variant],key=lambda r:r['summary']['config']['seed']) for variant in VARIANTS}
    assert all([r['summary']['config']['seed'] for r in group] == [42,43,44] for group in grouped.values())
    # Only feature path may vary; default additions in the shared trainer preserve P0's protocol.
    scientific_keys = ['profile','epochs','batch_size','precision','learning_rate','weight_decay','patience','min_delta','init','normalize','scheduler_monitor','limit','data_dir','feature_file','label_dir']
    assert all({k:r['summary']['config'][k] for k in scientific_keys} == {k:runs[0]['summary']['config'][k] for k in scientific_keys} for r in runs)
    for variant in ('A0','A1','A2'):
        path = Path(state['features'][variant])
        assert sha256(path) == state['features_sha256'][variant]
        assert all(json.loads((r['path']/'data_manifest.json').read_text())['mllm_features_sha256'] == state['features_sha256'][variant] for r in grouped[variant])
    stats = {variant:{key:{'mean':float(np.mean([r['summary']['test'][key.split('_')[0]][key.split('_')[1]]*100 for r in group])),
                          'std_sample':float(np.std([r['summary']['test'][key.split('_')[0]][key.split('_')[1]]*100 for r in group],ddof=1))}
                      for key in KEYS} for variant,group in grouped.items()}
    effects = {}
    for first, second in [('A0','A1'),('A1','A2'),('P0','A0'),('A0','A2')]:
        key = second+'_minus_'+first
        effects[key] = mean_seed_bootstrap(grouped[first], grouped[second])
        effects[key]['per_seed'] = {str(seed):paired_bootstrap(a['path']/'predictions/test.jsonl', b['path']/'predictions/test.jsonl',seed=20261007)
                                    for seed,a,b in zip((42,43,44), grouped[first], grouped[second])}
    prompts = {v:{'metadata':json.loads((Path(state['features'][v]).parent/'metadata.json').read_text()),
                  'provenance':json.loads((Path(state['features'][v]).parent/'extraction_provenance.json').read_text())}
               for v in ('A0','A1','A2')}
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
        values = [stats[v][task+'_f1']['mean'] for v in VARIANTS]
        sd = [stats[v][task+'_f1']['std_sample'] for v in VARIANTS]
        ax.errorbar(range(4),values,yerr=sd,fmt='o',capsize=5)
        for i,v in enumerate(VARIANTS):
            ax.scatter([i-.1,i,i+.1], [r['summary']['test'][task]['f1']*100 for r in grouped[v]], s=16)
        ax.set_xticks(range(4),VARIANTS); ax.set_ylabel('Macro positive-class F1 (%)'); ax.set_title(task.capitalize()+'; mean ± sample SD')
    fig.tight_layout()
    for suffix in ('png','pdf'): fig.savefig(figure_dir/f'prompt_f1.{suffix}',dpi=180)
    plt.close(fig)
    lines = ['# CycleTCM E5a 提示词替换实验结果','',f'生成时间：{datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")}。完成 A0/A1/A2 各 3 个种子，共 9 个正式运行；P0 复用既有 3 种子。','',
             '## 协议','', '固定 Qwen3-VL-4B-Instruct 本地权重、BF16 单次前向、最后一层全序列 masked mean、2560 维特征、224×224 原有图像与受试者划分。所有全量特征含 5109 张图；full 模型继续使用 FP32、Adam、batch 32、最多 200 epochs、patience 50 和验证集任务平均 Acc 选模；测试为全部 895 位受试者，固定阈值 >0.5。训练 seeds=42/43/44，除提示词产生的特征外，模型和训练配置相同。','',
             'A0/A1/A2 的 system/user 文本逐字取自验证方案附录 B。图片均在 user 消息内、位于文本之前，与原提取器顺序一致。没有生成 JSON 或执行显式反思，因此本实验衡量的是提示词对隐状态特征与下游分类的影响，不能代表 TongueBench 的生成判读指标。','',
             '归档 A0 含显式标签清单，A1 以标签定义代替清单且新增目标句；A1/A2 的开头目标句也不同。保留原文使其符合方案的归档复用要求，但存在这些伴随变化；A1−A0 应理解为归档知识提示版本效应，A2−A1 为归档反思提示版本效应，不能完全排除措辞或长度效应。P0−A0 同时变化语言、粒度、格式和提示内容，仅作整体比较。脏腑标签未出现在 A 组提示词，脏腑变化仅为次级读出。','',
             '## 三种子结果','', '| 提示词 | 证候 Acc (%) | 证候 F1 (%) | 脏腑 Acc (%) | 脏腑 F1 (%) |','| --- | ---: | ---: | ---: | ---: |']
    for v in VARIANTS:
        cells = [f'{stats[v][k]["mean"]:.2f} ± {stats[v][k]["std_sample"]:.2f}' for k in KEYS]
        lines.append('| '+v+' | '+' | '.join(cells)+' |')
    lines += ['', '先逐标签计算正类 F1，再在 8 项证候／5 项脏腑内取平均，最后对 3 个训练种子取均值和样本标准差。','', '![提示词三种子 F1](figures/prompt_f1.png)','',
              '## 配对差异','', '按受试者配对重采样 10,000 次，每次对三个固定训练种子分别计算指标后取均值；该 CI 衡量这三个已训练模型的测试样本不确定性，不包含重新训练的种子总体不确定性。逐种子区间保存在 paired_bootstrap.json。以下差值均为后一配置减前一配置；CI 跨零时不宣称有效提升。多项比较未进行校正，显著结果按探索性证据解读。','',
              '| 比较 | 证候 F1 Δ (pp) [95% CI] | 脏腑 F1 Δ (pp) [95% CI] |','| --- | ---: | ---: |']
    for key,value in effects.items():
        cells = []
        for k in ('syndrome_f1','organ_f1'):
            low,high = value['ci95_pp'][k]
            cells.append(f'{value["effect_pp"][k]:+.2f} [{low:+.2f}, {high:+.2f}]')
        lines.append('| '+key+' | '+' | '.join(cells)+' |')
    lines += ['', '## 逐类变化与解释','']
    for first,second in [('A0','A1'),('A1','A2')]:
        a = np.mean([[c['f1']*100 for c in r['metrics']['per_class']] for r in grouped[first]],axis=0)
        b = np.mean([[c['f1']*100 for c in r['metrics']['per_class']] for r in grouped[second]],axis=0)
        indices = np.argsort(np.abs(b[:8]-a[:8]))[::-1][:3]
        lines.append(f'{second}−{first} 的证候变化最大的三个标签：'+ '；'.join(f'{LABELS[i]} {b[i]-a[i]:+.2f} pp' for i in indices)+'。逐类 Acc、F1、支持度和混淆矩阵见 per_class_results.csv。')
        effect = effects[second+'_minus_'+first]
        low,high = effect['ci95_pp']['syndrome_f1']
        verdict = '观察到正向差异，但受限于三种子、归档措辞伴随变化及未校正多重比较。' if low > 0 else '观察到负向差异；该归档提示版本没有提高当前协议下的证候 F1。' if high < 0 else '区间跨零，当前结果不足以宣称该归档提示版本有效提高证候 F1，也不足以证明二者等效。'
        lines.append(verdict)
        lines.append('')
    spleen = grouped['P0'][0]['metrics']['per_class'][10]
    lines += [f'Spleen 的测试支持度为阳性 {spleen["positive"]}、阴性 {spleen["negative"]}，严重不平衡；其 F1 不应独立用来说明提示词的临床知识价值。','',
              '## 复现与产物','', '完整 feature JSON、逐图预测和权重保留在本地 data/ 与 outputs/，不推送受试者级资料。此目录保存完整汇总、逐类结果、prompt 原文与哈希、训练来源清单、配对区间以及图表。','',
              '```bash', 'for variant in A0 A1 A2; do', '  uv run --no-sync python scripts/extract_prompt_features.py \\', '    --variant "$variant" --model-dir /path/to/Qwen3-VL-4B-Instruct \\', '    --images-dir data/processed/CycleTCM/images \\', '    --output-dir "data/features/prompt_20261007_$variant" --resume', 'done', 'uv run --no-sync python scripts/run_prompt_ablation.py \\', '  --features data/features/prompt_20261007_A0/all_features.json \\', '             data/features/prompt_20261007_A1/all_features.json \\', '             data/features/prompt_20261007_A2/all_features.json',
              'uv run --no-sync python scripts/report_prompt_ablation.py --suite '+str(suite.relative_to(ROOT)), '```','']
    (output/'prompt_ablation_report.md').write_text('\n'.join(lines))
    print(f'REPORT {output}', flush=True)


if __name__ == '__main__':
    main()
