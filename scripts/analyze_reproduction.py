"""Generate a timestamped scientific report from completed, audited suite runs."""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from report_reproduction import PAPER, build_report, predictions
from train.data import LABELS, split_records
from train.evaluation import PAPER_F1, metrics
from utils.experiment import run_directory, sha256, write_json

MODELS = ['B', 'visual', 'full']
COLORS = {'B': '#8497a6', 'visual': '#227c9d', 'full': '#b94f55'}
EXTRA_FIGURES = ['seed_comparison', 'ablation_comparison', 'paired_effects',
                 'class_balance_errors', 'per_class_differences']


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, directory, name):
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(directory / f'{name}.{suffix}', dpi=200, bbox_inches='tight')
    plt.close(fig)


def load_runs(suite):
    state = json.loads((suite / 'suite_status.json').read_text())
    runs, manifest = [], []
    for item in state['queue']:
        if item['status'] != 'complete':
            continue
        path = Path(item['run'])
        assert json.loads((path / 'status.json').read_text())['status'] == 'complete'
        summary = json.loads((path / 'summary.json').read_text())
        measured = json.loads((path / 'metrics/test.json').read_text())
        assert not summary['engineering_only'] and summary['stop_reason'] != 'independent_evaluation'
        assert summary['config']['model'] == item['model'] and summary['config']['seed'] == item['seed']
        indexed = predictions(path / 'predictions/test.jsonl')
        rows = [indexed[name] for name in sorted(indexed)]
        assert len({row['subject_id'] for row in rows}) == 895
        assert all(row['label_order'] == LABELS and row['split'] == 'test'
                   and row['checkpoint_sha256'] == summary['checkpoint_sha256'] for row in rows)
        recomputed = metrics([row['labels'] for row in rows], [row['probabilities'] for row in rows])
        assert measured['checkpoint_sha256'] == summary['checkpoint_sha256']
        assert recomputed['per_class'] == measured['per_class']
        for task in ('syndrome', 'organ'):
            for metric in ('acc', 'f1', 'sen', 'pre'):
                assert recomputed[task][metric] == measured[task][metric] == summary['test'][task][metric]
        with (path / 'history.csv').open(newline='') as handle:
            history = list(csv.DictReader(handle))
        assert [int(row['epoch']) for row in history] == list(range(len(history)))
        environment = json.loads((path / 'environment.json').read_text())
        runs.append({'path': path, 'summary': summary, 'metrics': measured,
                     'predictions': indexed, 'history': history, 'environment': environment})
        manifest.append({'model': item['model'], 'seed': item['seed'], 'run': str(path),
                         'checkpoint_sha256_recorded': summary['checkpoint_sha256'],
                         'environment': environment,
                         'files_sha256': {name: sha256(path / name) for name in (
                             'summary.json', 'metrics/test.json', 'predictions/test.jsonl',
                             'config.json', 'data_manifest.json', 'history.csv', 'environment.json')}})
    reference = runs[0]['predictions']
    for run in runs:
        assert {name: (row['subject_id'], row['labels']) for name, row in run['predictions'].items()} == {
            name: (row['subject_id'], row['labels']) for name, row in reference.items()}
    return state, runs, manifest


def supplemental_figures(output, rows, runs, intervals):
    figures = output / 'figures'
    grouped = {model: [row for row in rows if row['model'] == model] for model in MODELS}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, task in zip(axes, ['syndrome', 'organ']):
        for position, model in enumerate(MODELS):
            group = grouped[model]
            values = [row[f'{task}_f1_percent'] for row in group]
            ax.errorbar(position, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0,
                        fmt='o', capsize=5, color=COLORS[model], markersize=8)
            for offset, row in zip(np.linspace(-.16, .16, len(group)), group):
                ax.scatter(position + offset, row[f'{task}_f1_percent'], color=COLORS[model], s=24)
                ax.annotate(str(row['seed']), (position + offset, row[f'{task}_f1_percent']),
                            xytext=(3, -11), textcoords='offset points', fontsize=8)
        ax.axhline(PAPER['full'][1 if task == 'syndrome' else 3], color='black',
                   linestyle='--', alpha=.6, label='Paper full model reference')
        ax.set_xticks(range(3), [f'{model} (n={len(grouped[model])})' for model in MODELS])
        ax.set_xlim(-.4, 2.45)
        ax.set_ylabel('Macro positive-class F1 (%)')
        ax.set_title(task.capitalize() + ': mean ± sample SD; points are seeds')
        ax.legend(fontsize=8)
    save_figure(fig, figures, 'seed_comparison')

    ablation = [next(row for row in rows if row['model'] == model and row['seed'] == 42) for model in PAPER]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task, index in zip(axes, ['syndrome', 'organ'], [1, 3]):
        measured = [row[f'{task}_f1_percent'] for row in ablation]
        target = [PAPER[row['model']][index] for row in ablation]
        ax.plot(range(6), target, 'o--', color='#78828a', label='Paper Table 3 reference')
        ax.plot(range(6), measured, 'o-', color='#227c9d', label='Local seed 42')
        ax.set_xticks(range(6), ['B', 'B+A', 'B+U', 'B+M', 'B+A+U', 'B+A+U+M'], rotation=25)
        ax.set_title(task.capitalize())
        ax.set_ylabel('Macro positive-class F1 (%)')
        ax.legend(fontsize=8)
    save_figure(fig, figures, 'ablation_comparison')

    by_cell = {(row['model'], row['seed']): row for row in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, task in zip(axes, ['syndrome', 'organ']):
        for position, (key, value) in enumerate(intervals.items()):
            first, second, seed_text = key.split('_vs_')[0], key.split('_vs_')[1].split('_seed')[0], key.split('_seed')[1]
            seed = int(seed_text)
            delta = by_cell[second, seed][f'{task}_f1_percent'] - by_cell[first, seed][f'{task}_f1_percent']
            lower, upper = value['ci95_pp'][f'{task}_f1']
            ax.plot([lower, upper], [position, position], color=COLORS[first], lw=2)
            ax.scatter(delta, position, color=COLORS[first], s=35)
        ax.set_yticks(range(len(intervals)), list(intervals))
        ax.axvline(0, color='black', linestyle='--', lw=1)
        ax.set_xlabel('Full minus comparator F1 (percentage points)')
        ax.set_title(task.capitalize() + ': paired subject bootstrap 95% CI')
        ax.invert_yaxis()
    save_figure(fig, figures, 'paired_effects')

    selected = {run['summary']['config']['model']: run for run in runs
                if run['summary']['config']['seed'] == 42 and run['summary']['config']['model'] in ('visual', 'full')}
    per_class = selected['visual']['metrics']['per_class']
    rates = np.array([[row['sen'], row['spe']] for model in ('visual', 'full')
                      for row in selected[model]['metrics']['per_class']]).reshape(2, 13, 2)
    error = 100 * (1 - rates.transpose(1, 0, 2).reshape(13, 4))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [1, 1.8]})
    positions = np.arange(13)
    positive = [row['positive'] for row in per_class]
    negative = [row['negative'] for row in per_class]
    axes[0].barh(positions, positive, color='#227c9d', label='Positive')
    axes[0].barh(positions, negative, left=positive, color='#d6dfe5', label='Negative')
    axes[0].set_yticks(positions, LABELS)
    axes[0].set_xlim(0, 895)
    axes[0].set_xlabel('Test subjects')
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    im = axes[1].imshow(error, aspect='auto', vmin=0, vmax=100, cmap='YlOrRd')
    axes[1].set_xticks(range(4), ['Visual FNR', 'Visual FPR', 'Full FNR', 'Full FPR'])
    axes[1].set_yticks(positions, LABELS)
    for i in range(13):
        for j in range(4):
            axes[1].text(j, i, f'{error[i,j]:.1f}', ha='center', va='center', fontsize=8,
                         color='white' if error[i,j] > 65 else 'black')
    fig.colorbar(im, ax=axes[1], label='Conditional error rate (%)', fraction=.04)
    save_figure(fig, figures, 'class_balance_errors')

    visual_f1 = np.array([row['f1'] for row in selected['visual']['metrics']['per_class']]) * 100
    full_f1 = np.array([row['f1'] for row in selected['full']['metrics']['per_class']]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, delta, title in zip(axes, [full_f1 - PAPER_F1, full_f1 - visual_f1],
                              ['Full seed 42 minus paper full model', 'Full minus visual, matched seed 42']):
        ax.barh(positions, delta, color=['#b94f55' if value < 0 else '#227c9d' for value in delta])
        ax.set_yticks(positions, LABELS)
        ax.axvline(0, color='black', lw=.8)
        ax.set_xlabel('Positive-class F1 difference (pp)')
        ax.set_title(title)
        ax.invert_yaxis()
    save_figure(fig, figures, 'per_class_differences')


def paired_qualitative(paths):
    if not paths:
        return None
    metadata = [json.loads(path.read_text()) for path in paths]
    assert metadata[0]['reference_predictions_sha256'] == metadata[1]['reference_predictions_sha256']
    assert metadata[0]['labels_requested'] == metadata[1]['labels_requested'] == ['TonguePale', 'Crack', 'Heart', 'Kidney']
    for data in metadata:
        run = Path(data['run'])
        summary = json.loads((run / 'summary.json').read_text())
        assert summary['config']['seed'] == 42 and summary['checkpoint_sha256'] == data['checkpoint_sha256']
        saved = predictions(run / 'predictions/test.jsonl')
        assert all(row['probability'] == saved[row['image_file']]['probabilities'][LABELS.index(row['label'])]
                   for row in data['selected'])
    indexed = [{(row['label'], row['image_file']): row for row in data['selected']} for data in metadata]
    assert indexed[0].keys() == indexed[1].keys()
    output = run_directory(ROOT / 'outputs/qualitative', 'paired_confusion')
    model_names = [json.loads((Path(data['run']) / 'config.json').read_text())['model'] for data in metadata]
    assert model_names == ['visual', 'full']
    outcome = {(1, 1): 'TP', (0, 0): 'TN', (0, 1): 'FP', (1, 0): 'FN'}
    figures, cases = {}, []
    for label in metadata[0]['labels_requested']:
        choices = [row for row in metadata[0]['selected'] if row['label'] == label]
        if not choices:
            continue
        fig, axes = plt.subplots(len(choices), 3, figsize=(9, 3 * len(choices)), squeeze=False)
        for position, row in enumerate(choices):
            name, actual = row['image_file'], row['actual']
            cell = outcome[actual, row['reference_predicted']]
            with Image.open(row['source_image']) as image:
                axes[position, 0].imshow(image.convert('RGB'))
            axes[position, 0].set_title(f'{label}: reference {cell}; y={actual}\n{name}', fontsize=9)
            case = {'label': label, 'image_file': name, 'actual': actual, 'reference_outcome': cell}
            for column, (path, data, index, model) in enumerate(zip(paths, metadata, indexed, model_names), 1):
                match = index[label, name]
                assert match['actual'] == actual
                axes[position, column].imshow(plt.imread(path.parent / match['output']))
                axes[position, column].set_title(f'{model} seed 42: p={match["probability"]:.3f}', fontsize=9)
                case[model] = {key: match[key] for key in (
                    'probability', 'predicted', 'cam_probability', 'probability_difference', 'cam_prediction_agrees')}
                cam = np.load((path.parent / match['output']).with_suffix('.npy'))
                assert cam.shape == (224, 224) and np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
            for ax in axes[position]:
                ax.axis('off')
            cases.append(case)
        fig.tight_layout()
        filename = f'qualitative_{label}.png'
        fig.savefig(output / filename, dpi=180, bbox_inches='tight')
        plt.close(fig)
        figures[label] = str(output / filename)
    result = {'output': str(output), 'figures': figures, 'cases': cases,
              'source_metadata': [str(path) for path in paths],
              'source_metadata_sha256': [sha256(path) for path in paths],
              'max_cpu_probability_difference': max(row[model]['probability_difference'] for row in cases for model in model_names),
              'cpu_prediction_disagreements': sum(not row[model]['cam_prediction_agrees'] for row in cases for model in model_names)}
    write_json(output / 'metadata.json', result)
    return result


def paper_report(output, state, rows, runs, intervals, baseline, qualitative):
    groups = {model: [row for row in rows if row['model'] == model] for model in MODELS}
    by_cell = {(row['model'], row['seed']): row for row in rows}
    common = sorted({row['seed'] for row in groups['visual']} & {row['seed'] for row in groups['full']})
    difference = {task: np.mean([by_cell['full', seed][f'{task}_f1_percent']
                                - by_cell['visual', seed][f'{task}_f1_percent'] for seed in common])
                  for task in ('syndrome', 'organ')}
    stamp = datetime.now().astimezone().isoformat(timespec='seconds')
    complete = len(rows) == len(state['queue']) and state['status'] == 'complete'
    stage = '固定训练队列已全部完成' if complete else f'阶段性结果：{len(rows)}/{len(state["queue"])} 个运行完成'
    lines = [
        '# CycleTCM 的可追溯复现与误差分析',
        '', f'生成时间：{stamp}。{stage}。', '',
        '## 摘要', '',
        f'本文在 TongueDx2 的固定受试者划分上，对 CycleTCM 的全局—局部视觉融合、跨任务双向专家交互及多模态语义融合进行代码口径复现。训练、验证和测试分别包含 3371、843 和 895 张图像，测试集每位受试者对应一张图像。所有报告结果均来自完整测试集，模型选择仅依据验证集。当前已完成 {len(rows)} 个预先规定的运行。'
        f'在已完成的配对种子 {common} 上，完整多模态模型相对纯视觉模型的证候与脏腑 macro positive-class F1 平均变化分别为 {difference["syndrome"]:+.2f} 和 {difference["organ"]:+.2f} 个百分点。'
        '这一结果未重现原论文中多模态模块带来的总体增益。逐类错误分析表明，类别不均衡会掩盖阴性样本的识别不足；固定规则的 GradCAM 对照进一步呈现正确与错误预测的模型响应。复现结论限定于本次数据划分、实现假设和特征版本，不能据此归因于某个未单独检验的实现差异。',
        '', '关键词：舌图多标签分类；CycleTCM；复现；消融实验；配对 bootstrap；GradCAM。', '',
        '## 1. 引言', '',
        'CycleTCM 将舌图的八项证候相关属性与五项脏腑属性建模为联合多标签分类任务。原论文提出 AGLFF、UWBMoE 和冻结多模态大模型特征，以整合全局外观、区域信息和语义表示。本文旨在检验现有实现能否重现主要性能与模块贡献，而非通过测试集调参追逐论文表格。论文数值作为外部参考，所有本地数值均对应可追踪的权重、逐图概率和评估记录。',
        '', '## 2. 材料与方法', '', '### 2.1 数据与评估', '',
        '沿用 fold1：训练/验证/测试分别为 3371/843/895 张图像，包含 3004/751/895 位受试者，三组无受试者交叉。输入采用已有的 224×224 七视图区域产物；heart_lung 位于图像下方舌尖、kidney 位于上方，liver 对应图像右侧。所有区域与标签均通过数据完整性审计。',
        '', '每个类别独立计算二分类 Accuracy、positive-class F1、Sensitivity 和 Precision，再分别对八项证候、五项脏腑标签取算术平均。正类预测固定为 sigmoid(logit)>0.5。F1=2TP/(2TP+FP+FN)；零分母时按既有实现记为 0。报告同时保留阴性数、Specificity、MCC 与 AUC，避免仅由多数类表现解释结果。',
        '', '### 2.2 实现与训练协议', '',
        '本文采用 code_compat 协议。三分支输入为整舌 RGB、edge/body 的六通道拼接和四个脏腑区域的十二通道拼接。B 使用三分支原始 pooled 向量，A/U/M 分别控制 AGLFF、UWBMoE 与 MLLM 特征模块；global-only 与 MLLM-only 是额外诊断基线。B 的三分支定义属于复现工作假设，不能视为原论文明确披露的 baseline 定义。',
        '', '优化器为 Adam，初始学习率 2×10⁻⁴，weight decay 10⁻⁴，physical batch size=32，FP32；最多训练 200 轮，patience=50、min_delta=0.001。两个任务的加权 BCE 相加，正类权重为各 split 的 inverse positive ratio。ReduceLROnPlateau 监控 train loss，factor=0.3、patience=5。按两个任务 validation macro Accuracy 的均值保存最优模型。保留原代码的独立七视图随机增强、ToTensor 无 ImageNet normalization，以及仅全局分支的 ImageNet V1 初始化；未将候选 reviewed 协议混入主结果。',
        '', '冻结 Qwen3-VL-4B-Instruct，采用固定提示和有效 token 的 masked-mean hidden-state pooling，形成 2560 维特征。由于旧缓存与当前 Transformers 环境的数值行为不一致，本次重提全部 5109 个向量，单独保留新版本及权重、提示、输入和 processor 的 provenance。缓存 SHA256 为 e9fb55ce38532e73ac58c86e1f3b14e420693a44e74f9ea044496a6bdd545；这一变化是实验差异，尚不能据此解释性能差距。',
        '', '### 2.3 统计与解释方法', '',
        '六组消融统一使用 seed 42；B、visual 和 full 的随机种子预先固定为 42/43/44。多种子统计为算术均值±样本标准差，未齐全的模型明确注明 n。配对比较在同一个 seed 内对相同受试者重采样 10,000 次，随机种子为 20261005，每次重新计算 macro 指标，取百分位 95% 区间；差值定义为 full 减去比较模型。该区间描述固定已训练模型下测试受试者抽样的不确定性，不包含训练随机性，也未进行多重比较校正。',
        '', '定性分析预先固定展示 TonguePale、Crack、Heart、Kidney，以 visual/42 的保存预测为参考，在各标签的 TP/TN/FP/FN 中选择文件名词典序最早的样本，并在相同样本上比较 visual/42 和 full/42。样本选择仅服务于事后误差描述，不参与训练或阈值选择。GradCAM 对正类 logit 求导，目标层为全局分支 backbone_whole.layer4[-1]，逐图 ReLU 后归一化到 [0,1]，以 0.45 透明度叠加。负类及假阴性图仍显示正类响应，不将热图解释为负类证据。',
        '', '## 3. 结果', '', '### 3.1 主要模型与多种子稳健性', '',
        '| 模型 | 已完成 seeds / n | 证候 Acc (%) | 证候 F1 (%) | 脏腑 Acc (%) | 脏腑 F1 (%) |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        group = groups[model]
        values = []
        for key in ('syndrome_acc', 'syndrome_f1', 'organ_acc', 'organ_f1'):
            data = [row[f'{key}_percent'] for row in group]
            values.append(f'{np.mean(data):.2f} ± {np.std(data, ddof=1):.2f}' if len(data) > 1 else f'{data[0]:.2f}')
        lines.append(f'| {model} | {[row["seed"] for row in group]} / {len(group)} | ' + ' | '.join(values) + ' |')
    lines += ['| 论文 full 参考 | 未披露 seeds | 86.05 | 72.75 | 80.09 | 82.67 |', '',
              '表 1. 每个运行均评估全部 895 张测试图。不同 n 的均值不可作为严格配对效应；配对效应只使用双方均已完成的种子。', '',
              '![多种子结果](figures/seed_comparison.png)', '',
              '图 1. 点为独立训练种子，误差线为样本标准差；论文 full 的横线为外部参考，不能视为本地复现或置信区间。', '',
              '### 3.2 消融实验', '',
              '| 配置（seed 42） | 本地证候 F1 | 论文证候 F1 | 差值 (pp) | 本地脏腑 F1 | 论文脏腑 F1 | 差值 (pp) |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for model in PAPER:
        row = by_cell[model, 42]
        s, o = row['syndrome_f1_percent'], row['organ_f1_percent']
        lines.append(f'| {model} | {s:.2f} | {PAPER[model][1]:.2f} | {s-PAPER[model][1]:+.2f} | {o:.2f} | {PAPER[model][3]:.2f} | {o-PAPER[model][3]:+.2f} |')
    visual, full, base = (by_cell[model, 42] for model in ('visual', 'full', 'B'))
    lines += ['', f'在 seed 42 上，A+U 的纯视觉组合相对 B 的证候/脏腑 F1 变化为 '
              f'{visual["syndrome_f1_percent"]-base["syndrome_f1_percent"]:+.2f}/'
              f'{visual["organ_f1_percent"]-base["organ_f1_percent"]:+.2f} pp。'
              f'纯视觉模型与论文对应行相差 {visual["syndrome_f1_delta_pp"]:+.2f}/'
              f'{visual["organ_f1_delta_pp"]:+.2f} pp，而完整模型与论文对应行相差 '
              f'{full["syndrome_f1_delta_pp"]:+.2f}/{full["organ_f1_delta_pp"]:+.2f} pp。'
              '结果说明本次差距并非所有配置都以相同幅度出现，需按配置分别检验潜在差异来源。']
    lines += ['', '![消融对照](figures/ablation_comparison.png)', '',
              '图 2. 原论文与本地 seed 42 的六组模块组合。配置排列不是训练时间序列，折线仅用于展示模块组合之间的结果差异。单种子消融不能支持所有模块均具有稳定增益的结论。', '',
              '### 3.3 成对模型差异', '',
              '| 比较（full − 比较模型） | seed | 证候 ΔF1 [95% CI] (pp) | 脏腑 ΔF1 [95% CI] (pp) |',
              '| --- | ---: | ---: | ---: |']
    for key, value in intervals.items():
        first = key.split('_vs_')[0]
        seed = int(key.split('_seed')[1])
        cells = []
        for task in ('syndrome', 'organ'):
            delta = by_cell['full', seed][f'{task}_f1_percent'] - by_cell[first, seed][f'{task}_f1_percent']
            low, high = value['ci95_pp'][f'{task}_f1']
            cells.append(f'{delta:+.2f} [{low:+.2f}, {high:+.2f}]')
        lines.append(f'| full − {first} | {seed} | ' + ' | '.join(cells) + ' |')
    if all(value['ci95_pp'][f'{task}_f1'][1] < 0 for key, value in intervals.items()
           if key.startswith('visual_vs_full') for task in ('syndrome', 'organ')):
        lines += ['', '在所有已完成的 visual/full 配对种子上，两任务 F1 差值区间的上界均小于 0。这是在固定模型条件下对受试者抽样不确定性的描述，不能替代更多训练种子、其他划分或外部数据集的验证。']
    lines += ['', '![成对效应](figures/paired_effects.png)', '',
              '图 3. 受试者层面的配对差值及百分位 95% 区间，0 表示两模型指标相同。TVMoE 未提供同 split 预测，因此本图不复现论文对 TVMoE 的 bootstrap 比较。', '',
              '### 3.4 类别不均衡与逐类误差', '',
              f'以训练集多数标签构造的固定预测基线，在测试集的证候 Acc/F1 为 {baseline["syndrome"]["acc"]*100:.2f}%/{baseline["syndrome"]["f1"]*100:.2f}%，脏腑为 {baseline["organ"]["acc"]*100:.2f}%/{baseline["organ"]["f1"]*100:.2f}%。这一基线未使用测试标签选择预测规则。',
              'Spleen 的测试阳性为 889/895，阴性仅 6 张；始终预测阳性即可获得 Acc=99.33%、F1=99.66%，而 Specificity=0。FurThick 阳性为 870/895。因此，高 F1 本身不足以证明有效识别少数阴性样本。下图将阳性/阴性 support 与条件错误率共同展示：FNR=FN/(TP+FN)，FPR=FP/(TN+FP)。',
              '', '![类别构成与错误率](figures/class_balance_errors.png)', '',
              '图 4. seed 42 的测试类别构成及 visual/full 的条件错误率。每类具有不同分母，图中百分比不等于该类错误样本占全部测试图的比例。阴性支持量极少的类别，其 FPR 波动应谨慎解释。',
              '', '![逐类差异](figures/per_class_differences.png)', '',
              '图 5. full/42 相对论文 full 及本地 visual/42 的逐类 F1 差值；两种比较分别对应外部数值参考和同 split、同种子的本地配对。13 类的 support、混淆计数、Sensitivity、Specificity、MCC 与 AUC 见 per_class_analysis.csv。', '',
              '### 3.5 固定样本上的定性比较', '']
    if qualitative:
        lines += [
            f'共展示 {len(qualitative["cases"])} 个固定标签—样本组合及两种模型的响应，所有热图均由实际保存的最优权重计算。CPU 单图解释前向与原 GPU batch 评价存在数值差异，所选样本的最大概率绝对差为 {qualitative["max_cpu_probability_difference"]:.6g}，阈值分类不一致的模型—样本组合数为 {qualitative["cpu_prediction_disagreements"]}。表中预测及概率使用原正式测试记录，CPU 概率另存于 metadata.json，不覆盖正式结果。',
            '', '| 标签 | 样本代号 | visual 参考类别 | 真值 | visual p / 预测 | full p / 预测 |',
            '| --- | --- | --- | ---: | ---: | ---: |']
        case_ids = {(row['label'], row['image_file']): f'C{i:02d}' for i, row in enumerate(qualitative['cases'], 1)}
        for row in qualitative['cases']:
            case_id = case_ids[row['label'], row['image_file']]
            lines.append(f'| {row["label"]} | {case_id} | {row["reference_outcome"]} | {row["actual"]} | {row["visual"]["probability"]:.3f} / {row["visual"]["predicted"]} | {row["full"]["probability"]:.3f} / {row["full"]["predicted"]} |')
        lines += ['', '样本代号按本地 paired_confusion/metadata.json 的 cases 顺序对应；原始图像文件名及完整解释记录保留本地，不将逐图原始预测文件提交到 Git。']
        corrected = [row for row in qualitative['cases'] if row['visual']['predicted'] != row['actual']
                     and row['full']['predicted'] == row['actual']]
        worsened = [row for row in qualitative['cases'] if row['visual']['predicted'] == row['actual']
                    and row['full']['predicted'] != row['actual']]
        lines += ['', f'在这个按参考混淆类别分层选取的集合中，full 修正了 {len(corrected)} 个 visual 错例，同时新增 {len(worsened)} 个错误。由于集合人为平衡 TP/TN/FP/FN，这一计数不代表全测试集的错误率或增益。']
        for group, description in ((corrected, '修正'), (worsened, '新增错误')):
            if group:
                row = group[0]
                case_id = case_ids[row['label'], row['image_file']]
                lines += ['', f'例如，{case_id} 的 {row["label"]} 真值为 {row["actual"]}，'
                          f'visual 的概率为 {row["visual"]["probability"]:.3f}，full 为 {row["full"]["probability"]:.3f}，'
                          f'对应一次{description}。同一图像的不同标签须分别解释，不能将某个标签的改进外推到整张图像或总体模型。']
        for label, path in qualitative['figures'].items():
            relative = os.path.relpath(path, output)
            lines += ['', f'![{label} 固定样本对照]({relative})', '',
                      f'图 6（{label}）. 从上至下按参考模型的 TP/TN/FP/FN 展示实际存在的组合，列为原图、visual/42 与 full/42。类别描述属于 visual/42，full 的预测可不同。']
        lines += ['', '上述图像只支持有限样本上的模型响应描述。每张 CAM 独立归一化，颜色不能用于比较两模型的绝对证据强度；单一全局分支的 GradCAM 也不能表示完整三分支或 MLLM 路径的贡献。对错误样本保留热图与数值对照，但不把视觉上显著的区域认定为临床证据。定性原图与热图保留在本地 outputs/qualitative，不上传到 Git。']
    else:
        lines += ['定性材料尚未提供；本版本不展示占位热图。']
    lines += ['', '## 4. 讨论', '',
              '本次实现可完成全部样本的训练与评价，且保存的最佳权重可独立加载；但工程可重复运行与复现论文性能是两个不同的判断。已完成的配对结果没有支持完整多模态模型优于纯视觉模型的结论。可能相关的因素包括新旧 Qwen 特征数值行为、未披露的初始化或数据增强细节、B 的定义及优化设置；本次没有逐一控制这些因素，不能作因果归因，也不能据此否定方法在其他协议下的有效性。',
              '', '本研究仅采用一个 subject split 和少量预设种子，单种子消融尤其可能受到训练随机性的影响。部分阴性类别的支持量极少，macro positive-class F1 应与 Specificity、MCC 和混淆计数共同解读。测试集只用于锁定协议后的评价及事后描述，未据此更改缓存、模型超参数、分类阈值或筛选报告中的种子。',
              '', '原始 visual/43 中断目录曾在并发清理时被误删。续训命令、完整 132 轮历史、原套件 stdout、最终权重与全部预测仍在；原启动 environment/config 等副本已丢失，原 revision 无法据留存文件确证。此次续训真实性已由原 stdout 的 72 轮记录与继承历史逐项一致验证，但应保留这一 provenance 缺口，而不声称审计链完全无损。',
              '', '外部 Signnet、MIRnet、ILPnet、HWmixer、TDFnet 与 TVMoE 的同 split 实现和逐图预测尚未齐备；论文 Sankey 的 8×5 权重提取公式也未取得，因此未生成声称复现这两类实验的结果。MoE 的四个 expert 路由权重不能直接替代八个证候到五个器官的关联。本文没有增加未经定义的图或混入额外 late-fusion 实验。',
              '', '## 5. 结论', '',
              f'在本次固定 code_compat 协议和已完成的配对种子上，完整多模态模型相对纯视觉模型的两任务 F1 平均变化为 {difference["syndrome"]:+.2f}/{difference["organ"]:+.2f} pp，尚未复现论文报告的总体增益。'
              + ('主要结果、六组消融及三种子固定训练队列已完成；外部对比与 Sankey 仍属未完成内容。' if complete else '当前结果为阶段性快照，full/44 未完成时不将其视为三种子最终结论。')
              + '后续研究应通过只使用验证集的受控协议比较检验差异来源，并补充外部模型预测和解释方法定义。本文提供可复算的预测、统计表、图形及来源记录，为这些工作建立基准。',
              '', '## 参考材料', '',
              '1. Du Y., Lu C., Fu B., Fang Y., Shan C. MLLM-Enhanced Region-Aware Bidirectional Evidence-Based Model for Tongue Diagnosis. 本地提供的论文稿，出版信息未核实。[原文](../../../materials/0903_paper.pdf)。',
              '2. [复现计划](../../../docs/reproduction_plan.md)与[执行记录](../../../docs/reproduction_progress.md)。',
              '3. main_results.csv、ablations.csv、multi_seed.json、paired_bootstrap.json、resources.csv、source_manifest.json 和 analysis.json 为本报告的结构化依据。',
              '', '## 附录 A. 每个运行的结果', '',
              '| 模型 | seed | 最优 epoch（从 0 起） | 证候 Acc / F1 (%) | 脏腑 Acc / F1 (%) |',
              '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["model"]} | {row["seed"]} | {row["best_epoch"]} | {row["syndrome_acc_percent"]:.2f} / {row["syndrome_f1_percent"]:.2f} | {row["organ_acc_percent"]:.2f} / {row["organ_f1_percent"]:.2f} |')
    environment = runs[0]['environment']
    lines += ['', '## 附录 B. 运行环境与资源', '',
              f'训练设备为 {environment["gpu"]}，显存 {environment["gpu_memory_gib"]:.2f} GiB，CUDA build {environment["cuda_build"]}。依赖版本读取各运行的 environment.json，而非预先计划中的版本推测。首个运行记录：'
              + '；'.join(f'{key}={value}' for key, value in environment['versions'].items()) + '。',
              '', f'首个运行的 TF32 开关为 matmul={environment["matmul_tf32"]}、cuDNN={environment["cudnn_tf32"]}；FP32 指模型与训练张量精度，不等于所有 CUDA 内核均关闭 TF32。',
              '', '各模型实际 epoch 数、峰值 allocated 显存、每轮包含 checkpoint 的平均耗时及选中/最低验证 loss 见 resources.csv。中断续训的 history 包含继承轮次，不把各目录的墙钟时间直接相加。完成后仅保留 best checkpoint；训练中保留 last 以支持恢复。', '',
              '![训练轨迹](figures/training_curves.png)', '',
              '图 A1. 所有正式运行的验证 weighted BCE 与选模 Accuracy 轨迹。验证 loss 与选模指标不同，最小 loss 所在轮次不必等于保存的最优 epoch。', '']
    (output / 'paper_report.md').write_text('\n'.join(lines))


def publish(output):
    names = ['paper_report.md', 'reproduction_report.md', 'main_results.csv', 'ablations.csv',
             'multi_seed.json', 'paired_bootstrap.json', 'run_status.json', 'analysis.json',
             'source_manifest.json', 'per_class_analysis.csv', 'resources.csv']
    names += [f'figures/{name}.png' for name in ('macro_f1', 'per_class_f1', 'training_curves')]
    names += [f'figures/{name}.{suffix}' for name in EXTRA_FIGURES for suffix in ('png', 'pdf')]
    paths = [str((output / name).relative_to(ROOT)) for name in names]
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip()
    if branch != 'shujuecn':
        raise RuntimeError('Publishing requires shujuecn branch')
    subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
    subprocess.run(['git', 'commit', '--only', '-m', f'results: scientific reproduction analysis {output.name}', '--', *paths],
                   cwd=ROOT, check=True)
    subprocess.run(['git', 'push', 'fork', 'shujuecn'], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True)
    parser.add_argument('--qualitative-metadata', type=Path, nargs=2, metavar=('VISUAL', 'FULL'))
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    state, runs, manifest = load_runs(args.suite.resolve())
    output = run_directory(ROOT / 'reports/reproduction', 'analysis')
    rows = build_report([run['path'] for run in runs], output)
    intervals = json.loads((output / 'paired_bootstrap.json').read_text())
    config = runs[0]['summary']['config']
    train = split_records(config['feature_file'], config['label_dir'])['train']
    majority = np.array([[row[label] for label in LABELS] for row in train]).mean(0) > .5
    truth = [row['labels'] for row in runs[0]['predictions'].values()]
    baseline = metrics(truth, np.broadcast_to(majority.astype(float), (895, 13)))
    per_class, resources = [], []
    for run in runs:
        config = run['summary']['config']
        per_class += [{'model': config['model'], 'seed': config['seed'], **row} for row in run['metrics']['per_class']]
        history = run['history']
        seconds = [float(row['seconds_with_checkpoint']) for row in history if row.get('seconds_with_checkpoint')]
        peaks = [float(row['peak_allocated_gib']) for row in history if row.get('peak_allocated_gib')]
        resources.append({'model': config['model'], 'seed': config['seed'], 'epochs': len(history),
                          'best_epoch': run['summary']['best_epoch'],
                          'epoch_seconds_mean_with_checkpoint': float(np.mean(seconds)),
                          'peak_allocated_gib': max(peaks),
                          'best_selected_val_loss': float(history[run['summary']['best_epoch']]['val_loss']),
                          'minimum_val_loss': min(float(row['val_loss']) for row in history),
                          'minimum_val_loss_epoch': int(min(history, key=lambda row: float(row['val_loss']))['epoch'])})
    write_csv(output / 'per_class_analysis.csv', per_class)
    write_csv(output / 'resources.csv', resources)
    supplemental_figures(output, rows, runs, intervals)
    qualitative = paired_qualitative([path.resolve() for path in args.qualitative_metadata] if args.qualitative_metadata else None)
    write_json(output / 'source_manifest.json', {'suite': str(args.suite.resolve()), 'queue_snapshot': state,
               'script_sha256': sha256(Path(__file__)), 'runs': manifest})
    qualitative_summary = ({key: value for key, value in qualitative.items() if key != 'cases'}
                           | {'selected_cases': len(qualitative['cases'])}) if qualitative else None
    write_json(output / 'analysis.json', {'completed': len(rows), 'planned': len(state['queue']),
               'suite_status_at_snapshot': state['status'], 'majority_train_labels': majority.astype(int).tolist(),
               'majority_baseline': baseline, 'qualitative': qualitative_summary,
               'verification': {'all_895_unique_subjects': True, 'shared_labels_and_samples': True,
                                'predictions_recompute_saved_metrics_exactly': True,
                                'checkpoint_hash_references_consistent': True}})
    paper_report(output, state, rows, runs, intervals, baseline, qualitative)
    if args.publish:
        publish(output)
    print(f'ANALYSIS {output}', flush=True)


if __name__ == '__main__':
    main()
