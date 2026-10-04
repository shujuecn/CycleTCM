"""Build tables, curves and paired subject bootstrap from actual saved predictions."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from train.data import LABELS
from train.evaluation import PAPER_ACC, PAPER_F1
from utils.experiment import run_directory, write_json, sha256

PAPER = {'B': (83.56,68.35,77.50,78.82), 'BA': (84.09,70.47,78.75,81.64),
         'BU': (84.86,69.83,78.86,80.77), 'BM': (84.18,70.69,78.59,81.33),
         'visual': (85.18,71.60,79.46,81.83), 'full': (86.05,72.75,80.09,82.67)}


def predictions(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == len({row['image_file'] for row in rows}) == 895
    return {row['image_file']:row for row in rows}


def paired_bootstrap(first_path, second_path, iterations=10000, seed=20261005):
    first, second = predictions(first_path), predictions(second_path)
    assert set(first) == set(second)
    names = sorted(first)
    subjects = sorted({first[name]['subject_id'] for name in names})
    index = {subject:i for i,subject in enumerate(subjects)}
    grouped = np.zeros((2,len(subjects),13,4),dtype=np.int64)
    for name in names:
        assert first[name]['labels'] == second[name]['labels']
        assert first[name]['subject_id'] == second[name]['subject_id']
        actual = np.asarray(first[name]['labels'],dtype=bool)
        for model, row in enumerate([first[name],second[name]]):
            predicted = np.asarray(row['probabilities']) > .5
            grouped[model,index[row['subject_id']]] += np.stack(
                [actual&predicted,~actual&~predicted,~actual&predicted,actual&~predicted],axis=-1)
    rng = np.random.default_rng(seed)
    differences = []
    for start in range(0,iterations,250):
        draws = rng.integers(len(subjects),size=(min(250,iterations-start),len(subjects)))
        totals = grouped[:,draws].sum(axis=2)
        tp, tn, fp, fn = np.moveaxis(totals,-1,0)
        acc = (tp+tn)/(tp+tn+fp+fn)
        f1 = np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0)
        values = np.stack([acc[:,:,:8].mean(-1),f1[:,:,:8].mean(-1),acc[:,:,8:].mean(-1),f1[:,:,8:].mean(-1)],axis=-1)
        differences.append((values[1]-values[0])*100)
    differences = np.concatenate(differences)
    return {'definition':'second minus first; paired subject resampling', 'iterations':iterations,'seed':seed,
            'subjects':len(subjects), 'first_predictions_sha256':sha256(first_path), 'second_predictions_sha256':sha256(second_path),
            'ci95_pp':dict(zip(['syndrome_acc','syndrome_f1','organ_acc','organ_f1'],np.percentile(differences,[2.5,97.5],axis=0).T.tolist()))}


def build_report(run_paths, output):
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    figures=output/'figures'
    figures.mkdir(exist_ok=True)
    rows, runs, states = [], [], []
    for path in run_paths:
        path=Path(path).resolve()
        status=json.loads((path/'status.json').read_text()) if (path/'status.json').is_file() else {'status':'unrecorded'}
        states.append({'run':str(path), **status})
        if not (path/'summary.json').is_file():
            continue
        summary=json.loads((path/'summary.json').read_text())
        if summary['engineering_only'] or 'test' not in summary or summary['stop_reason']=='independent_evaluation':
            continue
        measured=json.loads((path/'metrics/test.json').read_text())
        assert measured['samples']==895
        config=summary['config']
        row={'model':config['model'],'seed':config['seed'],'profile':config['profile'],'best_epoch':summary['best_epoch'],
             'stop_reason':summary['stop_reason'],'test_images':895,'run':str(path),
             'checkpoint_sha256':summary['checkpoint_sha256']}
        targets=PAPER.get(config['model'])
        for i,key in enumerate(['syndrome_acc','syndrome_f1','organ_acc','organ_f1']):
            task,metric=key.split('_')
            row[key+'_percent']=summary['test'][task][metric]*100
            row[key+'_paper_percent']=targets[i] if targets else None
            row[key+'_delta_pp']=row[key+'_percent']-targets[i] if targets else None
        rows.append(row)
        runs.append((path,summary,measured))
    write_json(output/'run_status.json',states)
    if rows:
        with (output/'main_results.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        with (output/'ablations.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(row for row in rows if row['model'] in PAPER and row['seed']==42)
        labels=[f'{row["model"]}/{row["seed"]}' for row in rows]
        fig,axes=plt.subplots(1,2,figsize=(max(10,len(rows)*.8),5))
        for ax,task,target in zip(axes,['syndrome','organ'],[72.75,82.67]):
            ax.bar(np.arange(len(rows)),[row[task+'_f1_percent'] for row in rows])
            ax.axhline(target,color='red',linestyle='--',label='Paper full model (reference)')
            ax.set_xticks(np.arange(len(rows)),labels,rotation=45,ha='right')
            ax.set_title(task+' macro positive-class F1')
            ax.set_ylabel('F1 (%)')
            ax.set_ylim(0,100)
            ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figures/'macro_f1.png',dpi=150)
        plt.close(fig)
        per_class=np.array([[item['f1']*100 for item in measured['per_class']] for _,_,measured in runs])
        fig,ax=plt.subplots(figsize=(14,max(3,len(rows)*.5)))
        im=ax.imshow(per_class,aspect='auto',vmin=0,vmax=100,cmap='viridis')
        ax.set_xticks(np.arange(13),LABELS,rotation=45,ha='right')
        ax.set_yticks(np.arange(len(rows)),labels)
        ax.set_title('Per-class positive F1 (%) — all 895 test images')
        for i in range(len(rows)):
            for j in range(13):
                ax.text(j,i,f'{per_class[i,j]:.1f}',ha='center',va='center',fontsize=7,color='white' if per_class[i,j]<50 else 'black')
        fig.colorbar(im,ax=ax)
        fig.tight_layout()
        fig.savefig(figures/'per_class_f1.png',dpi=150)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for path,summary,_ in runs:
            if not (path/'history.csv').is_file():
                continue
            with (path/'history.csv').open() as handle:
                history=list(csv.DictReader(handle))
            label=f'{summary["config"]["model"]}/{summary["config"]["seed"]}'
            axes[0].plot([int(r['epoch'])+1 for r in history],[float(r['val_loss']) for r in history],label=label)
            axes[1].plot([int(r['epoch'])+1 for r in history],[float(r['selection_acc'])*100 for r in history],label=label)
        for ax,title in zip(axes,['Validation weighted BCE','Validation task-average macro accuracy (%)']):
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figures/'training_curves.png',dpi=150)
        plt.close(fig)
    by_seed={(summary['config']['model'],summary['config']['seed']):path for path,summary,_ in runs}
    intervals={}
    for seed in [42,43,44]:
        for first,second in [('B','full'),('visual','full')]:
            if (first,seed) in by_seed and (second,seed) in by_seed:
                key=f'{first}_vs_{second}_seed{seed}'
                intervals[key]=paired_bootstrap(by_seed[first,seed]/'predictions/test.jsonl',by_seed[second,seed]/'predictions/test.jsonl')
    write_json(output/'paired_bootstrap.json',intervals)
    statistics=[]
    for model in ['B','visual','full']:
        group=[row for row in rows if row['model']==model]
        if len(group)>=2:
            statistics.append({'model':model,'seeds':[row['seed'] for row in group],
                               **{key:{'mean':float(np.mean([row[key] for row in group])),
                                       'std_sample':float(np.std([row[key] for row in group],ddof=1))}
                                  for key in ['syndrome_acc_percent','syndrome_f1_percent','organ_acc_percent','organ_f1_percent']}})
    write_json(output/'multi_seed.json',statistics)
    lines=['# CycleTCM 复现执行报告','',
           '只汇总实际完成且覆盖全部 895 测试图的运行；短跑为工程检查，不计入论文复现结果。',
           '论文值是参考列。`B` 使用三分支工作假设；原仓库未公开消融 baseline 的明确实现。',
           '固定 fold1、Adam、physical batch 32、FP32、最多 200 epoch、patience 50；按 validation task-average macro Acc 选 checkpoint。',
           'code_compat 保留独立七图增强、无 ImageNet normalization、global-only 预训练、inverse-positive-ratio 权重、validation 自己的权重及 train_loss scheduler。',
           '本次 Qwen 特征在锁定的 Transformers 4.57.6 下重新提取；旧缓存来自不同数值行为，另行保留。',
           '缓存权重、prompt、输入、processor token 和逐图特征 provenance 在 data/features 的时间戳目录中。',
           '', '| 模型 | seed | best epoch (0-based) | 证候 Acc / F1 (%) | 脏腑 Acc / F1 (%) |',
           '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        lines.append(f'| {row["model"]} | {row["seed"]} | {row["best_epoch"]} | {row["syndrome_acc_percent"]:.2f} / {row["syndrome_f1_percent"]:.2f} | {row["organ_acc_percent"]:.2f} / {row["organ_f1_percent"]:.2f} |')
    lines+=['','六组消融和主模型多 seed 的运行状态见 `run_status.json`；未完成时不宣称整个复现完成。',
            'TVMoE 等外部实现/同 split 逐图预测、论文 Sankey 8×5 公式仍缺失，不能复现对应比较和图。',
            'bootstrap.json 仅包含实际存在的本地模型配对；不把本地比较区间说成论文 TVMoE 比较区间。',
            'Spleen 等类别高度不均衡；逐类 support、阴性数、TP/TN/FP/FN 与 AUC 定义状态保存在各运行 metrics 中。',
            '区域 montage 中 heart_lung 位于图像下方舌尖、kidney 位于上方，liver 取图像右侧；以实际代码和图像坐标为准。',
            '', '数据审计与工程检查见 docs/reproduction_progress.md。','']
    (output/'reproduction_report.md').write_text('\n'.join(lines))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'reports/reproduction')
    args=parser.parse_args()
    output=run_directory(args.output_dir,'report')
    rows=build_report(args.runs,output)
    print(f'REPORT {len(rows)} formal runs: {output}')


if __name__=='__main__':
    main()
