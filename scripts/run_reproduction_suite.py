"""Run the fixed primary/ablation/seed queue sequentially and publish completed milestones."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from utils.experiment import run_directory, write_json
from report_reproduction import build_report

QUEUE=[('global',42),('mllm',42),('visual',42),('full',42),
       ('B',42),('BA',42),('BU',42),('BM',42),
       ('B',43),('visual',43),('full',43),('B',44),('visual',44),('full',44)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/reproduction/code_compat.json')
    parser.add_argument('--publish',action='store_true',help='Commit aggregate reports and push fork/shujuecn at each completed run')
    args=parser.parse_args()
    features=args.features.expanduser().resolve()
    assert len(json.loads(features.read_text()))==5109, 'Full feature cache is required'
    if args.publish:
        branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
        assert branch=='shujuecn', 'Milestone publishing requires shujuecn branch'
    output=run_directory(ROOT/'outputs/reproduction','suite')
    report=ROOT/'reports/reproduction'/output.name
    report.mkdir(parents=True)
    state={'status':'running','pid':os.getpid(),'started_at':output.name.split('_suite')[0],
           'output_dir':str(output),'report_dir':str(report),'queue':[
               {'model':model,'seed':seed,'status':'pending'} for model,seed in QUEUE]}
    config=json.loads(args.config.read_text())
    config['mllm_features_file']=str(features)
    write_json(output/'fixed_config.json',config)
    write_json(report/'fixed_config.json',config)
    write_json(output/'suite_status.json',state)
    print(f'SUITE {output}',flush=True)
    run_paths=[]
    for entry in state['queue']:
        if shutil.disk_usage(ROOT).free < 20*2**30:
            raise OSError('Less than 20 GiB free; cannot safely write full training checkpoints')
        entry['status']='running'
        before=set(output.iterdir())
        started=time.monotonic()
        command=[sys.executable,str(ROOT/'src/train/reproduce.py'),'--config',str(output/'fixed_config.json'),
                 '--model',entry['model'],'--seed',str(entry['seed']),'--output-dir',str(output)]
        print(f'START {entry["model"]} seed={entry["seed"]}',flush=True)
        with (output/f'{entry["model"]}_seed{entry["seed"]}.log').open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            entry['pid']=process.pid
            write_json(output/'suite_status.json',state)
            returncode=process.wait()
        created=[path for path in set(output.iterdir())-before if path.is_dir() and (path/'config.json').is_file()]
        assert len(created)==1, f'Expected exactly one run directory: {created}'
        run_paths.extend(created)
        entry.update(status='complete' if returncode==0 else 'failed',returncode=returncode,
                     seconds=time.monotonic()-started,run=str(created[0]))
        write_json(output/'suite_status.json',state)
        build_report(run_paths,report)
        write_json(report/'suite_status.json',state)
        if args.publish:
            subprocess.run(['git','add','--',str(report.relative_to(ROOT))],cwd=ROOT,check=True)
            subprocess.run(['git','commit','-m',f'results: {entry["model"]} seed {entry["seed"]} reproduction milestone'],cwd=ROOT,check=True)
            subprocess.run(['git','push','fork','shujuecn'],cwd=ROOT,check=True)
        print(f'{entry["status"].upper()} {entry["model"]} seed={entry["seed"]} seconds={entry["seconds"]:.1f}',flush=True)
        if returncode:
            state['status']='failed'
            write_json(output/'suite_status.json',state)
            write_json(report/'suite_status.json',state)
            raise RuntimeError(f'Failed run: {created[0]}; queue stopped')
    state['status']='complete'
    write_json(output/'suite_status.json',state)
    write_json(report/'suite_status.json',state)
    if args.publish:
        subprocess.run(['git','add','--',str(report.relative_to(ROOT))],cwd=ROOT,check=True)
        subprocess.run(['git','commit','-m','results: complete primary, six ablation and three-seed suite'],cwd=ROOT,check=True)
        subprocess.run(['git','push','fork','shujuecn'],cwd=ROOT,check=True)
    print(f'COMPLETE SUITE {report}',flush=True)


if __name__=='__main__':
    main()
