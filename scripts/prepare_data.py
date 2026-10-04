"""Prepare seven tongue views and split manifests outside the source tree."""

import os, sys, csv, json, argparse
from pathlib import Path
import cv2
from multiprocessing import Pool

SRC = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, os.path.join(SRC, 'src'))
from utils.paths import PROCESSED_DATA_DIR, RAW_DATA_DIR
from utils.experiment import run_directory
from data_preprocessed import data_segment_regions as reg
from data_preprocessed import data_segment_organs as org

DATA = str(RAW_DATA_DIR)
ROOT = str(PROCESSED_DATA_DIR)
PP = os.path.join(ROOT, 'pp')

LABELS = ['TonguePale','TipSideRed','Spot','Ecchymosis','Crack','Toothmark','FurThick','FurYellow','Heart','Lung','Spleen','Liver','Kidney']

def configure_paths(raw_data_dir, output_dir):
    global DATA, ROOT, PP
    DATA = os.path.abspath(os.path.expanduser(raw_data_dir))
    ROOT = os.path.abspath(os.path.expanduser(output_dir))
    PP = os.path.join(ROOT, 'pp')

def ensure_dirs():
    for d in ['images','images_body','images_edge','images_heart_lung','images_kidney','images_liver','images_spleen','pp']:
        os.makedirs(os.path.join(ROOT, d), exist_ok=True)

def load_rows():
    rows = []
    for f in ['train_fold1.csv','val_fold1.csv','test.csv']:
        split = f.split('.')[0].replace('_fold1','')
        with open(os.path.join(DATA, 'list', f), newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                row['_split'] = split
                rows.append(row)
    return rows

def worker(row):
    ip = row['image_path']  # e.g. 0001_1000/2_1.jpg or test/1_1.jpg
    seg_jpg = os.path.join(DATA, 'seg', ip)
    img = cv2.imread(seg_jpg)
    if img is None:
        return None
    name = os.path.splitext(os.path.basename(ip))[0] + '.png'
    pp_path = os.path.join(PP, name)
    cv2.imwrite(pp_path, img)
    # whole image (resized later)
    cv2.imwrite(os.path.join(ROOT, 'images', name), img)
    try:
        reg.process_image(pp_path, os.path.join(ROOT,'images_body'), os.path.join(ROOT,'images_edge'), 0.15)
        org.process_single_image(pp_path,
            {'top_edge': os.path.join(ROOT,'images_heart_lung'),
             'bottom_edge': os.path.join(ROOT,'images_kidney'),
             'right_edge': os.path.join(ROOT,'images_liver'),
             'center_rect': os.path.join(ROOT,'images_spleen')},
            r=0.196, r2=0.632, r_liver=0.10)
    except Exception as e:
        print('seg err', name, e)
    return name

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-data-dir', default=str(RAW_DATA_DIR),
                        help='TongueDx2 release directory containing list/ and seg/')
    parser.add_argument('--output-dir', default=str(PROCESSED_DATA_DIR.parent),
                        help='Parent for a new timestamp-prefixed processed dataset')
    parser.add_argument('--workers', type=int, default=8, help='Number of preprocessing workers')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be at least 1')
    configure_paths(args.raw_data_dir, run_directory(args.output_dir, 'CycleTCM'))
    ensure_dirs()
    rows = load_rows()
    print('rows', len(rows))
    with Pool(args.workers, initializer=configure_paths, initargs=(DATA, ROOT)) as p:
        names = p.map(worker, rows)
    ok = [n for n in names if n]
    print('processed', len(ok))
    # resize everything to 224x224
    for d in ['images','images_body','images_edge','images_heart_lung','images_kidney','images_liver','images_spleen']:
        for f in os.listdir(os.path.join(ROOT, d)):
            p = os.path.join(ROOT, d, f)
            img = cv2.imread(p)
            if img is None: continue
            img = cv2.resize(img, (224, 224))
            cv2.imwrite(p, img)
    # region outputs use _body/_edge naming; organ outputs use _heart_lung etc.
    feats = []
    for row in rows:
        name = os.path.splitext(os.path.basename(row['image_path']))[0]
        rec = {'id': int(row['id']), 'image_file': name + '.png'}
        rec['img_whole'] = f'images/{name}.png'
        rec['img_body'] = f'images_body/{name}_body.png'
        rec['img_edge'] = f'images_edge/{name}_edge.png'
        rec['img_heart_lung'] = f'images_heart_lung/{name}_heart_lung.png'
        rec['img_kidney'] = f'images_kidney/{name}_kidney.png'
        rec['img_liver'] = f'images_liver/{name}_liver.png'
        rec['img_spleen'] = f'images_spleen/{name}_spleen.png'
        for k in LABELS:
            rec[k] = int(row[k])
        feats.append(rec)
    feat_path = os.path.join(ROOT, 'feature_all_encoded.json')
    with open(feat_path, 'w', encoding='utf-8') as f:
        json.dump(feats, f)
    print('wrote', feat_path, len(feats))
    label_dir = os.path.join(ROOT, 'labels', 'json')
    os.makedirs(label_dir, exist_ok=True)
    for split_file, out in [('train_fold1.csv','train_dataset.json'), ('val_fold1.csv','val_dataset.json'), ('test.csv','test.json')]:
        rows_s = [r for r in rows if r['_split'] == split_file.split('.')[0].replace('_fold1','')]
        out_rows = []
        for r in rows_s:
            rec = {'id': int(r['id'])}
            for k in LABELS: rec[k] = int(r[k])
            out_rows.append(rec)
        p = os.path.join(label_dir, out)
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(out_rows, f)
        print('wrote', p, len(out_rows))


if __name__ == '__main__':
    main()
