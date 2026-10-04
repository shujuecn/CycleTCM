"""
Train CycleTCM for tongue image classification.
8-class syndrome and 5-class organ prediction with weighted BCE for class imbalance.
"""

import os
import sys
import json
import random
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, matthews_corrcoef
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.model_visual import CycleTCM, count_parameters, print_model_info
from train.runtime import add_path_arguments, configure_logging, training_path_config
from utils.paths import FEATURE_FILE

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0, restore_best_weights=True, start_epoch=0):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.start_epoch = start_epoch
        self.best_acc = None
        self.best_epoch = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
    def __call__(self, val_acc, model, epoch):
        if epoch < self.start_epoch:
            if self.best_acc is None or val_acc > self.best_acc + self.min_delta:
                self.best_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(model)
                self.counter = 0
            return False
        if self.best_acc is None:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.save_checkpoint(model)
        elif val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights:
                    self.load_checkpoint(model)
                return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = copy.deepcopy(model.state_dict())
    
    def load_checkpoint(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class TongueImageDataset(Dataset):

    def __init__(self, feature_file, base_dir=None, transform=None, id_list=None):
        self.feature_file = feature_file
        self.base_dir = base_dir if base_dir else os.path.dirname(feature_file)
        self.transform = transform
        
        logger.info(f"Loading feature file: {feature_file}")
        with open(feature_file, 'r', encoding='utf-8') as f:
            self.all_features = json.load(f)
        
        if id_list is not None:
            id_set = set(id_list)
            self.features = [f for f in self.all_features if f.get('id') in id_set]
            logger.info(f"Filtered by ID list: {len(self.features)}/{len(self.all_features)} samples")
        else:
            self.features = self.all_features
        
        self.valid_samples = []
        for sample in self.features:
            sample_id = sample.get('id')
            if not sample_id:
                continue
                
            img_whole_path = sample.get('img_whole', '')
            img_edge_path = sample.get('img_edge', '')
            img_body_path = sample.get('img_body', '')
            img_heart_lung_path = sample.get('img_heart_lung', '')
            img_spleen_path = sample.get('img_spleen', '')
            img_liver_path = sample.get('img_liver', '')
            img_kidney_path = sample.get('img_kidney', '')
            
            if not all([img_whole_path, img_edge_path, img_body_path, img_heart_lung_path, 
                       img_spleen_path, img_liver_path, img_kidney_path]):
                logger.warning(f"Sample {sample_id} missing image paths, skipped")
                continue
            
            def get_full_path(img_path):
                if os.path.isabs(img_path):
                    return img_path
                else:
                    return os.path.join(self.base_dir, img_path)
            
            img_whole_full = get_full_path(img_whole_path)
            img_edge_full = get_full_path(img_edge_path)
            img_body_full = get_full_path(img_body_path)
            img_heart_lung_full = get_full_path(img_heart_lung_path)
            img_spleen_full = get_full_path(img_spleen_path)
            img_liver_full = get_full_path(img_liver_path)
            img_kidney_full = get_full_path(img_kidney_path)
            
            all_paths = [img_whole_full, img_edge_full, img_body_full, 
                        img_heart_lung_full, img_spleen_full, img_liver_full, img_kidney_full]
            all_exist = all(os.path.exists(p) for p in all_paths)
            
            if all_exist:
                self.valid_samples.append(sample)
            else:
                missing = []
                path_names = ['img_whole', 'img_edge', 'img_body', 'img_heart_lung', 
                             'img_spleen', 'img_liver', 'img_kidney']
                for name, path in zip(path_names, all_paths):
                    if not os.path.exists(path):
                        missing.append(name)
                logger.warning(f"Sample {sample_id} images not found: {missing}, skipped")
        
        logger.info(f"Loaded {len(self.valid_samples)}/{len(self.features)} valid samples from {feature_file}")
    
    def __len__(self):
        return len(self.valid_samples)
    
    def compute_sample_weights(self, resample_strategy='weighted'):
        syndrome_labels = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 
                          'Crack', 'Toothmark', 'FurThick', 'FurYellow']
        
        class_ratios = {}
        for label in syndrome_labels:
            pos_count = sum(sample[label] for sample in self.valid_samples)
            class_ratios[label] = pos_count / len(self.valid_samples) if len(self.valid_samples) > 0 else 0.0
        
        minority_classes = [label for label, ratio in class_ratios.items() if ratio < 0.10]
        if minority_classes:
            logger.info(f"Minority classes (pos ratio < 10%): {len(minority_classes)}: {minority_classes}")
            for label in minority_classes:
                logger.info(f"  {label}: pos ratio = {class_ratios[label]:.4f}")
        
        weights = []
        for sample in self.valid_samples:
            if resample_strategy == 'weighted':
                weight = 1.0

                for label in syndrome_labels:
                    if sample[label] == 1:
                        ratio = class_ratios[label]
                        if ratio < 0.15:
                            weight += 1.8
                        elif ratio < 0.25:
                            weight += 1.2
            
            elif resample_strategy == 'balanced':
                weight = 1.0

                for label in syndrome_labels:
                    if sample[label] == 1:
                        ratio = class_ratios[label]
                        if ratio > 0:
                            weight += 1.0 / (ratio + 0.01)
            
            elif resample_strategy == 'aggressive':
                weight = 1.0

                for label in syndrome_labels:
                    if sample[label] == 1:
                        ratio = class_ratios[label]
                        if ratio < 0.10:
                            weight += 4.0
                        elif ratio < 0.15:
                            weight += 2.5
                        elif ratio < 0.25:
                            weight += 1.5
            
            else:
                weight = 1.0
            
            weights.append(weight)
        
        weights = np.array(weights, dtype=np.float32)
        logger.info(f"Resample strategy: {resample_strategy}")
        logger.info(f"Sample weight range: [{weights.min():.4f}, {weights.max():.4f}], mean: {weights.mean():.4f}")
        
        return weights
    
    def __getitem__(self, idx):
        sample = self.valid_samples[idx]
        sample_id = sample['id']
        
        # get image paths
        img_whole_path = sample.get('img_whole', '')
        img_edge_path = sample.get('img_edge', '')
        img_body_path = sample.get('img_body', '')
        img_heart_lung_path = sample.get('img_heart_lung', '')
        img_spleen_path = sample.get('img_spleen', '')
        img_liver_path = sample.get('img_liver', '')
        img_kidney_path = sample.get('img_kidney', '')
        

        def get_full_path(img_path):
            if os.path.isabs(img_path):
                return img_path
            else:
                return os.path.join(self.base_dir, img_path)
        
        img_whole_full = get_full_path(img_whole_path)
        img_edge_full = get_full_path(img_edge_path)
        img_body_full = get_full_path(img_body_path)
        img_heart_lung_full = get_full_path(img_heart_lung_path)
        img_spleen_full = get_full_path(img_spleen_path)
        img_liver_full = get_full_path(img_liver_path)
        img_kidney_full = get_full_path(img_kidney_path)
        

        def load_image(img_path):
            try:
                image = Image.open(img_path).convert('RGB')

                if self.transform:
                    image = self.transform(image)
                return image
            except Exception as e:
                logger.error(f"Failed to load image {img_path}: {e}")

                placeholder = Image.new('RGB', (256, 256), color=(0, 0, 0))
                if self.transform:
                    placeholder = self.transform(placeholder)
                return placeholder
        
        img_whole = load_image(img_whole_full)
        img_edge = load_image(img_edge_full)
        img_body = load_image(img_body_full)
        img_heart_lung = load_image(img_heart_lung_full)
        img_spleen = load_image(img_spleen_full)
        img_liver = load_image(img_liver_full)
        img_kidney = load_image(img_kidney_full)
        
        # get syndrome labels
        syndrome_labels = torch.tensor([
            sample.get('TonguePale', 0),
            sample.get('TipSideRed', 0),
            sample.get('Spot', 0),
            sample.get('Ecchymosis', 0),
            sample.get('Crack', 0),
            sample.get('Toothmark', 0),
            sample.get('FurThick', 0),
            sample.get('FurYellow', 0)
        ], dtype=torch.float32)
        
        # get organ labels
        organ_labels = torch.tensor([
            sample.get('Heart', 0),
            sample.get('Lung', 0),
            sample.get('Spleen', 0),
            sample.get('Liver', 0),
            sample.get('Kidney', 0)
        ], dtype=torch.float32)
        
        return {
            'id': sample_id,
            'img_whole': img_whole,
            'img_edge': img_edge,
            'img_body': img_body,
            'img_heart_lung': img_heart_lung,
            'img_spleen': img_spleen,
            'img_liver': img_liver,
            'img_kidney': img_kidney,
            'syndrome_labels': syndrome_labels,
            'organ_labels': organ_labels
        }


def calculate_metrics(predictions, labels, threshold=0.5):

    pred_binary = (predictions > threshold).astype(int)
    labels_binary = labels.astype(int)
    
    num_samples, num_classes = labels_binary.shape
    
    # calculate each class accuracy, sensitivity, specificity, precision
    per_class_accuracy = []
    per_class_sensitivity = []
    per_class_specificity = []
    per_class_precision = []
    
    for c in range(num_classes):
        tp = np.sum((pred_binary[:, c] == 1) & (labels_binary[:, c] == 1))
        tn = np.sum((pred_binary[:, c] == 0) & (labels_binary[:, c] == 0))
        fp = np.sum((pred_binary[:, c] == 1) & (labels_binary[:, c] == 0))
        fn = np.sum((pred_binary[:, c] == 0) & (labels_binary[:, c] == 1))
        
        total = tp + tn + fp + fn
        if total > 0:
            acc = (tp + tn) / total
        else:
            acc = 0.0
        per_class_accuracy.append(acc)
        
        if tp + fn > 0:
            sen = tp / (tp + fn)
        else:
            sen = 0.0
        per_class_sensitivity.append(sen)
        
        if tn + fp > 0:
            spe = tn / (tn + fp)
        else:
            spe = 0.0
        per_class_specificity.append(spe)
        
        if tp + fp > 0:
            pre = tp / (tp + fp)
        else:
            pre = 0.0
        per_class_precision.append(pre)
    
    per_class_accuracy = np.array(per_class_accuracy)
    per_class_sensitivity = np.array(per_class_sensitivity)
    per_class_specificity = np.array(per_class_specificity)
    per_class_precision = np.array(per_class_precision)
    
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels_binary, pred_binary, average=None, zero_division=0
    )
    
    # macro average
    precision_macro = np.mean(precision_per_class)
    recall_macro = np.mean(recall_per_class)
    f1_macro = np.mean(f1_per_class)
    accuracy_macro = np.mean(per_class_accuracy)
    sensitivity_macro = np.mean(per_class_sensitivity)
    specificity_macro = np.mean(per_class_specificity)
    precision_calc_macro = np.mean(per_class_precision)
    
    try:
        auc = roc_auc_score(labels_binary, predictions, average='macro')
    except:
        auc = 0.0

    per_class_mcc = []
    for c in range(num_classes):
        try:
            mcc = matthews_corrcoef(labels_binary[:, c], pred_binary[:, c])
            per_class_mcc.append(mcc)
        except:
            per_class_mcc.append(0.0)
    
    per_class_mcc = np.array(per_class_mcc)
    mcc_macro = np.mean(per_class_mcc)
    
    return {
        'per_class_acc_mean': accuracy_macro,
        'per_class_accuracy': per_class_accuracy,
        'per_class_precision': precision_per_class,
        'per_class_recall': recall_per_class,
        'per_class_f1': f1_per_class,
        'per_class_mcc': per_class_mcc,
        'per_class_sensitivity': per_class_sensitivity,
        'per_class_specificity': per_class_specificity,
        'per_class_precision_calc': per_class_precision,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'auc': auc,
        'mcc': mcc_macro,
        'sensitivity_mean': sensitivity_macro,
        'specificity_mean': specificity_macro,
        'precision_calc_mean': precision_calc_macro
    }


def weighted_sigmoid_cross_entropy(predictions, targets, pos_weights, eps=1e-6):

    predictions = torch.clamp(predictions, eps, 1 - eps)
    logits = torch.logit(predictions)
    pos_weights = pos_weights.to(predictions.device, dtype=predictions.dtype)
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weights,
        reduction='none'
    )
    return loss


def log_per_class_metrics(metrics, class_names, dataset_name):

    logger.info(
        f"  {dataset_name} - mean acc: {metrics['per_class_acc_mean']:.4f}, "
        f"mean F1: {metrics['f1']:.4f}, "
        f"AUC: {metrics['auc']:.4f}, "
        f"MCC: {metrics['mcc']:.4f}, "
        f"mean SEN: {metrics['sensitivity_mean']:.4f}, "
        f"mean SPE: {metrics['specificity_mean']:.4f}, "
        f"mean PRE: {metrics['precision_calc_mean']:.4f}"
    )
    
    for name, acc, f1, mcc, sen, spe, pre in zip(
        class_names, 
                                  metrics['per_class_accuracy'], 
                                  metrics['per_class_f1'],
        metrics['per_class_mcc'],
        metrics['per_class_sensitivity'],
        metrics['per_class_specificity'],
        metrics['per_class_precision_calc']
    ):
        logger.info(
            f"    {name:<15} acc: {acc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | "
            f"SEN: {sen:.4f} | SPE: {spe:.4f} | PRE: {pre:.4f}"
        )


def train_one_epoch(model, dataloader, syndrome_pos_weights, organ_pos_weights, optimizer, device, epoch):
    """
    Train one epoch (weighted BCE, syndrome + organ loss).
    """
    model.train()
    total_loss = 0
    num_batches = 0
    syndrome_preds_all = []
    syndrome_labels_all = []
    organ_preds_all = []
    organ_labels_all = []
    
    syndrome_pos_weights = syndrome_pos_weights.to(device)
    organ_pos_weights = organ_pos_weights.to(device)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - train')
    for batch_idx, batch in enumerate(pbar):
        try:
            img_whole = batch['img_whole'].to(device)
            img_edge = batch['img_edge'].to(device)
            img_body = batch['img_body'].to(device)
            img_heart_lung = batch['img_heart_lung'].to(device)
            img_spleen = batch['img_spleen'].to(device)
            img_liver = batch['img_liver'].to(device)
            img_kidney = batch['img_kidney'].to(device)
            syndrome_labels = batch['syndrome_labels'].to(device)
            organ_labels = batch['organ_labels'].to(device)
            
            # forward propagation
            optimizer.zero_grad()
            syndrome_outputs, organ_outputs = model(
                img_whole, img_edge, img_body, img_heart_lung, 
                img_spleen, img_liver, img_kidney
            )
            syndrome_predictions = torch.sigmoid(syndrome_outputs)  # to prob
            organ_predictions = torch.sigmoid(organ_outputs)  # to prob
            
            # use weighted BCE loss
            syndrome_loss = F.binary_cross_entropy_with_logits(
                syndrome_outputs,
                syndrome_labels,
                pos_weight=syndrome_pos_weights,
                reduction='mean'
            )
            organ_loss = F.binary_cross_entropy_with_logits(
                organ_outputs,
                organ_labels,
                pos_weight=organ_pos_weights,
                reduction='mean'
            )
            
            # total loss 
            loss = syndrome_loss + organ_loss
            
            # backward propagation
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # store predictions and labels
            syndrome_preds_all.append(syndrome_predictions.detach().cpu().numpy())
            syndrome_labels_all.append(syndrome_labels.detach().cpu().numpy())
            organ_preds_all.append(organ_predictions.detach().cpu().numpy())
            organ_labels_all.append(organ_labels.detach().cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
            
        except Exception as e:
            logger.error(f"Batch {batch_idx} error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # calculate metrics
    if len(syndrome_preds_all) > 0:
        syndrome_preds_all = np.vstack(syndrome_preds_all)
        syndrome_labels_all = np.vstack(syndrome_labels_all)
        organ_preds_all = np.vstack(organ_preds_all)
        organ_labels_all = np.vstack(organ_labels_all)
        syndrome_metrics = calculate_metrics(syndrome_preds_all, syndrome_labels_all)
        organ_metrics = calculate_metrics(organ_preds_all, organ_labels_all)
    else:
        syndrome_metrics = {}
        organ_metrics = {}
        logger.warning(f"Epoch {epoch} train: no successful batches")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    return avg_loss, syndrome_metrics, organ_metrics


def validate(model, dataloader, syndrome_pos_weights, organ_pos_weights, device, epoch):

    model.eval()
    total_loss = 0
    num_batches = 0
    syndrome_preds_all = []
    syndrome_labels_all = []
    organ_preds_all = []
    organ_labels_all = []
    
    syndrome_pos_weights = syndrome_pos_weights.to(device)
    organ_pos_weights = organ_pos_weights.to(device)
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} - val')
        for batch_idx, batch in enumerate(pbar):
            try:
                img_whole = batch['img_whole'].to(device)
                img_edge = batch['img_edge'].to(device)
                img_body = batch['img_body'].to(device)
                img_heart_lung = batch['img_heart_lung'].to(device)
                img_spleen = batch['img_spleen'].to(device)
                img_liver = batch['img_liver'].to(device)
                img_kidney = batch['img_kidney'].to(device)
                syndrome_labels = batch['syndrome_labels'].to(device)
                organ_labels = batch['organ_labels'].to(device)
                
                # forward propagation
                syndrome_outputs, organ_outputs = model(img_whole, img_edge, img_body, img_heart_lung, 
                                                        img_spleen, img_liver, img_kidney)
                syndrome_predictions = torch.sigmoid(syndrome_outputs)
                organ_predictions = torch.sigmoid(organ_outputs)
                
                # use weighted BCE loss
                syndrome_loss = F.binary_cross_entropy_with_logits(
                    syndrome_outputs,
                    syndrome_labels,
                    pos_weight=syndrome_pos_weights,
                    reduction='mean'
                )
                organ_loss = F.binary_cross_entropy_with_logits(
                    organ_outputs,
                    organ_labels,
                    pos_weight=organ_pos_weights,
                    reduction='mean'
                )
                loss = syndrome_loss + organ_loss
                
                total_loss += loss.item()
                num_batches += 1
                
                # store predictions and labels
                syndrome_preds_all.append(syndrome_predictions.cpu().numpy())
                syndrome_labels_all.append(syndrome_labels.cpu().numpy())
                organ_preds_all.append(organ_predictions.cpu().numpy())
                organ_labels_all.append(organ_labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                logger.error(f"Val batch {batch_idx} error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # calculate metrics
    if len(syndrome_preds_all) > 0:
        syndrome_preds_all = np.vstack(syndrome_preds_all)
        syndrome_labels_all = np.vstack(syndrome_labels_all)
        organ_preds_all = np.vstack(organ_preds_all)
        organ_labels_all = np.vstack(organ_labels_all)
        syndrome_metrics = calculate_metrics(syndrome_preds_all, syndrome_labels_all)
        organ_metrics = calculate_metrics(organ_preds_all, organ_labels_all)
    else:
        syndrome_metrics = {}
        organ_metrics = {}
        logger.warning(f"Epoch {epoch} val: no successful batches")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    return avg_loss, syndrome_metrics, organ_metrics


def compute_class_weights(feature_file, id_list=None, base_dir=None, eps=1e-3):

    # load feature data
    with open(feature_file, 'r', encoding='utf-8') as f:
        all_features = json.load(f)
    
    if id_list is not None:
        id_set = set(id_list)
        labels = [f for f in all_features if f.get('id') in id_set]
    else:
        labels = all_features

    # calculate syndrome weights
    syndrome_labels = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    syndrome_pos_weights = []
    for label in syndrome_labels:
        pos_count = sum(sample.get(label, 0) for sample in labels)
        pos_ratio = pos_count / len(labels) if len(labels) > 0 else eps
        ratio = max(pos_ratio, eps)
        pos_weight = 1.0 / ratio
        syndrome_pos_weights.append(pos_weight)
    
    syndrome_pos_weights = np.array(syndrome_pos_weights)
    logger.info(f"Syndrome pos weights: {syndrome_pos_weights} (TonguePale, TipSideRed, Spot, Ecchymosis, Crack, Toothmark, FurThick, FurYellow)")
    
    # calculate organ weights
    organ_labels = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
    organ_pos_weights = []
    for label in organ_labels:
        pos_count = sum(sample.get(label, 0) for sample in labels)
        pos_ratio = pos_count / len(labels) if len(labels) > 0 else eps
        ratio = max(pos_ratio, eps)
        pos_weight = 1.0 / ratio
        organ_pos_weights.append(pos_weight)
    
    organ_pos_weights = np.array(organ_pos_weights)
    logger.info(f"Organ pos weights: {organ_pos_weights} (Heart, Lung, Spleen, Liver, Kidney)")
    
    return (torch.tensor(syndrome_pos_weights, dtype=torch.float32),
            torch.tensor(organ_pos_weights, dtype=torch.float32))


def predict_test(model, config, device):

    logger.info(f"\n{'='*80}")
    logger.info(f"Predicting test.json")
    logger.info(f"{'='*80}")
    
    feature_file = config.get('feature_file', str(FEATURE_FILE))
    test_label_file = os.path.join(config['label_dir'], 'test.json')
    base_dir = config.get('base_dir', os.path.dirname(feature_file))
    
    logger.info("Loading test ID list...")
    with open(test_label_file, 'r', encoding='utf-8') as f:
        test_labels = json.load(f)
        test_ids = [item['id'] for item in test_labels]
    
    logger.info(f"Test IDs: {len(test_ids)}")
    
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    test_dataset = TongueImageDataset(feature_file, base_dir=base_dir, transform=test_transform, id_list=test_ids)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    model.eval()
    total_loss = 0
    num_batches = 0
    syndrome_preds_all = []
    syndrome_labels_all = []
    organ_preds_all = []
    organ_labels_all = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Predict test.json')
        for batch_idx, batch in enumerate(pbar):
            try:
                img_whole = batch['img_whole'].to(device)
                img_edge = batch['img_edge'].to(device)
                img_body = batch['img_body'].to(device)
                img_heart_lung = batch['img_heart_lung'].to(device)
                img_spleen = batch['img_spleen'].to(device)
                img_liver = batch['img_liver'].to(device)
                img_kidney = batch['img_kidney'].to(device)
                syndrome_labels = batch['syndrome_labels'].to(device)
                organ_labels = batch['organ_labels'].to(device)

                syndrome_outputs, organ_outputs = model(img_whole, img_edge, img_body, img_heart_lung,
                                                        img_spleen, img_liver, img_kidney)
                syndrome_predictions = torch.sigmoid(syndrome_outputs)
                organ_predictions = torch.sigmoid(organ_outputs)
                
                # unweighted BCE for test loss
                syndrome_loss = F.binary_cross_entropy_with_logits(
                    syndrome_outputs, syndrome_labels, reduction='mean'
                )
                organ_loss = F.binary_cross_entropy_with_logits(
                    organ_outputs, organ_labels, reduction='mean'
                )
                loss = syndrome_loss + organ_loss

                total_loss += loss.item()
                num_batches += 1

                syndrome_preds_all.append(syndrome_predictions.cpu().numpy())
                syndrome_labels_all.append(syndrome_labels.cpu().numpy())
                organ_preds_all.append(organ_predictions.cpu().numpy())
                organ_labels_all.append(organ_labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
                
            except Exception as e:
                logger.error(f"Predict batch {batch_idx} error: {e}")
                continue
    
    if len(syndrome_preds_all) > 0:
        syndrome_preds_all = np.vstack(syndrome_preds_all)
        syndrome_labels_all = np.vstack(syndrome_labels_all)
        organ_preds_all = np.vstack(organ_preds_all)
        organ_labels_all = np.vstack(organ_labels_all)
        syndrome_metrics = calculate_metrics(syndrome_preds_all, syndrome_labels_all)
        organ_metrics = calculate_metrics(organ_preds_all, organ_labels_all)
    else:
        syndrome_metrics = {}
        organ_metrics = {}
        logger.warning("Test: no successful batches")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    syndrome_names = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    organ_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
    
    logger.info(f"\nTest (test.json) results:")
    logger.info(f"Test loss: {avg_loss:.4f}")
    logger.info(f"\nSyndrome metrics:")
    log_per_class_metrics(syndrome_metrics, syndrome_names, "Test")
    logger.info(f"\nOrgan metrics:")
    log_per_class_metrics(organ_metrics, organ_names, "Test")
    logger.info("-----------------")
    
    return avg_loss, syndrome_metrics, organ_metrics



def train_model(config):
    """
    Train the model.
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting CycleTCM training (weighted BCE, 7 images, syndrome + organ)")
    logger.info(f"{'='*80}")
    
    feature_file = config.get('feature_file', str(FEATURE_FILE))
    train_label_file = os.path.join(config['label_dir'], f'train_dataset.json')
    val_label_file = os.path.join(config['label_dir'], f'val_dataset.json')
    base_dir = config.get('base_dir', os.path.dirname(feature_file))
    
    logger.info("Loading train/val ID lists...")
    with open(train_label_file, 'r', encoding='utf-8') as f:
        train_labels = json.load(f)
        train_ids = [item['id'] for item in train_labels]
    
    with open(val_label_file, 'r', encoding='utf-8') as f:
        val_labels = json.load(f)
        val_ids = [item['id'] for item in val_labels]
    
    logger.info(f"Train IDs: {len(train_ids)}")
    logger.info(f"Val IDs: {len(val_ids)}")
    
    # compute class (pos) weights before training
    logger.info("Computing train class (pos) weights...")
    train_syndrome_pos_weights, train_organ_pos_weights = compute_class_weights(feature_file, id_list=train_ids, base_dir=base_dir)
    
    logger.info("Computing val class (pos) weights...")
    val_syndrome_pos_weights, val_organ_pos_weights = compute_class_weights(feature_file, id_list=val_ids, base_dir=base_dir)
    
    # image transformation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.1))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = TongueImageDataset(feature_file, base_dir=base_dir, transform=train_transform, id_list=train_ids)
    val_dataset = TongueImageDataset(feature_file, base_dir=base_dir, transform=val_transform, id_list=val_ids)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    device = torch.device(config['device'])
    model = CycleTCM(num_classes1=8, num_classes2=5, pretrained=config.get('pretrained', False)).to(device)
    
    print_model_info(model)
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 1e-4)
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=5
    )
    
    syndrome_names = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
    organ_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
    
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 50),
        min_delta=config.get('early_stopping_min_delta', 0.0),
        restore_best_weights=True,
        start_epoch=0
    )
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 0
    current_lr = optimizer.param_groups[0]['lr']
    
    
    for epoch in range(1, config['num_epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['num_epochs']}")
        
        # train
        train_loss, train_syndrome_metrics, train_organ_metrics = train_one_epoch(
            model, train_loader, train_syndrome_pos_weights, train_organ_pos_weights, optimizer, device, epoch
        )
        
        # validate
        val_loss, val_syndrome_metrics, val_organ_metrics = validate(
            model, val_loader, val_syndrome_pos_weights, val_organ_pos_weights, device, epoch
        )
        
        # update learning rate
        old_lr = current_lr
        scheduler.step(train_loss)
        current_lr = optimizer.param_groups[0]['lr']

        if current_lr != old_lr:
            logger.info(f"LR {old_lr:.2e} -> {current_lr:.2e}")
        
        # record metrics
        logger.info(f"\nEpoch {epoch} results:")
        logger.info(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")
        logger.info(f"\nSyndrome metrics:")
        log_per_class_metrics(train_syndrome_metrics, syndrome_names, "Train")
        log_per_class_metrics(val_syndrome_metrics, syndrome_names, "Val")
        logger.info(f"\nOrgan metrics:")
        log_per_class_metrics(train_organ_metrics, organ_names, "Train")
        log_per_class_metrics(val_organ_metrics, organ_names, "Val")
        logger.info("-----------------")
        
        train_acc = (train_syndrome_metrics['per_class_acc_mean'] + train_organ_metrics['per_class_acc_mean']) / 2
        val_acc = (val_syndrome_metrics['per_class_acc_mean'] + val_organ_metrics['per_class_acc_mean']) / 2
        
        # record best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"Better model (loss) at epoch {epoch}, val loss: {val_loss:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            logger.info(f"Better model (acc) at epoch {epoch}, val acc: {val_acc:.4f}")
        
        # check early stopping
        if early_stopping(val_acc, model, epoch):
            logger.info(f"\nEarly stop at epoch {epoch}, patience={early_stopping.patience}")
            logger.info(f"Best val acc: {early_stopping.best_acc:.4f} at epoch {early_stopping.best_epoch}")
            logger.info(f"No improvement for {epoch - early_stopping.best_epoch} epochs ({early_stopping.best_epoch} -> {epoch})")
            break
    
    logger.info(f"\nTraining done. Best epoch: {best_epoch}, val loss: {best_val_loss:.4f}, val acc: {best_val_acc:.4f}")
    
    return best_val_loss, best_epoch, model, early_stopping


def main():

    parser = argparse.ArgumentParser(description='Train CycleTCM model')
    add_path_arguments(parser, 'visual')
    args = parser.parse_args()
    paths = training_path_config(args)
    configure_logging(paths['output_dir'])
    
    seed = 42
    set_seed(seed)
    logger.info(f"Random seed: {seed}")

    config = {
        **paths,
        'batch_size': 32,
        'num_workers': 4,
        'num_epochs': 200,
        'learning_rate': 2e-4,
        'weight_decay': 1e-4,
        'device': 'cuda:0',
        'pretrained': True,
        'save_checkpoint': True,
        'early_stopping_patience': 50,
        'early_stopping_min_delta': 0.001
    }
    
    logger.info("="*80)
    logger.info("Starting CycleTCM (tongue image classification, weighted BCE, 7 images, syndrome + organ)")
    logger.info(f"Feature file: {config['feature_file']}")
    logger.info(f"Train: train_dataset.json")
    logger.info(f"Val: val_dataset.json")
    logger.info("="*80)
    logger.info(f"Config:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    
    # Train model
    best_val_loss, best_epoch, model, early_stopping = train_model(config)
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("Training summary")
    logger.info("="*80)
    logger.info(f"Best val loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    logger.info("="*80)
    
    # Load best model 
    device = torch.device(config['device'])
    
    if early_stopping.best_weights is not None:
        model.load_state_dict(early_stopping.best_weights)
    
    # Predict test.json
    test_loss, test_syndrome_metrics, test_organ_metrics = predict_test(model, config, device)
    
    if config.get('output_log'):
        syndrome_names = ['TonguePale', 'TipSideRed', 'Spot', 'Ecchymosis', 'Crack', 'Toothmark', 'FurThick', 'FurYellow']
        organ_names = ['Heart', 'Lung', 'Spleen', 'Liver', 'Kidney']
        
        syndrome_mean_acc = np.mean(test_syndrome_metrics['per_class_accuracy'])
        syndrome_mean_f1 = np.mean(test_syndrome_metrics['per_class_f1'])
        syndrome_mean_mcc = np.mean(test_syndrome_metrics['per_class_mcc'])
        syndrome_mean_sen = test_syndrome_metrics['sensitivity_mean']
        syndrome_mean_spe = test_syndrome_metrics['specificity_mean']
        syndrome_mean_pre = test_syndrome_metrics['precision_calc_mean']
        syndrome_auc = test_syndrome_metrics['auc']
        organ_mean_acc = np.mean(test_organ_metrics['per_class_accuracy'])
        organ_mean_f1 = np.mean(test_organ_metrics['per_class_f1'])
        organ_mean_mcc = np.mean(test_organ_metrics['per_class_mcc'])
        organ_mean_sen = test_organ_metrics['sensitivity_mean']
        organ_mean_spe = test_organ_metrics['specificity_mean']
        organ_mean_pre = test_organ_metrics['precision_calc_mean']
        organ_auc = test_organ_metrics['auc']
        
        with open(config['output_log'], 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Syndrome metrics:\n")
            for name, acc, f1, mcc, sen, spe, pre in zip(
                syndrome_names, 
                                         test_syndrome_metrics['per_class_accuracy'], 
                                         test_syndrome_metrics['per_class_f1'],
                test_syndrome_metrics['per_class_mcc'],
                test_syndrome_metrics['per_class_sensitivity'],
                test_syndrome_metrics['per_class_specificity'],
                test_syndrome_metrics['per_class_precision_calc']
            ):
                f.write(f"{name:<15} acc: {acc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | "
                       f"SEN: {sen:.4f} | SPE: {spe:.4f} | PRE: {pre:.4f}\n")
            f.write(f"Syndrome mean acc: {syndrome_mean_acc:.4f} | mean F1: {syndrome_mean_f1:.4f} | AUC: {syndrome_auc:.4f} | "
                   f"MCC: {syndrome_mean_mcc:.4f} | mean SEN: {syndrome_mean_sen:.4f} | mean SPE: {syndrome_mean_spe:.4f} | "
                   f"mean PRE: {syndrome_mean_pre:.4f}\n")
            f.write(f"\nOrgan metrics:\n")
            for name, acc, f1, mcc, sen, spe, pre in zip(
                organ_names, 
                                         test_organ_metrics['per_class_accuracy'], 
                                         test_organ_metrics['per_class_f1'],
                test_organ_metrics['per_class_mcc'],
                test_organ_metrics['per_class_sensitivity'],
                test_organ_metrics['per_class_specificity'],
                test_organ_metrics['per_class_precision_calc']
            ):
                f.write(f"{name:<15} acc: {acc:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f} | "
                       f"SEN: {sen:.4f} | SPE: {spe:.4f} | PRE: {pre:.4f}\n")
            f.write(f"Organ mean acc: {organ_mean_acc:.4f} | mean F1: {organ_mean_f1:.4f} | AUC: {organ_auc:.4f} | "
                   f"MCC: {organ_mean_mcc:.4f} | mean SEN: {organ_mean_sen:.4f} | mean SPE: {organ_mean_spe:.4f} | "
                   f"mean PRE: {organ_mean_pre:.4f}\n")
            f.write(f"{'='*80}\n")
        logger.info(f"Per-class acc/F1/MCC/SEN/SPE/PRE written to: {config['output_log']}")
    
    logger.info("\n" + "="*80)
    logger.info("Final summary")
    logger.info("="*80)
    logger.info(f"Best val loss: {best_val_loss:.4f} at Epoch {best_epoch}")
    logger.info(f"Test loss: {test_loss:.4f}")
    logger.info("="*80)


if __name__ == '__main__':
    main()
