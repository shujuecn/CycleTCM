"""Small multi-label Lovasz hinge loss used for the loss ablation."""

import torch


def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    if p == 0:
        return gt_sorted
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp_min(1.0)
    if p > 1:
        jaccard[1:p] -= jaccard[:-1]
    return jaccard


def lovasz_hinge_flat(logits, labels):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, order = torch.sort(errors, descending=True)
    labels_sorted = labels[order]
    return torch.dot(torch.relu(errors_sorted), _lovasz_grad(labels_sorted))


def multilabel_lovasz(logits, labels):
    """Mean positive-class IoU surrogate over labels in a batch."""
    return torch.stack([lovasz_hinge_flat(logits[:, i], labels[:, i])
                        for i in range(logits.shape[1])]).mean()
