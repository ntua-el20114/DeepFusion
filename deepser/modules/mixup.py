import torch


def mixup_features(features, labels, alpha=0.2):
    """MixUp augmentation"""
    batch_size = features.size(0)
    lam = torch.distributions.beta.Beta(alpha, alpha).sample().to(features.device) if alpha > 0 else torch.ones(1).to(features.device)
    index = torch.randperm(batch_size).to(features.device)
    mixed_features = lam * features + (1 - lam) * features[index, :]
    if len(labels.shape) > 1:
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]
    else:
        mixed_labels = (labels, labels[index], lam)
    return mixed_features, mixed_labels
