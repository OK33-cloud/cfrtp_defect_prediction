"""SegFormer-B0。モデル8が使う。実装は PyTorch。"""

import torch
from transformers import SegformerForSemanticSegmentation

IMAGE_SIZE = (128, 128)


def create_segformer(dropout_rate):
    return SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/mit-b0",
        num_labels=1,
        id2label={0: "void"},
        label2id={"void": 0},
        classifier_dropout_prob=float(dropout_rate),
        ignore_mismatched_sizes=True,
    )


def masked_tversky_loss(pred, y_and_mask, alpha, beta):
    y_true = y_and_mask[..., 0]
    mask = y_and_mask[..., 1]
    y_pred = pred[:, 0]
    true_positive = torch.sum(y_true * y_pred * mask)
    false_positive = torch.sum((1.0 - y_true) * y_pred * mask)
    false_negative = torch.sum(y_true * (1.0 - y_pred) * mask)
    return (alpha * false_positive + beta * false_negative) / (
        true_positive + alpha * false_positive + beta * false_negative + 1e-6
    )


def segformer_probabilities(model, images):
    logits = model(pixel_values=images).logits
    logits = torch.nn.functional.interpolate(
        logits, size=IMAGE_SIZE, mode="bilinear", align_corners=False
    )
    return torch.sigmoid(logits)
