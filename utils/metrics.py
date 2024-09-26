import numpy as np
from sklearn.metrics import average_precision_score

def compute_iou_per_class(mask1, mask2, num_classes):
    iou_scores = np.zeros(num_classes)
    for cls in range(num_classes):
        mask1_cls = (mask1 == cls)
        mask2_cls = (mask2 == cls)
        intersection = np.logical_and(mask1_cls, mask2_cls).sum()
        union = np.logical_or(mask1_cls, mask2_cls).sum()
        if union == 0:
            iou_scores[cls] = float('nan')  
        else:
            iou_scores[cls] = intersection / union
    return iou_scores


def calculate_iou(mask1, mask2, num_classes):

    if len(mask1.shape) == 2:
        mask1 = mask1[np.newaxis, :, :]
        mask2 = mask2[np.newaxis, :, :]
    num_layers = mask1.shape[0]
    all_iou_scores = []

    for i in range(num_layers):
        layer_iou_scores = compute_iou_per_class(mask1[i], mask2[i], num_classes)
        all_iou_scores.append(layer_iou_scores)

    iou = np.nanmean(all_iou_scores, axis=0)
    average_iou = np.nanmean(iou[1:])
    return average_iou, iou

def calculate_ap_for_segmentation(predicted_probabilities, true_labels):
    predicted_probabilities_flat = predicted_probabilities.flatten()
    true_labels_flat = true_labels.flatten()

    ap_score = average_precision_score(true_labels_flat, predicted_probabilities_flat)
    return ap_score