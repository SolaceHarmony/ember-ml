"""
Classification metrics for the ember_ml library.

This module provides metrics utilities for classification tasks.
"""

from typing import Dict, Tuple, Any
from ember_ml import tensor, stats
from ember_ml.types import TensorLike
from ember_ml import ops

try:  # Optional dependency
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_curve, auc, precision_recall_curve as sk_precision_recall_curve,
        average_precision_score as sk_average_precision_score,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback path
    accuracy_score = precision_score = recall_score = f1_score = None
    roc_curve = auc = sk_precision_recall_curve = sk_average_precision_score = None


def _basic_macro_metrics(y_true: TensorLike, y_pred: TensorLike) -> Tuple[float, float, float, float]:
    """Compute accuracy/precision/recall/F1 without sklearn."""
    true_list = tensor.to_numpy(tensor.convert_to_tensor(y_true)).tolist()
    pred_list = tensor.to_numpy(tensor.convert_to_tensor(y_pred)).tolist()
    if not true_list:
        return 0.0, 0.0, 0.0, 0.0

    labels = sorted(set(true_list) | set(pred_list))
    total = len(true_list)
    accuracy = sum(int(t == p) for t, p in zip(true_list, pred_list)) / total

    precisions = []
    recalls = []
    for label in labels:
        tp = sum(1 for t, p in zip(true_list, pred_list) if t == label and p == label)
        fp = sum(1 for t, p in zip(true_list, pred_list) if t != label and p == label)
        fn = sum(1 for t, p in zip(true_list, pred_list) if t == label and p != label)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

    precision_macro = sum(precisions) / len(labels)
    recall_macro = sum(recalls) / len(labels)
    if precision_macro + recall_macro > 0:
        f1_macro = 2 * precision_macro * recall_macro / (precision_macro + recall_macro)
    else:
        f1_macro = 0.0

    return accuracy, precision_macro, recall_macro, f1_macro

def classification_metrics(y_true: TensorLike, y_pred: TensorLike) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    if accuracy_score is None:
        acc, prec, rec, f1 = _basic_macro_metrics(y_true, y_pred)
        return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}

    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

def binary_classification_metrics(y_true: TensorLike, y_pred: TensorLike, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    # Convert to tensor and apply threshold
    y_true_tensor = tensor.convert_to_tensor(y_true)
    y_pred_tensor = tensor.convert_to_tensor(y_pred)
    y_pred_binary = tensor.cast(ops.greater(y_pred_tensor, tensor.convert_to_tensor(threshold)), dtype=tensor.int32)
    
    # Calculate confusion matrix elements using tensor operations
    tp = stats.sum(ops.logical_and(ops.equal(y_true_tensor, 1), ops.equal(y_pred_binary, 1)))
    tn = stats.sum(ops.logical_and(ops.equal(y_true_tensor, 0), ops.equal(y_pred_binary, 0)))
    fp = stats.sum(ops.logical_and(ops.equal(y_true_tensor, 0), ops.equal(y_pred_binary, 1)))
    fn = stats.sum(ops.logical_and(ops.equal(y_true_tensor, 1), ops.equal(y_pred_binary, 0)))
    
    # Calculate metrics using tensor operations
    total = ops.add(ops.add(tp, tn), ops.add(fp, fn))
    accuracy = ops.divide(ops.add(tp, tn), total)
    
    # Handle division by zero cases
    precision_denominator = ops.add(tp, fp)
    precision = ops.where(
        ops.greater(precision_denominator, 0),
        ops.divide(tp, precision_denominator),
        tensor.convert_to_tensor(0.0)
    )
    
    recall_denominator = ops.add(tp, fn)
    recall = ops.where(
        ops.greater(recall_denominator, 0),
        ops.divide(tp, recall_denominator),
        tensor.convert_to_tensor(0.0)
    )
    
    precision_recall_sum = ops.add(precision, recall)
    f1 = ops.where(
        ops.greater(precision_recall_sum, 0),
        ops.divide(
            ops.multiply(tensor.convert_to_tensor(2.0), ops.multiply(precision, recall)),
            precision_recall_sum
        ),
        tensor.convert_to_tensor(0.0)
    )
    
    # Return tensor values for further processing
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def confusion_matrix(y_true: TensorLike, y_pred: TensorLike, normalize: bool = False) -> TensorLike:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        Confusion matrix
    """
    # Convert inputs to tensors
    y_true_tensor = tensor.convert_to_tensor(y_true)
    y_pred_tensor = tensor.convert_to_tensor(y_pred)
    
    # Get number of classes
    n_classes = tensor.cast(
        ops.add(
            stats.max(
                stats.max(y_true_tensor), 
                stats.max(y_pred_tensor)
            ), 
            tensor.convert_to_tensor(1)
        ), 
        dtype=tensor.int32
    )
    
    # Initialize confusion matrix
    cm = tensor.zeros((n_classes, n_classes), dtype=tensor.int32)
    
    # Build confusion matrix using tensor operations
    for i in range(tensor.shape(y_true_tensor)[0]):
        true_idx = tensor.cast(y_true_tensor[i], dtype=tensor.int32)
        pred_idx = tensor.cast(y_pred_tensor[i], dtype=tensor.int32)
        
        # Use tensor indexing and update
        cm_i = cm[true_idx]
        cm_i = tensor.tensor_scatter_nd_update(
            cm_i, 
            [[pred_idx]], 
            [ops.add(cm_i[pred_idx], tensor.convert_to_tensor(1))]
        )
        cm = tensor.tensor_scatter_nd_update(
            cm,
            [[true_idx]],
            [cm_i]
        )
    
    # Normalize if requested
    if normalize:
        cm = tensor.cast(cm, dtype=tensor.float32)
        row_sums = stats.sum(cm, axis=1)
        row_sums = tensor.expand_dims(row_sums, axis=1)
        # Avoid division by zero
        row_sums = stats.max(row_sums, tensor.convert_to_tensor(1e-10))
        cm = ops.divide(cm, row_sums)
        
    return cm

def roc_auc(y_true: TensorLike, y_score: TensorLike) -> Tuple[TensorLike, TensorLike, TensorLike, float]:
    """
    Compute ROC curve and AUC.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Tuple of (fpr, tpr, thresholds, auc)
    """
    if roc_curve is None or auc is None:
        raise ImportError(
            "scikit-learn is required for roc_auc. Install scikit-learn to enable this metric."
        )

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, roc_auc

def precision_recall_curve(y_true: TensorLike, y_score: TensorLike) -> Tuple[TensorLike, TensorLike, TensorLike]:
    """
    Compute precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    if sk_precision_recall_curve is None:
        raise ImportError(
            "scikit-learn is required for precision_recall_curve. Install scikit-learn to enable this metric."
        )
    
    precision, recall, thresholds = sk_precision_recall_curve(y_true, y_score)
    
    return precision, recall, thresholds

def average_precision_score(y_true: TensorLike, y_score: TensorLike) -> float:
    """
    Compute average precision score.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Average precision score
    """
    if sk_average_precision_score is None:
        raise ImportError(
            "scikit-learn is required for average_precision_score. Install scikit-learn to enable this metric."
        )
    
    return sk_average_precision_score(y_true, y_score)
