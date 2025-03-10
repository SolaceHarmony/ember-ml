"""
Metrics utilities for the ember_ml library.

This module provides metrics utilities for the ember_ml library.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Dict, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)

def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro'),
        'recall': recall_score(y_true, y_pred, average='macro'),
        'f1': f1_score(y_true, y_pred, average='macro')
    }

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }

def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute binary classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary of metrics
    """
    y_pred_binary = (y_pred > threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred_binary == 1))
    tn = np.sum((y_true == 0) & (y_pred_binary == 0))
    fp = np.sum((y_true == 0) & (y_pred_binary == 1))
    fn = np.sum((y_true == 1) & (y_pred_binary == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
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

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Whether to normalize the confusion matrix
        
    Returns:
        Confusion matrix
    """
    n_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    if normalize:
        cm = cm.astype(float)
        row_sums = cm.sum(axis=1)
        cm = cm / row_sums[:, np.newaxis]
        
    return cm

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute ROC curve and AUC.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Tuple of (fpr, tpr, thresholds, auc)
    """
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    
    return fpr, tpr, thresholds, roc_auc

def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Tuple of (precision, recall, thresholds)
    """
    from sklearn.metrics import precision_recall_curve
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    
    return precision, recall, thresholds

def average_precision_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute average precision score.
    
    Args:
        y_true: True binary labels
        y_score: Target scores
        
    Returns:
        Average precision score
    """
    from sklearn.metrics import average_precision_score
    
    return average_precision_score(y_true, y_score)