"""
Visualization utilities for the ember_ml library.

This module provides visualization utilities for the ember_ml library.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Tuple, Optional, Dict, Any
import io
from PIL import Image

def plot_wave(wave: TensorLike, sample_rate: int = 44100, title: str = "Wave Plot") -> plt.Figure:
    """
    Plot a wave signal.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.arange(len(wave)) / sample_rate
    ax.plot(time, wave)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.grid(True)
    return fig

def plot_spectrogram(wave: TensorLike, sample_rate: int = 44100, title: str = "Spectrogram") -> plt.Figure:
    """
    Plot a spectrogram of a wave signal.
    
    Args:
        wave: Wave signal
        sample_rate: Sample rate in Hz
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(wave, Fs=sample_rate, cmap="viridis")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    return fig

def plot_confusion_matrix(cm: TensorLike, class_names: List[str] = None, title: str = "Confusion Matrix") -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # Show all ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig

def plot_roc_curve(fpr: TensorLike, tpr: TensorLike, roc_auc: float, title: str = "ROC Curve") -> plt.Figure:
    """
    Plot a ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under the ROC curve
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    return fig

def plot_precision_recall_curve(precision: TensorLike, recall: TensorLike, title: str = "Precision-Recall Curve") -> plt.Figure:
    """
    Plot a precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, color='darkorange', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    return fig

def plot_learning_curve(train_scores: List[float], val_scores: List[float], title: str = "Learning Curve") -> plt.Figure:
    """
    Plot a learning curve.
    
    Args:
        train_scores: Training scores
        val_scores: Validation scores
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_scores) + 1)
    ax.plot(epochs, train_scores, 'b', label='Training')
    ax.plot(epochs, val_scores, 'r', label='Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig

def fig_to_image(fig: plt.Figure) -> Image.Image:
    """
    Convert a matplotlib figure to a PIL Image.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        PIL Image
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def plot_to_numpy(fig: plt.Figure) -> TensorLike:
    """
    Convert a matplotlib figure to a numpy array.
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Numpy array
    """
    # Draw the figure
    fig.canvas.draw()
    
    # Convert to numpy array
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    return data