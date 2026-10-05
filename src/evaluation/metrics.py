"""
Evaluation Metrics Module
Comprehensive metrics, confusion matrix, and error analysis for resume classification.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import os
import logging

logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, labels=None) -> dict:
    """
    Compute comprehensive classification metrics.
    
    Returns dict with accuracy, macro/weighted precision, recall, F1,
    and per-class metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Per-class metrics
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    metrics['classification_report'] = report
    metrics['classification_report_text'] = classification_report(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    return metrics


def plot_confusion_matrix(
    y_true, y_pred, labels, save_path=None, figsize=(16, 14), title='Confusion Matrix'
):
    """
    Generate and optionally save a high-quality confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    
    plt.close(fig)
    return cm


def analyze_confusion(cm, labels) -> list:
    """
    Identify most confused class pairs from a confusion matrix.
    Returns list of (actual, predicted, count) sorted by count descending.
    """
    confusions = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i != j and cm[i, j] > 0:
                confusions.append({
                    'actual': actual,
                    'predicted': predicted,
                    'count': cm[i, j],
                })
    confusions.sort(key=lambda x: x['count'], reverse=True)
    return confusions


def perform_error_analysis(
    df: pd.DataFrame,
    y_true,
    y_pred,
    y_proba=None,
    text_col='Resume_str',
    id_col='ID',
    max_errors=50,
) -> pd.DataFrame:
    """
    Collect and analyze misclassified samples.
    
    Args:
        df: DataFrame with resume text and IDs
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        text_col: Column name for resume text
        id_col: Column name for resume ID
        max_errors: Maximum number of errors to collect
        
    Returns:
        DataFrame of errors with analysis columns
    """
    errors = []
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    
    error_mask = y_true_arr != y_pred_arr
    error_indices = np.where(error_mask)[0]
    
    for idx in error_indices[:max_errors]:
        row = df.iloc[idx]
        error_info = {
            'resume_id': row[id_col] if id_col in df.columns else idx,
            'actual': y_true_arr[idx],
            'predicted': y_pred_arr[idx],
            'text_preview': str(row[text_col])[:200] if text_col in df.columns else '',
            'text_length': len(str(row[text_col]).split()) if text_col in df.columns else 0,
        }
        
        if y_proba is not None:
            error_info['confidence'] = float(np.max(y_proba[idx]))
        
        errors.append(error_info)
    
    error_df = pd.DataFrame(errors)
    
    if len(error_df) > 0:
        # Categorize possible causes
        error_df['possible_cause'] = error_df.apply(_categorize_error, axis=1)
    
    return error_df


def _categorize_error(row) -> str:
    """Heuristic categorization of error causes."""
    if row.get('text_length', 0) < 50:
        return 'Very short resume'
    if row.get('confidence', 1.0) > 0.8:
        return 'Possible mislabeled sample'
    if row.get('confidence', 1.0) < 0.3:
        return 'Low confidence / generic resume'
    return 'Class overlap or insufficient features'


def print_model_comparison(results: list):
    """Pretty-print a model comparison table."""
    if not results:
        return
    
    df = pd.DataFrame(results)
    cols = ['model', 'features', 'accuracy', 'macro_f1', 'weighted_f1', 'training_time']
    available = [c for c in cols if c in df.columns]
    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)
    print(df[available].to_string(index=False))
    print("=" * 80)
