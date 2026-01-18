"""
Model evaluation with comprehensive metrics and visualizations.

Generates confusion matrices, ROC curves, cost-benefit analysis, and more.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, classification_report
)
import shap

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_cost_savings(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fn_cost: float = 10000.0,
    fp_cost: float = 100.0
) -> Dict[str, float]:
    """
    Calculate cost savings from fraud detection.
    
    Args:
        y_true: Ground truth labels (0=legitimate, 1=fraud)
        y_pred: Predicted labels
        fn_cost: Cost of false negative (missed fraud) in INR
        fp_cost: Cost of false positive (false alarm) in INR
        
    Returns:
        Dictionary with cost analysis
        
    Raises:
        ValueError: If arrays have different lengths
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate costs
    total_fn_cost = fn * fn_cost
    total_fp_cost = fp * fp_cost
    total_cost = total_fn_cost + total_fp_cost
    
    # Baseline: catch nothing (all false negatives)
    baseline_cost = (fn + tp) * fn_cost
    savings = baseline_cost - total_cost
    
    # Alternative baseline: catch everything (all false positives on legit transactions)
    # catch_all_cost = (tn + fp) * fp_cost
    
    return {
        "total_cost": float(total_cost),
        "fn_cost": float(total_fn_cost),
        "fp_cost": float(total_fp_cost),
        "baseline_cost": float(baseline_cost),
        "savings": float(savings),
        "savings_percentage": float((savings / baseline_cost) * 100) if baseline_cost > 0 else 0,
        "detected_fraud_amount": float(tp * fn_cost),  # Fraud we caught
        "missed_fraud_amount": float(fn * fn_cost)  # Fraud we missed
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str = './outputs/evaluation/confusion_matrix.png'
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str = './outputs/evaluation/roc_curve.png'
):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved ROC curve to {output_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: str = './outputs/evaluation/precision_recall_curve.png'
):
    """Plot and save Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve', fontweight='bold')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    logger.info(f"Saved Precision-Recall curve to {output_path}")


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 10,
    output_path: str = './outputs/evaluation/feature_importance.png'
):
    """Plot and save feature importance (for tree-based models)."""
    try:
        import joblib
        
        # Check if it's an XGBoost model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            logger.warning("Model doesn't have feature_importances_ attribute")
            return
        
        # Create DataFrame
        feature_imp = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_imp, x='importance', y='feature', palette='viridis')
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Saved feature importance to {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not plot feature importance: {str(e)}")


def plot_shap_summary(
    model,
    X: pd.DataFrame,
    output_path: str = './outputs/evaluation/shap_summary.png'
):
    """Generate and save SHAP summary plot."""
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Summary plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, show=False, max_display=10)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP summary to {output_path}")
        
    except Exception as e:
        logger.warning(f"Could not generate SHAP summary: {str(e)}")


def generate_evaluation_report(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    output_dir: str = './outputs/evaluation'
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report with all metrics and plots.
    
    Args:
        model_name: Name of the model being evaluated
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary with all evaluation metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Evaluating {model_name}")
    logger.info(f"{'='*60}\n")
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    pr_auc = average_precision_score(y_true, y_pred_proba)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Cost analysis
    cost_analysis = calculate_cost_savings(y_true, y_pred)
    
    # Generate plots
    plot_confusion_matrix(y_true, y_pred, f"{output_dir}/{model_name}_confusion_matrix.png")
    plot_roc_curve(y_true, y_pred_proba, f"{output_dir}/{model_name}_roc_curve.png")
    plot_precision_recall_curve(y_true, y_pred_proba, f"{output_dir}/{model_name}_pr_curve.png")
    
    # Compile report
    report = {
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "f2_score": float(f2),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc)
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "cost_analysis": cost_analysis
    }
    
    # Save report
    report_path = f"{output_dir}/{model_name}_evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info("\nMetrics Summary:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  F2 Score: {f2:.4f}")
    logger.info(f"  ROC-AUC: {roc_auc:.4f}")
    logger.info(f"  PR-AUC: {pr_auc:.4f}")
    
    logger.info("\nCost Analysis:")
    logger.info(f"  Total Cost: ₹{cost_analysis['total_cost']:,.0f}")
    logger.info(f"  Savings: ₹{cost_analysis['savings']:,.0f} ({cost_analysis['savings_percentage']:.1f}%)")
    logger.info(f"  Detected Fraud: ₹{cost_analysis['detected_fraud_amount']:,.0f}")
    logger.info(f"  Missed Fraud: ₹{cost_analysis['missed_fraud_amount']:,.0f}")
    
    logger.info(f"\nReport saved to {report_path}")
    
    return report


def main():
    """Evaluate ensemble model on test set."""
    from models.ensemble import FraudEnsemble
    
    # Load test data
    test_df = pd.read_csv('./data/processed/test_latest.csv')
    X_test = test_df.drop('Class', axis=1)
    y_test = test_df['Class']
    
    # Load ensemble
    ensemble = FraudEnsemble()
    
    # Generate predictions
    y_pred_proba = ensemble.predict_proba(X_test)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Generate evaluation report
    report = generate_evaluation_report(
        model_name='fraud_ensemble_v1',
        y_true=y_test.values,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba
    )
    
    # Also evaluate XGBoost alone for comparison
    import joblib
    xgb_model = joblib.load('./models/saved_models/xgboost.pkl')
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_pred = (xgb_pred_proba >= 0.5).astype(int)
    
    xgb_report = generate_evaluation_report(
        model_name='xgboost_standalone',
        y_true=y_test.values,
        y_pred=xgb_pred,
        y_pred_proba=xgb_pred_proba
    )
    
    # Generate feature importance and SHAP for XGBoost
    plot_feature_importance(xgb_model, X_test.columns.tolist())
    plot_shap_summary(xgb_model, X_test.sample(min(1000, len(X_test))))
    
    logger.info("\n" + "="*60)
    logger.info("Evaluation complete! Check ./outputs/evaluation/ for all plots and reports.")
    logger.info("="*60)


if __name__ == "__main__":
    main()
