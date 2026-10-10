"""
Train and Evaluation Pipeline for Structural Damage Image Classification.
Calculates syllabus metric comparison table, confusion matrices, ROC curves,
PCA projection, sample prediction visual gallery, and saves model artifacts.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, roc_curve
)

from config import (
    CONFUSION_PLOT_PATH, ROC_PLOT_PATH, PCA_PLOT_PATH,
    SAMPLE_PRED_PATH, METRICS_CSV_PATH, MODELS_DIR,
    ROC_DATA_PATH, SAMPLE_DIR
)
from models import initialize_all_models
from preprocessing import preprocess_image


def evaluate_model_performance(model, X_test, y_test, model_name):
    """
    Computes detailed evaluation metrics including confusion matrix breakdown.
    Metrics strictly follow course guidelines:
    - Overall Accuracy
    - Sensitivity (Recall)
    - Specificity: TN / (TN + FP)
    - Precision
    - F1-Score
    - ROC-AUC
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.decision_function(X_test)
        probs = 1.0 / (1.0 + np.exp(-probs))

    preds = model.predict(X_test)

    cm = confusion_matrix(y_test, preds)
    # cm layout: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()

    total_undamaged = TN + FP
    total_damaged = FN + TP

    true_undamaged_pct = (TN / total_undamaged) * 100.0 if total_undamaged > 0 else 0.0
    false_undamaged_pct = (FP / total_undamaged) * 100.0 if total_undamaged > 0 else 0.0
    true_damaged_pct = (TP / total_damaged) * 100.0 if total_damaged > 0 else 0.0
    false_damaged_pct = (FN / total_damaged) * 100.0 if total_damaged > 0 else 0.0

    acc = accuracy_score(y_test, preds) * 100.0
    rec = recall_score(y_test, preds) * 100.0       # Sensitivity
    spec = (TN / (TN + FP)) * 100.0 if (TN + FP) > 0 else 0.0  # Specificity
    prec = precision_score(y_test, preds) * 100.0 if (TP + FP) > 0 else 0.0
    f1 = f1_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, probs)

    fpr, tpr, thresholds = roc_curve(y_test, probs)

    metrics = {
        'Model': model_name,
        'Overall Accuracy (%)': acc,
        'Sensitivity / Recall (%)': rec,
        'Specificity (%)': spec,
        'Precision (%)': prec,
        'F1-Score': f1,
        'ROC-AUC': roc_auc,
        'True Undamaged (%)': true_undamaged_pct,
        'False Damaged (%)': false_undamaged_pct,
        'True Damaged (%)': true_damaged_pct,
        'False Undamaged (%)': false_damaged_pct,
        'TN': TN, 'FP': FP, 'FN': FN, 'TP': TP,
        'fpr': fpr, 'tpr': tpr, 'cm': cm, 'probs': probs, 'preds': preds
    }
    return metrics


def plot_confusion_matrices(eval_results):
    """Generates side-by-side confusion matrix heatmaps for all models."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    class_labels = ['Undamaged', 'Damaged']

    for idx, (m_name, res) in enumerate(eval_results.items()):
        cm = res['cm']
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100.0
        
        annot_matrix = np.array([
            [f"{cm[0,0]}\n({cm_norm[0,0]:.1f}%)", f"{cm[0,1]}\n({cm_norm[0,1]:.1f}%)"],
            [f"{cm[1,0]}\n({cm_norm[1,0]:.1f}%)", f"{cm[1,1]}\n({cm_norm[1,1]:.1f}%)"]
        ])

        sns.heatmap(cm_norm, annot=annot_matrix, fmt='', cmap='Blues', cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels, ax=axes[idx],
                    annot_kws={"fontsize": 11, "weight": "bold"})
        axes[idx].set_title(f"{m_name}\nAccuracy: {res['Overall Accuracy (%)']:.1f}%", fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
        axes[idx].set_ylabel('True Label', fontsize=10)

    plt.tight_layout()
    plt.savefig(CONFUSION_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Confusion matrices saved to: {CONFUSION_PLOT_PATH}")


def plot_roc_curves(eval_results):
    """Generates comparative ROC curves for all models."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(8, 6))

    colors = ['#1f77b4', '#2ca02c', '#d62728']
    for idx, (m_name, res) in enumerate(eval_results.items()):
        plt.plot(res['fpr'], res['tpr'], color=colors[idx % len(colors)], lw=2.2,
                 label=f"{m_name} (AUC = {res['ROC-AUC']:.4f})")

    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC) Benchmark\nStructural Damage Image Classification (Raw Pixels)',
              fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ROC curves saved to: {ROC_PLOT_PATH}")


def plot_pca_projection(X_pca, y_test):
    """Plots 2D PCA projection of raw pixel vectors."""
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.figure(figsize=(8, 6))

    mask_dmg = (y_test == 1)
    mask_undmg = (y_test == 0)

    plt.scatter(X_pca[mask_undmg, 0], X_pca[mask_undmg, 1], color='#1f77b4', alpha=0.6,
                s=35, label='Undamaged Structural Elements', edgecolors='none')
    plt.scatter(X_pca[mask_dmg, 0], X_pca[mask_dmg, 1], color='#d62728', alpha=0.6,
                s=35, label='Damaged Structural Elements', edgecolors='none')

    plt.title('2D PCA Projection of 2,304-Dimensional Raw Pixel Space\nStructural Damage vs Undamaged Concrete/Masonry',
              fontsize=12, fontweight='bold', pad=12)
    plt.xlabel('Principal Component 1', fontsize=11, fontweight='bold')
    plt.ylabel('Principal Component 2', fontsize=11, fontweight='bold')
    plt.legend(loc='upper right', frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(PCA_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  PCA projection saved to: {PCA_PLOT_PATH}")


def plot_sample_predictions(models):
    """Generates visual prediction gallery on curated benchmark images."""
    dmg_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "Damaged", "*.*")))[:3]
    undmg_files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "Undamaged", "*.*")))[:3]
    all_files = [(f, 1) for f in dmg_files] + [(f, 0) for f in undmg_files]

    if not all_files:
        return

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, (fpath, true_label) in enumerate(all_files):
        if i >= 6:
            break
        vec, gray = preprocess_image(fpath)
        vec_reshaped = vec.reshape(1, -1)

        votes = []
        for name, m in models.items():
            pred = m.predict(vec_reshaped)[0]
            votes.append(pred)

        consensus_label = "DAMAGE DETECTED" if sum(votes) >= 2 else "UNDAMAGED"
        true_text = "Damaged" if true_label == 1 else "Undamaged"
        color = 'red' if consensus_label == "DAMAGE DETECTED" else 'green'

        axes[i].imshow(gray, cmap='gray')
        fname_short = os.path.basename(fpath).replace('_', ' ').replace('.jpg', '')[:25]
        axes[i].set_title(f"Truth: {true_text}\nConsensus: {consensus_label} ({sum(votes)}/3)\n{fname_short}",
                          fontsize=9, fontweight='bold', color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(SAMPLE_PRED_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Sample predictions gallery saved to: {SAMPLE_PRED_PATH}")


def train_and_eval_all(data_dict):
    """
    Trains all 3 classical models, performs evaluation, generates all figures,
    and exports summary table and joblib models.
    """
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    X_pca_test = data_dict['X_pca_test']

    models = initialize_all_models()
    eval_results = {}
    roc_export = {}

    print("\nTraining and evaluating syllabus models...")
    for m_name, model in models.items():
        print(f"  Training {m_name}...")
        model.fit(X_train, y_train)

        metrics = evaluate_model_performance(model, X_test, y_test, m_name)
        eval_results[m_name] = metrics
        roc_export[m_name] = {
            'fpr': metrics['fpr'],
            'tpr': metrics['tpr'],
            'auc': metrics['ROC-AUC']
        }

        # Save model joblib
        safe_name = m_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '').replace('-', '_')
        save_path = os.path.join(MODELS_DIR, f"{safe_name}.joblib")
        joblib.dump(model, save_path)

    # Save ROC curves data for Streamlit app
    joblib.dump(roc_export, ROC_DATA_PATH)

    # Build DataFrame matching Table 1
    rows = []
    for m_name, res in eval_results.items():
        rows.append({
            'Model': m_name,
            'Accuracy (%)': round(res['Overall Accuracy (%)'], 2),
            'Sensitivity (%)': round(res['Sensitivity / Recall (%)'], 2),
            'Specificity (%)': round(res['Specificity (%)'], 2),
            'Precision (%)': round(res['Precision (%)'], 2),
            'F1-Score': round(res['F1-Score'], 4),
            'ROC-AUC': round(res['ROC-AUC'], 4),
            'True Undamaged (%)': round(res['True Undamaged (%)'], 2),
            'False Damaged (%)': round(res['False Damaged (%)'], 2),
            'True Damaged (%)': round(res['True Damaged (%)'], 2),
            'False Undamaged (%)': round(res['False Undamaged (%)'], 2)
        })

    df_results = pd.DataFrame(rows)
    df_results.to_csv(METRICS_CSV_PATH, index=False)
    print(f"\nBenchmark Metrics Summary:\n{df_results[['Model', 'Accuracy (%)', 'Sensitivity (%)', 'Specificity (%)', 'F1-Score', 'ROC-AUC']]}")

    # Generate Visualizations Suite
    print("\nGenerating visual figures in outputs/...")
    plot_confusion_matrices(eval_results)
    plot_roc_curves(eval_results)
    plot_pca_projection(X_pca_test, y_test)
    plot_sample_predictions(models)

    return df_results, eval_results, models
