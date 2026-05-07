import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns

def scanlist_subgroup_check(df, sg_axis, diagnosis_col='diagnosis'):
    print(f"\n===== Subgroup wise counts =====")
    for gax in sg_axis:
        ncounts = df.groupby([diagnosis_col,gax]).size().reset_index(name='count')
        print(f"=> Evaluation axis: {gax}")
        print(f"{ncounts.to_string(index=False)}")
    print(f"================================\n")

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_roc_curve(y_true, y_probs, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0,1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def get_eval_metrics(y_true, y_pred, pos_label=1, average='binary'):
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=pos_label, average=average)
    recall = recall_score(y_true, y_pred, pos_label=pos_label, average=average)
    return cm, precision, recall

def get_metric(df, label_col, predict_col, prob_col, positive_label):
    y_true = (df[label_col]==positive_label).astype(int)
    y_pred = (df[predict_col]==positive_label).astype(int)
    y_prob_positive = np.where(
        df[predict_col] == positive_label,
        df[prob_col],
        1 - df[prob_col]
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0),
        "f1score": f1_score(y_true, y_pred, average='binary', zero_division=0),
        "auroc": roc_auc_score(y_true, y_prob_positive),
        "auprc": average_precision_score(y_true, y_prob_positive),
        "confidence": np.mean(df.loc[df[label_col]==positive_label, prob_col])
    }
    return metrics

def group_analysis(csv_file, label_col, predict_col, prob_col, sensitive_col, positive_label, output_file=None):
    df = pd.read_csv(csv_file)
    grp_metrics = {}
    metrics = get_metric(df, label_col, predict_col, prob_col, positive_label)
    for met_id, met_val in metrics.items():
        if met_id not in grp_metrics:
            grp_metrics[met_id] = {}
        grp_metrics[met_id]['overall'] = met_val
    if sensitive_col == 'age':
        intervals = [0,65,100]
        for idx in range(len(intervals)-1):
            mask = ((df['age']>intervals[idx]) & (df['age']<=intervals[idx+1]))
            df.loc[mask,'age_group'] = f"({intervals[idx]},{intervals[idx+1]}]"
        sensitive_col = 'age_group'
    axis_vals = df[sensitive_col].unique().tolist()
    for aval in axis_vals:
        mask = (df[sensitive_col]==aval)
        metrics = get_metric(df[mask], label_col, predict_col, prob_col, positive_label)
        for met_id, met_val in metrics.items():
            grp_metrics[met_id][aval] = met_val
    for metric in grp_metrics.keys():
        metvals = np.array([grp_metrics[metric][aid] for aid in axis_vals])
        biasval = (np.max(metvals) - np.min(metvals)) #/grp_metrics[metric]['overall'] if (grp_metrics[metric]['overall']!=0.0) else 0.0
        grp_metrics[metric]['bias'] = float(biasval)
    report = f"Evaluation axis: {sensitive_col} - {axis_vals}\n"
    for metric in grp_metrics.keys():
        for aval in ['overall','bias']+axis_vals:
            prefix_string = f"{metric}_{aval}"
            report += f"{prefix_string:30s}: {grp_metrics[metric][aval]:.4f}\n"
    print(report)
    if output_file:
        with open(output_file,'a') as f:
            f.write(report + "\n")
    return grp_metrics

def plot_hbar(df, labels, opfile=None):
    ysize = max(5,round(len(df[labels[0]])*0.25))
    plt.figure(figsize=(12,ysize))
    sns.barplot(data=df, x=labels[1], y=labels[0], orient='h')
    plt.yticks(fontsize=10)
    plt.tight_layout()
    if opfile:
        plt.savefig(opfile)
        plt.close()
    else:
        plt.show()