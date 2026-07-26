import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
    classification_report
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def run_evaluation(df_scored):
    print("Evaluating Anomaly Detection & Attack Classification Performance...")

    y_true_binary = (df_scored["label"] != "normal").astype(int)
    y_pred_binary = df_scored["is_anomaly"].astype(int)
    risk_scores = df_scored["risk_score"].values

    cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
    tn, fp, fn, tp = cm_binary.ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="binary"
    )
    acc = accuracy_score(y_true_binary, y_pred_binary)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    try:
        roc_auc = roc_auc_score(y_true_binary, risk_scores / 100.0)
    except Exception:
        roc_auc = 0.0

    top_1_percent_count = max(1, int(len(df_scored) * 0.01))
    top_alerts = df_scored.sort_values(by="risk_score", ascending=False).head(top_1_percent_count)
    top_1_precision = (top_alerts["label"] != "normal").mean()

    df_attacks_only = df_scored[df_scored["label"] != "normal"].copy()
    
    if len(df_attacks_only) > 0:
        y_true_class = df_attacks_only["label"]
        y_pred_class = df_attacks_only["predicted_attack"]
        
        class_report = classification_report(y_true_class, y_pred_class, output_dict=True)
        unique_classes = sorted(list(set(y_true_class) | set(y_pred_class)))
        cm_multiclass = confusion_matrix(y_true_class, y_pred_class, labels=unique_classes)
        cm_multiclass_list = cm_multiclass.tolist()
    else:
        class_report = {}
        cm_multiclass_list = []
        unique_classes = []

    report_dict = {
        "dataset_summary": {
            "total_logs": len(df_scored),
            "normal_logs": int((df_scored["label"] == "normal").sum()),
            "attack_logs": int((df_scored["label"] != "normal").sum()),
            "detected_anomalies": int(y_pred_binary.sum())
        },
        "anomaly_detection_metrics": {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "false_positive_rate": float(fpr),
            "top_1_percent_alert_precision": float(top_1_precision),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            }
        },
        "attack_classification_metrics": {
            "classes": unique_classes,
            "confusion_matrix": cm_multiclass_list,
            "detailed_report": class_report
        }
    }

    json_path = os.path.join(ROOT_DIR, "reports", "evaluation_report.json")
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    txt_path = os.path.join(ROOT_DIR, "reports", "evaluation_summary.txt")
    with open(txt_path, "w") as f:
        f.write("=====================================================\n")
        f.write(" CYBERSECURITY BEHAVIORAL ANOMALY DETECTION REPORT  \n")
        f.write("=====================================================\n\n")
        f.write(f"Total Logs Evaluated : {len(df_scored)}\n")
        f.write(f"Normal Events        : {(df_scored['label'] == 'normal').sum()}\n")
        f.write(f"Attacks Injected     : {(df_scored['label'] != 'normal').sum()}\n\n")
        f.write("--- ANOMALY DETECTION METRICS (ISOLATION FOREST) ---\n")
        f.write(f"Accuracy               : {acc * 100:.2f}%\n")
        f.write(f"Precision              : {precision * 100:.2f}%\n")
        f.write(f"Recall                 : {recall * 100:.2f}%\n")
        f.write(f"F1-Score               : {f1 * 100:.2f}%\n")
        f.write(f"ROC-AUC Score          : {roc_auc:.4f}\n")
        f.write(f"False Positive Rate    : {fpr * 100:.2f}%\n")
        f.write(f"Top 1% Alert Precision : {top_1_precision * 100:.2f}%\n\n")
        f.write("--- CONFUSION MATRIX ---\n")
        f.write(f"True Negatives  (Normal -> Normal)  : {tn}\n")
        f.write(f"False Positives (Normal -> Anomaly) : {fp}\n")
        f.write(f"False Negatives (Attack -> Normal)  : {fn}\n")
        f.write(f"True Positives  (Attack -> Anomaly) : {tp}\n\n")
        f.write("=====================================================\n")

    print(f"Evaluation report successfully saved to '{json_path}' and '{txt_path}'.")

    return report_dict

if __name__ == "__main__":
    full_path = os.path.join(ROOT_DIR, "data", "full_risk_scored_logs.csv")
    if not os.path.exists(full_path):
        from risk_engine import calculate_risk_scores
        df_classified = pd.read_csv(os.path.join(ROOT_DIR, "data", "classified_logs.csv"))
        df_full, _ = calculate_risk_scores(df_classified)
    else:
        df_full = pd.read_csv(full_path)

    run_evaluation(df_full)
