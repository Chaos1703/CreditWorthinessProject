from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

def evaluate_credit_model(model_name, y_true, y_pred=None, y_proba=None, threshold=0.5):
    # Generate predictions from probabilities if provided
    if y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)

    # Classification metrics
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Good Credit (0)", "Bad Credit (1)"],
        output_dict=True
    )
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba) if y_proba is not None else None

    # Print results
    print(f"\n🔍 [{model_name}] Classification Report @ Threshold = {threshold}")
    print("="*50)
    print(classification_report(
        y_true, y_pred, target_names=["Good Credit (0)", "Bad Credit (1)"]
    ))
    print("="*50)

    print("🧠 Explanation of Key Metrics:")
    print("- Precision (Good Credit): {:.2f} → {:.0f}% of those predicted as good credit were correct.".format(
        report['Good Credit (0)']['precision'], report['Good Credit (0)']['precision'] * 100))
    print("- Recall (Good Credit)   : {:.2f} → {:.0f}% of all actual good credits were correctly identified.".format(
        report['Good Credit (0)']['recall'], report['Good Credit (0)']['recall'] * 100))
    print("- Precision (Bad Credit) : {:.2f} → {:.0f}% of those predicted as bad credit were correct.".format(
        report['Bad Credit (1)']['precision'], report['Bad Credit (1)']['precision'] * 100))
    print("- Recall (Bad Credit)    : {:.2f} → {:.0f}% of all actual bad credits were correctly identified.".format(
        report['Bad Credit (1)']['recall'], report['Bad Credit (1)']['recall'] * 100))
    print("- Accuracy               : {:.2f} → {:.0f}% of all predictions were correct.".format(
        acc, acc * 100))

    if roc_auc is not None:
        print("- ROC-AUC                : {:.2f} → Model distinguishes good vs. bad credit with {:.0f}% confidence.".format(
            roc_auc, roc_auc * 100))
    print("="*50)


def plot_default_probability_distribution(model_name, y_true, y_proba, threshold=0.5):
    plt.figure(figsize=(8, 5))
    
    sns.histplot(y_proba[y_true == 0], bins=20, color='green', label='Good Credit (0)', stat="density", kde=True)
    sns.histplot(y_proba[y_true == 1], bins=20, color='red', label='Bad Credit (1)', stat="density", kde=True)
    
    plt.axvline(threshold, color='black', linestyle='--', label=f'Threshold = {threshold:.2f}')
    
    plt.title(f'{model_name}: Probability of Default Distribution')
    plt.xlabel('Predicted Probability of Default')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def find_best_threshold(y_true, y_probs):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return best_threshold, best_f1, precisions[best_idx], recalls[best_idx]


def modelValuation(model_name, y_true, y_proba, use_best_threshold=False):
    if use_best_threshold:
        threshold, best_f1, precision, recall = find_best_threshold(y_true, y_proba)
        print(f"\n[⚙️ {model_name}] Using Best Threshold = {threshold:.4f} (F1 = {best_f1:.3f}, "
              f"P = {precision:.3f}, R = {recall:.3f})")
    else:
        threshold = 0.5
        print(f"\n[⚙️ {model_name}] Using Default Threshold = 0.5")

    y_pred = (y_proba >= threshold).astype(int)

    evaluate_credit_model(model_name, y_true, y_pred=y_pred, y_proba=y_proba, threshold=threshold)

    plot_default_probability_distribution(model_name, y_true, y_proba, threshold=threshold)

    return y_pred, threshold

