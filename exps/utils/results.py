import pandas as pd
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from icecream import ic

# Function to load and evaluate predictions from CSV file
def evaluate_multiclass_classification(path):
    # Load the CSV file
    df = pd.read_csv(path+'results.csv')
    # Read labels from file to an array
    with open(path+'labels.txt', 'r') as file:
        labels = file.read().splitlines()

    # Extract predictions and ground truth
    y_pred = df['Predictions']
    y_true = df['Ground Truth']

    # Calculate accuracy
    accuracy = metrics.accuracy_score(y_true, y_pred)
    # Calculate balanced accuracy
    balanced_accuracy = metrics.balanced_accuracy_score(y_true, y_pred)

    # Calculate weighted precision, recall, and F1-score
    precision = metrics.precision_score(y_true, y_pred, average='weighted')
    recall = metrics.recall_score(y_true, y_pred, average='weighted')
    f1_score = metrics.f1_score(y_true, y_pred, average='weighted')

    # Calculate macro-averaged precision, recall, and F1-score
    macro_precision = metrics.precision_score(y_true, y_pred, average='macro')
    macro_recall = metrics.recall_score(y_true, y_pred, average='macro')
    macro_f1_score = metrics.f1_score(y_true, y_pred, average='macro')

    # Calculate micro-averaged precision, recall, and F1-score
    micro_precision = metrics.precision_score(y_true, y_pred, average='micro')
    micro_recall = metrics.recall_score(y_true, y_pred, average='micro')
    micro_f1_score = metrics.f1_score(y_true, y_pred, average='micro')

    # Calculate AUC ROC
    label_mapping = {label: i for i, label in enumerate(labels)}
    y_true_numeric = np.array([label_mapping[label] for label in y_true])
    y_true_binarized = label_binarize(y_true_numeric, classes=np.unique(y_true_numeric))
    y_pred_numeric = np.array([label_mapping[label] for label in y_pred])
    y_pred_binarized = label_binarize(y_pred_numeric, classes=np.unique(y_pred_numeric))
    auc_roc_ovr = metrics.roc_auc_score(y_true_binarized, y_pred_binarized, average='macro', multi_class='ovr')
    auc_roc_ovo = metrics.roc_auc_score(y_true_binarized, y_pred_binarized, average='macro', multi_class='ovo')

    # Calculate Cohen's kappa
    cohen_kappa = metrics.cohen_kappa_score(y_true, y_pred)

    # Calculate confusion matrix
    conf_matrix = metrics.confusion_matrix(y_true, y_pred)

    # Print the metrics
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Balanced Accuracy: {balanced_accuracy:.4f}')
    print('')
    print(f'Precision (weighted): {precision:.4f}')
    print(f'Recall (weighted): {recall:.4f}')
    print(f'F1-score (weighted): {f1_score:.4f}')
    print('')
    print(f'Precision (macro): {macro_precision:.4f}')
    print(f'Recall (macro): {macro_recall:.4f}')
    print(f'F1-score (macro): {macro_f1_score:.4f}')
    print('')
    print(f'Precision (micro): {micro_precision:.4f}')
    print(f'Recall (micro): {micro_recall:.4f}')
    print(f'F1-score (micro): {micro_f1_score:.4f}')
    print('')
    print(f'AUC ROC (macro - OVR): {auc_roc_ovr:.4f}')
    print(f'AUC ROC (macro - OVO): {auc_roc_ovo:.4f}')
    print('')
    print(f'Cohen\'s Kappa: {cohen_kappa:.4f}')

    # Plot the confusion matrix
    plt.figure(figsize=(20, 8))

    # Plot the first subplot: original confusion matrix
    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')

    # Standardize the confusion matrix
    standardized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)

    # Plot the second subplot: standardized confusion matrix
    plt.subplot(1, 2, 2)
    sns.heatmap(standardized_conf_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Standardized Confusion Matrix')

    # Adjust the layout
    plt.tight_layout()

    # Show the figure
    plt.show()

# Example usage:
if __name__ == "__main__":
    path = './'  # Replace with your CSV file path
    evaluate_multiclass_classification(path)
