# src/utils/helpers.py
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_decision_tree(clf, feature_names, class_names=None):
    plt.figure(figsize=(12,8))
    plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
    plt.tight_layout()
    plt.show()

def plot_confusion(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
