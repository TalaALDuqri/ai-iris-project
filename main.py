# main.py
from src.models.model import train_decision_tree
from src.data.load_data import load_iris_df
from src.utils.helpers import plot_decision_tree, plot_confusion

if __name__ == "__main__":
    clf, X_test, y_test, y_pred = train_decision_tree()
    df = load_iris_df()
    feature_names = df.columns[:-1].tolist()
    class_names = ['setosa', 'versicolor', 'virginica']
    # plot confusion matrix and decision tree
    plot_confusion(y_test, y_pred, labels=class_names)
    plot_decision_tree(clf, feature_names, class_names)
