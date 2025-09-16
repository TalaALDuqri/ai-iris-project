# src/models/model.py
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.data.load_data import load_iris_df

def train_decision_tree(test_size=0.2, random_state=42):
    df = load_iris_df()
    X = df.iloc[:, :-1]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    # return model and test sets for evaluation/visualization
    return clf, X_test, y_test, y_pred
