# src/data/load_data.py
from sklearn.datasets import load_iris
import pandas as pd

def load_iris_df(save_csv=False):
    """
    Returns a pandas DataFrame with iris features and 'target' column.
    If save_csv=True, saves a copy to data/sample_iris.csv
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    if save_csv:
        import os
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/sample_iris.csv', index=False)
    return df
