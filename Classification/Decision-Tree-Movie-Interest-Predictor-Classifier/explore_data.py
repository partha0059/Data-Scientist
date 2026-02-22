import pandas as pd

try:
    df = pd.read_csv('Task_Dataset_Movie_Interests_DecisionTree.csv')
    print("Columns:", df.columns.tolist())
    print("\nShape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum())
    print("\nUnique Values in 'Interest':", df['Interest'].unique())
    print("\nUnique Values in 'Gender':", df['Gender'].unique())
    print("\nDescription:\n", df.describe())
except Exception as e:
    print(f"Error: {e}")
