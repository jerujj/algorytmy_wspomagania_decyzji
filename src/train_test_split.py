import pandas as pd
import numpy as np

def train_test_split(ratings, test_ratio, seed=42):
    np.random.seed(seed)  # For reproducibility
    # Shuffle each user's ratings and split into train and test sets
    def split_user_ratings(df):
        df = df.sample(frac=1, random_state=seed)  # Shuffle
        n_test = int(len(df) * test_ratio)
        test_df = df.iloc[:n_test]
        train_df = df.iloc[n_test:]
        return train_df, test_df

    # Group by user_id and apply the splitting function
    train_list, test_list = zip(*ratings.groupby('user_id').apply(split_user_ratings))
    train = pd.concat(train_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)
    
    return train, test

# Example usage
"""ratings_df = pd.read_csv('resources/archive/ratings.csv')
train_set, test_set = train_test_split(ratings_df, 0.2)
print("Training Set:")
print(f"Total rows: {train_set.shape[0]}")
print(f"Sample data:\n{train_set.sample(5, random_state=1)}") 

print("\nTesting Set:")
print(f"Total rows: {test_set.shape[0]}")
print(f"Sample data:\n{test_set.sample(5, random_state=1)}")"""