import pandas as pd
import numpy as np

def custom_train_test_split(ratings, test_ratio):
    """
    Splitting books ratings into train and test sets.

    Using every user ratings, divide ratings into train and test sets, such that test set is equal to test_ratio*user_ratings and is made of even-indexed book ratings, sorted from the highest rated.

    Parameters
    ----------
    ratings: `DataFrame`
        DataFrame created from "Ratings.csv" file with proper dtype definitions.

    test_ratio: `int`
        coefficient indicating how many ratings will be saved into test set.

    Returns
    -------
    train_set
        dataset created to train the book recommendation model
    
    test_set
        dataset created to test the book recommendation model
    """
    ratings = ratings.sort_values(['User-ID', 'Book-Rating'], ascending=[True, False])

    train_indices = []
    test_indices = []

    for user_id in ratings['User-ID'].unique():
        user_ratings = ratings[ratings['User-ID'] == user_id]

        num_test_ratings = int(np.ceil(len(user_ratings) * test_ratio))

        user_ratings_indices = user_ratings.index.tolist()
        test_indices.extend(user_ratings_indices[1:num_test_ratings*2:2])

    train_indices = list(set(ratings.index) - set(test_indices))

    train_set = ratings.loc[train_indices]
    test_set = ratings.loc[test_indices]

    train_set = train_set.reset_index(drop=True)
    test_set = test_set.reset_index(drop=True)

    return train_set, test_set