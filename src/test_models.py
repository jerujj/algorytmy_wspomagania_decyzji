import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import random
from cleaning_data import clean_data
from tqdm import tqdm

def load_model_and_data(model_path, version):
    model_filename = os.path.join(model_path, f'knn_model_{version}.joblib')
    matrix_filename = os.path.join(model_path, f'user_item_matrix_{version}.npz')
    
    model = joblib.load(model_filename)
    user_item_matrix = load_npz(matrix_filename)
    
    return model, user_item_matrix

def create_user_vector(user_ratings, num_items):
    user_vector = np.zeros(num_items)
    for item, rating in user_ratings.items():
        if item > 9998:
            item = 9998
        user_vector[item] = rating
    return user_vector

def predict_for_user(model, user_item_matrix, new_user_vector, n_neighbors, n_recommendations):
    _, indices = model.kneighbors([new_user_vector], n_neighbors=n_neighbors)
    similar_users = indices.flatten()
    
    similar_users_matrix = user_item_matrix[similar_users].toarray()
    mean_ratings = similar_users_matrix.mean(axis=0)
    
    unrated_items_mask = new_user_vector == 0
    mean_ratings[~unrated_items_mask] = -np.inf
    recommended_items = np.argsort(mean_ratings)[-n_recommendations:][::-1]
    
    return recommended_items

def evaluate_recommendations(original_user_vector, recommendations, n_recommendations):
    non_zero_ratings_mask = original_user_vector > 0
    non_zero_indices = np.where(non_zero_ratings_mask)[0]
    sorted_indices = np.argsort(original_user_vector[non_zero_indices])
    top_liked_indices = non_zero_indices[sorted_indices[-2:]]
    least_liked_indices = non_zero_indices[sorted_indices[:2]]

    points = 0
    max_points = n_recommendations * 10  # The maximum possible points is 10 points per recommendation

    for i, item in enumerate(recommendations):
        if item in top_liked_indices:
            rank = np.where(top_liked_indices == item)[0][0]
            # Points decrease linearly from n_recommendations to 1
            if rank == 0:
                points += (n_recommendations - i) * 10
            else:
                points += max((n_recommendations - 1 - i), 1) * 10
        elif item in least_liked_indices:
            rank = np.where(least_liked_indices == item)[0][0]
            # Negative points decrease linearly from n_recommendations to 1
            if rank == 0:
                points -= (n_recommendations - i) * 10
            else:
                points -= max((n_recommendations - 1 - i), 1) * 10

    return points / max_points

def count_models(model_path):
    files = os.listdir(model_path)
    joblib_files = [file for file in files if file.endswith('.joblib')]
    return len(joblib_files)

def test_knn_models(model_path, ratings_df, n_recommendations_list, n_neighbors_list):
    results = []
    unique_users = ratings_df['user_id'].unique()
    random_users = random.sample(list(unique_users), 100)

    num_models = count_models(model_path)
    
    for version in tqdm(range(1, num_models + 1), "KNN version"):
        model, user_item_matrix = load_model_and_data(model_path, version)
        num_items = user_item_matrix.shape[1]
        
        for n_recommendations in n_recommendations_list:
            for n_neighbors in n_neighbors_list:
                total_points = 0
                for user_id in random_users:
                    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
                    user_ratings_dict = user_ratings.set_index('book_id')['rating'].to_dict()
                    user_vector = create_user_vector(user_ratings_dict, num_items)
                    
                    test_user_vector = user_vector.copy()
                    non_zero_ratings_mask = test_user_vector > 0
                    non_zero_indices = np.where(non_zero_ratings_mask)[0]
                    sorted_indices = np.argsort(test_user_vector[non_zero_indices])
                    top_liked_indices = non_zero_indices[sorted_indices[-2:]]
                    least_liked_indices = non_zero_indices[sorted_indices[:2]]
                    test_user_vector[top_liked_indices] = 0
                    test_user_vector[least_liked_indices] = 0
                    
                    recommendations = predict_for_user(model, user_item_matrix, test_user_vector, n_neighbors, n_recommendations)
                    points_ratio = evaluate_recommendations(user_vector, recommendations, n_recommendations)
                    total_points += points_ratio
                
                results.append({
                    'model_version': version,
                    'n_recommendations': n_recommendations,
                    'n_neighbors': n_neighbors,
                    'points_ratio': total_points / len(random_users)
                })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(model_path, 'knn_model_evaluation_results.csv'), index=False)

# Load ratings data
ratings_df = pd.read_csv('resources/archive/ratings.csv')
cleaned_ratings = clean_data(ratings_df, 10, 10)

# Specify the model path and the number of models
model_path = 'src/ModelsKNN/Research2406061853KNN'

# Define the values for n_recommendations and n_neighbors
n_recommendations_list = [5, 10, 20, 50]
n_neighbors_list = [1, 2, 5, 10, 20, 50]

# Test the KNN models and save the results
test_knn_models(model_path, cleaned_ratings, n_recommendations_list, n_neighbors_list)
