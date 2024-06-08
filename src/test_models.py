import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
import random
from cleaning_data import clean_data

def load_model_and_data(model_path, version):
    model_filename = os.path.join(model_path, f'knn_model_{version}.joblib')
    matrix_filename = os.path.join(model_path, f'user_item_matrix_{version}.npz')
    
    model = joblib.load(model_filename)
    user_item_matrix = load_npz(matrix_filename)
    
    return model, user_item_matrix

def create_user_vector(user_ratings, num_items):
    user_vector = np.zeros(num_items)
    for item, rating in user_ratings.items():
        user_vector[item] = rating
    return user_vector

def predict_for_user(model, user_item_matrix, new_user_vector, n_recommendations=20):
    _, indices = model.kneighbors([new_user_vector], n_neighbors=20)
    similar_users = indices.flatten()
    
    similar_users_matrix = user_item_matrix[similar_users].toarray()
    mean_ratings = similar_users_matrix.mean(axis=0)
    
    unrated_items_mask = new_user_vector == 0
    mean_ratings[~unrated_items_mask] = -np.inf
    recommended_items = np.argsort(mean_ratings)[-n_recommendations:][::-1]
    
    return recommended_items

def evaluate_recommendations(original_user_vector, recommendations):
    top_liked_indices = np.argsort(original_user_vector)[-2:]
    least_liked_indices = np.argsort(original_user_vector)[:2]
    
    points = 0
    for i, item in enumerate(recommendations):
        if item in top_liked_indices:
            rank = np.where(top_liked_indices == item)[0][0]
            if rank == 0:
                points += (6 - i) * 10
            else:
                points += max((5 - i), 1) * 10
        elif item in least_liked_indices:
            rank = np.where(least_liked_indices == item)[0][0]
            if rank == 0:
                points -= (6 - i) * 10
            else:
                points -= max((5 - i), 1) * 10
                
    return points

def count_models(model_path):
    files = os.listdir(model_path)
    joblib_files = [file for file in files if file.endswith('.joblib')]
    return len(joblib_files)

def test_knn_models(model_path, ratings_df):
    results = []
    unique_users = ratings_df['user_id'].unique()
    random_users = random.sample(list(unique_users), 100)

    num_models = count_models(model_path)
    
    for version in range(1, num_models + 1):
        model, user_item_matrix = load_model_and_data(model_path, version)
        num_items = user_item_matrix.shape[1]
        
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
            
            recommendations = predict_for_user(model, user_item_matrix, test_user_vector)
            points = evaluate_recommendations(user_vector, recommendations)
            total_points += points
            
        results.append({
            'model_version': version,
            'points': total_points
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(model_path, 'knn_model_evaluation_results_v3.csv'), index=False)

# Load ratings data
ratings_df = pd.read_csv('resources/archive/ratings.csv')
cleaned_ratings = clean_data(ratings_df, 10, 10)

# Specify the model path and the number of models
model_path = 'src/ModelsKNN/Research2406061853KNN'

# Test the KNN models and save the results
test_knn_models(model_path, cleaned_ratings)
