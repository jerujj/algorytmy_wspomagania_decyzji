import os
import joblib
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import StandardScaler
import random
from cleaning_data import clean_data

def load_model_and_data(model_path, version):
    model_filename = os.path.join(model_path, f'nmf_model_{version}.joblib')
    matrix_filename = os.path.join(model_path, f'user_item_matrix_{version}.npz')
    scaler_filename = os.path.join(model_path, f'scaler_{version}.joblib')
    
    W, H, model = joblib.load(model_filename)
    user_item_matrix = load_npz(matrix_filename)
    scaler = joblib.load(scaler_filename)
    
    return W, H, model, user_item_matrix, scaler

def create_user_vector(user_ratings, num_items):
    user_vector = np.zeros(num_items)
    for item, rating in user_ratings.items():
        user_vector[item] = rating
    return user_vector

def predict_for_user(W, H, scaler, new_user_vector, n_recommendations=6):
    new_user_vector = scaler.transform([new_user_vector])
    user_latent = np.dot(new_user_vector, H.T)
    scores = np.dot(user_latent, H)
    
    recommended_items = np.argsort(scores.flatten())[-n_recommendations:][::-1]
    
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
    joblib_files = [file for file in files if file.endswith('.joblib') and 'nmf_model' in file]
    return len(joblib_files)

def test_nmf_models(model_path, ratings_df):
    results = []
    unique_users = ratings_df['user_id'].unique()
    random_users = random.sample(list(unique_users), 100)

    num_models = count_models(model_path)
    
    for version in range(1, num_models + 1):
        W, H, model, user_item_matrix, scaler = load_model_and_data(model_path, version)
        num_items = ratings_df['book_id'].nunique()
        
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
            
            recommendations = predict_for_user(W, H, scaler, test_user_vector)
            points = evaluate_recommendations(user_vector, recommendations)
            total_points += points
            
        results.append({
            'model_version': version,
            'points': total_points
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('nmf_model_evaluation_results.csv', index=False)
    results_df.to_csv(os.path.join(model_path, 'nmf_model_evaluation_results.csv'), index=False)

# Load ratings data
ratings_df = pd.read_csv('resources/archive/ratings.csv')
cleaned_ratings = clean_data(ratings_df, 10, 10)

# Specify the model path
model_path = 'src/ModelsKNN/Research2406070028NMF'

# Test the NMF models and save the results
test_nmf_models(model_path, cleaned_ratings)
