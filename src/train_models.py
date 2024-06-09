import os
import joblib
import pandas as pd
import datetime

from scipy.sparse import csr_matrix, save_npz
from sklearn.neighbors import NearestNeighbors
from cleaning_data import clean_data

def create_user_item_matrix(data):
    """ Transform the ratings DataFrame to a user-item matrix. """
    user_item_matrix = data.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    return csr_matrix(user_item_matrix.values)

def train_and_save_knn_model(data, model_path, params):
    user_item_matrix = create_user_item_matrix(data)
    model = NearestNeighbors(**params)

    try:
        model.fit(user_item_matrix)
    except Exception as e:
        print(f"Error training model with params {params}: {e}")
        return

    # Ensure the model directory exists
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    # Find the next available version number by robustly parsing filenames
    existing_files = os.listdir(model_path)
    versions = []
    for file in existing_files:
        # Ensure we only consider relevant files for versioning
        if 'knn_model' in file and 'joblib' in file:
            parts = file.replace('knn_model_', '').split('.')[0]
            try:
                version = int(parts)
                versions.append(version)
            except ValueError:
                continue
    next_version = max(versions) + 1 if versions else 1
    
    # Filename setup with versioning
    model_filename = os.path.join(model_path, f'knn_model_{next_version}.joblib')
    matrix_filename = os.path.join(model_path, f'user_item_matrix_{next_version}.npz')
    params_filename = os.path.join(model_path, f'knn_model_hyperparameters_{next_version}.csv')
    
    # Save the model, user-item matrix, and hyperparameters
    joblib.dump(model, model_filename)
    save_npz(matrix_filename, user_item_matrix)
    pd.DataFrame([params]).to_csv(params_filename, index=False)
    
    print(f"Saved Model to {model_filename}")
    print(f"Saved User-Item Matrix to {matrix_filename}")
    print(f"Saved Hyperparameters to {params_filename}")

# Example usage
ratings_df = pd.read_csv('resources/archive/ratings.csv')
cleaned_ratings = clean_data(ratings_df, 10, 10)
os.system('cls')
print("Data imported")
n_neighbors_list = [2, 5, 10, 20]
metrics_list = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
algorithms_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
leaf_size_list = [30, 45, 60, 75]

# Get the current date and time
today = datetime.datetime.now().strftime("%y%m%d%H%M")

for i, n_neighbors in enumerate(n_neighbors_list):
    for j, metric in enumerate(metrics_list):
        for k, algorithm in enumerate(algorithms_list):
            for l, leaf_size in enumerate(leaf_size_list):
                model_params = {
                    'n_neighbors': n_neighbors,
                    'metric': metric,
                    'algorithm': algorithm,
                    'leaf_size': leaf_size
                }
                model_path = f'src/ModelsKNN/Research{today}KNN'
                train_and_save_knn_model(cleaned_ratings, model_path, model_params)
                os.system('cls')
                print("Iteration")
                print(f"{i+1}/{len(n_neighbors_list)}")
                print(f"{j+1}/{len(metrics_list)}")
                print(f"{k+1}/{len(algorithms_list)}")
                print(f"{l+1}/{len(leaf_size_list)}")
                print("done")
