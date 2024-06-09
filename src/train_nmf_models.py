import os
import joblib
import pandas as pd
import numpy as np
import datetime
from scipy.sparse import csr_matrix, save_npz
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
from cleaning_data import clean_data

def create_user_item_matrix(data):
    """ Transform the ratings DataFrame to a user-item matrix. """
    user_item_matrix = data.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
    return csr_matrix(user_item_matrix.values)

def train_and_save_nmf_model(data, model_path, params):
    user_item_matrix = create_user_item_matrix(data)
    
    # Scale the data
    scaler = StandardScaler(with_mean=False)
    user_item_matrix = scaler.fit_transform(user_item_matrix)
    
    model = NMF(**params)
    
    try:
        W = model.fit_transform(user_item_matrix)
        H = model.components_
    except Exception as e:
        print(f"Error training model with params {params}: {e}")
        return

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    existing_files = os.listdir(model_path)
    versions = []
    for file in existing_files:
        if 'nmf_model' in file and 'joblib' in file:
            parts = file.replace('nmf_model_', '').split('.')[0]
            try:
                version = int(parts)
                versions.append(version)
            except ValueError:
                continue
    next_version = max(versions) + 1 if versions else 1
    
    model_filename = os.path.join(model_path, f'nmf_model_{next_version}.joblib')
    matrix_filename = os.path.join(model_path, f'user_item_matrix_{next_version}.npz')
    scaler_filename = os.path.join(model_path, f'scaler_{next_version}.joblib')
    params_filename = os.path.join(model_path, f'nmf_model_hyperparameters_{next_version}.csv')
    
    joblib.dump((W, H, model), model_filename)
    save_npz(matrix_filename, user_item_matrix)
    joblib.dump(scaler, scaler_filename)
    pd.DataFrame([params]).to_csv(params_filename, index=False)
    
    print(f"Saved Model to {model_filename}")
    print(f"Saved User-Item Matrix to {matrix_filename}")
    print(f"Saved Scaler to {scaler_filename}")
    print(f"Saved Hyperparameters to {params_filename}")

ratings_df = pd.read_csv('resources/archive/ratings.csv')
cleaned_ratings = clean_data(ratings_df, 10, 10)
os.system('cls')
print("Data imported")

# Example parameter sets for NMF
n_components_list = [10, 20, 30]
init_list = ['random', 'nndsvd']
alpha_W_list = [0.1, 0.5, 1.0]

today = datetime.datetime.now().strftime("%y%m%d%H%M")

for i, n_components in enumerate(n_components_list):
    for j, init in enumerate(init_list):
        for k, alpha_W in enumerate(alpha_W_list):
            model_params = {
                'n_components': n_components,
                'init': init,
                'alpha_W': alpha_W,
                'alpha_H': alpha_W,  # Use the same alpha for H to keep it simple
                'l1_ratio': 0.5
            }
            model_path = f'src/ModelsKNN/Research{today}NMF'
            train_and_save_nmf_model(cleaned_ratings, model_path, model_params)
            os.system('cls')
            print("Iteration")
            print(f"{i+1}/{len(n_components_list)}")
            print(f"{j+1}/{len(init_list)}")
            print(f"{k+1}/{len(alpha_W_list)}")
            print("done")
