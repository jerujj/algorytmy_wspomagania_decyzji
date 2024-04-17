from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from data_cleaning import *

def recommend_books(user_id, n_recommendations):
    books, ratings, _, _, _, _ = clean_csv_files(10)

    # Mapowanie User-ID i ISBN do unikalnych indeksów
    user_ids = ratings['User-ID'].unique()
    isbn_ids = ratings['ISBN'].unique()

    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    isbn_to_idx = {isbn: idx for idx, isbn in enumerate(isbn_ids)}

    # Tworzenie macierzy rzadkiej
    rows = ratings['User-ID'].map(user_id_to_idx).astype(int)
    cols = ratings['ISBN'].map(isbn_to_idx).astype(int)
    data = ratings['Book-Rating']

    ratings_sparse = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(isbn_ids)))

    # Model KNN na macierzy rzadkiej
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
    model_knn.fit(ratings_sparse)
    
    print(f"Rekomendacje dla użytkownika ID {user_id}:")
    if user_id not in user_id_to_idx:  # Sprawdzenie, czy ID użytkownika istnieje
        print("Nie znaleziono użytkownika.")
        return
    
    user_idx = user_id_to_idx[user_id]
    distances, indices = model_knn.kneighbors(ratings_sparse[user_idx], n_neighbors=n_recommendations+1)
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if i == 1: continue  # pomijamy pierwszy, bo to zapytanie użytkownika
        isbn = list(isbn_to_idx.keys())[list(isbn_to_idx.values()).index(idx)]
        if isbn in books['ISBN'].values:  # Sprawdzenie, czy ISBN istnieje w DataFrame books
            book_title = books.loc[books['ISBN'] == isbn, 'Book-Title'].iloc[0]
            print(f"{i}: ISBN: {isbn}, tytuł: {book_title}, Dystans: {dist:.4f}")
        else:
            print(f"{i}: ISBN: {isbn} nie znaleziono w bazie książek.")

# Przykładowe wywołanie funkcji
user_id_example = 276704  # Przykładowy ID użytkownika
recommend_books(user_id_example, 5)