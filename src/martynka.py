import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Załadowanie danych
books = pd.read_csv(r'resources\book_recommendation_dataset\Books.csv', usecols=['ISBN', 'Book-Title'], dtype={'ISBN': 'str', 'Book-Title': 'str'})
ratings = pd.read_csv(r'resources\book_recommendation_dataset\Ratings.csv', usecols=['User-ID', 'ISBN', 'Book-Rating'], dtype={'User-ID': 'int', 'ISBN': 'str', 'Book-Rating': 'float'})

# Przetwarzanie wstępne
## Usuwanie książek i użytkowników z niewielką liczbą ocen
min_book_ratings = 10
filter_books = ratings['ISBN'].value_counts() > min_book_ratings
filter_books = filter_books[filter_books].index.tolist()

min_user_ratings = 10
filter_users = ratings['User-ID'].value_counts() > min_user_ratings
filter_users = filter_users[filter_users].index.tolist()

ratings_filtered = ratings[(ratings['ISBN'].isin(filter_books)) & (ratings['User-ID'].isin(filter_users))]

# Mapowanie User-ID i ISBN do unikalnych indeksów
user_ids = ratings_filtered['User-ID'].unique()
isbn_ids = ratings_filtered['ISBN'].unique()

user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
isbn_to_idx = {isbn: idx for idx, isbn in enumerate(isbn_ids)}

# Tworzenie macierzy rzadkiej
rows = ratings_filtered['User-ID'].map(user_id_to_idx).astype(int)
cols = ratings_filtered['ISBN'].map(isbn_to_idx).astype(int)
data = ratings_filtered['Book-Rating']

ratings_sparse = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(isbn_ids)))

# Model KNN na macierzy rzadkiej
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
model_knn.fit(ratings_sparse)

# Funkcja do generowania rekomendacji       
def recommend_books(user_id, model, n_recommendations=5):
    """
    Generates and displays a list of recommended books for a given user, 
    based on a KNN model and the provided book ratings.
    
    Parameters:
    ----------
    user_id `int`: 
        The unique identifier of the user for whom the recommendations are to be generated.
    model `NearestNeighbors`: 
        The trained KNN model used to find the nearest neighbors.
    n_recommendations `int`: 
        The number of book recommendations to generate. Defaults to 5.
    """
    
    print(f"Rekomendacje dla użytkownika ID {user_id}:")
    if user_id not in user_id_to_idx:  # Sprawdzenie, czy ID użytkownika istnieje
        print("Nie znaleziono użytkownika.")
        return
    
    user_idx = user_id_to_idx[user_id]
    distances, indices = model.kneighbors(ratings_sparse[user_idx], n_neighbors=n_recommendations+1)
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
recommend_books(user_id_example, model_knn, 5)