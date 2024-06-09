from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import Book, Rating
# from .program.src.martynka import polecane_ksiazki
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os
import joblib
import numpy as np
from scipy.sparse import load_npz



def get_ratings():
    # Wczytanie danych z pliku CSV
    ratings = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\ratings2.csv')
    
    # # Zmiana nazw kolumn
    # ratings.rename(columns={'User-ID': 'User_ID', 'Book-Rating': 'Book_Rating'}, inplace=True)
    return ratings

def get_books():
    # Wczytanie danych z pliku CSV
    books = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\books2.csv')
  
    # # Zmiana nazw kolumn
    # books.rename(columns={'Book-Title': 'Book_Title', 'Image-URL-L': 'Image_URL_L'}, inplace=True)
    return books

# def rekomendacja():
#    # Załadowanie danych
#    books = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\books2.csv', usecols=['ISBN', 'Book-Title','Image-URL-L'], dtype={'ISBN': 'str', 'Book-Title': 'str', 'Image-URL-M': 'str'})
#    ratings = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\ratings2.csv', usecols=['User-ID', 'ISBN', 'Book-Rating'], dtype={'User-ID': 'int', 'ISBN': 'str', 'Book-Rating': 'float'})

#    # books = list(Book.objects.all().values('ISBN', 'Book-Title', 'Image-URL-L'))
#    # ratings = list(Rating.objects.all().values('User-ID', 'ISBN', 'Book-Rating'))

#    # Przetwarzanie wstępne
#    ## Usuwanie książek i użytkowników z niewielką liczbą ocen
#    min_book_ratings = 10
#    filter_books = ratings['ISBN'].value_counts() > min_book_ratings
#    filter_books = filter_books[filter_books].index.tolist()

#    min_user_ratings = 10
#    filter_users = ratings['User-ID'].value_counts() > min_user_ratings
#    filter_users = filter_users[filter_users].index.tolist()

#    ratings_filtered = ratings[(ratings['ISBN'].isin(filter_books)) & (ratings['User-ID'].isin(filter_users))]

#    # Mapowanie User-ID i ISBN do unikalnych indeksów
#    user_ids = ratings_filtered['User-ID'].unique()
#    isbn_ids = ratings_filtered['ISBN'].unique()

#    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
#    isbn_to_idx = {isbn: idx for idx, isbn in enumerate(isbn_ids)}

#    # Tworzenie macierzy rzadkiej
#    rows = ratings_filtered['User-ID'].map(user_id_to_idx).astype(int)
#    cols = ratings_filtered['ISBN'].map(isbn_to_idx).astype(int)
#    data = ratings_filtered['Book-Rating']

#    ratings_sparse = csr_matrix((data, (rows, cols)), shape=(len(user_ids), len(isbn_ids)))

#    # Model KNN na macierzy rzadkiej
#    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
#    model_knn.fit(ratings_sparse)
   
#    return(user_id_to_idx,ratings_sparse,isbn_to_idx,books,model_knn)


# def recommend_books(user_id, model, user_id_to_idx ,ratings_sparse,isbn_to_idx, books, n_recommendations=5):
#     """
#     Generates and displays a list of recommended books for a given user, 
#     based on a KNN model and the provided book ratings.
    
#     Parameters:
#     ----------
#     user_id `int`: 
#         The unique identifier of the user for whom the recommendations are to be generated.
#     model `NearestNeighbors`: 
#         The trained KNN model used to find the nearest neighbors.
#     n_recommendations `int`: 
#         The number of book recommendations to generate. Defaults to 5.
#     """
#     recommendations = []
#     print(f"Rekomendacje dla użytkownika ID {user_id}:")
#     if user_id not in user_id_to_idx:  # Sprawdzenie, czy ID użytkownika istnieje
#         print("Nie znaleziono użytkownika.")
#         return
    
#     user_idx = user_id_to_idx[user_id]
#     distances, indices = model.kneighbors(ratings_sparse[user_idx], n_neighbors=n_recommendations+1)
#     for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
#         if i == 1: continue  # Skip the user's own entry
#         isbn = list(isbn_to_idx.keys())[list(isbn_to_idx.values()).index(idx)]
#         book_info = books.loc[books['ISBN'] == isbn]
#         if not book_info.empty:
#             book_title = book_info['Book-Title'].iloc[0]
#             book_image_url = book_info['Image-URL-L'].iloc[0]  # Assuming the image URL column is named 'Image-URL-M'
#             recommendations.append({'title': book_title, 'ISBN': isbn, 'distance': dist, 'image': book_image_url})
#     return recommendations



def clean_data(ratings, min_ratings_per_book, min_ratings_per_user):
    # Remove rows with any missing values in 'book_id', 'user_id', or 'rating'
    ratings = ratings.dropna(subset=['book_id', 'user_id', 'rating'])
    
    while True:
        # Calculate the number of ratings per book and per user
        user_counts = ratings['user_id'].value_counts()
        book_counts = ratings['book_id'].value_counts()

        # Filter out users and books with ratings less than the minimum threshold
        ratings = ratings[ratings['user_id'].isin(user_counts[user_counts >= min_ratings_per_user].index)]
        ratings = ratings[ratings['book_id'].isin(book_counts[book_counts >= min_ratings_per_book].index)]
        
        # Check if further removals are needed
        if len(user_counts[user_counts < min_ratings_per_user]) == 0 and len(book_counts[book_counts < min_ratings_per_book]) == 0:
            break

        user_counts = ratings['user_id'].value_counts()
        book_counts = ratings['book_id'].value_counts()

    return ratings

def load_model_and_data(model_path):
    model_filename = os.path.join(model_path, f'knn_model_37.joblib')
    matrix_filename = os.path.join(model_path, f'user_item_matrix_37.npz')
    ratings = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\ratings2.csv')
    
    model = joblib.load(model_filename)
    user_item_matrix = load_npz(matrix_filename)
    
    cleaned_ratings = clean_data(ratings, 10, 10)
    
    return model, user_item_matrix, cleaned_ratings

def create_user_vector(user_ratings, num_items):
    user_vector = np.zeros(num_items)
    for item, rating in user_ratings.items():
        if item > 9998:
            item = 9998
        user_vector[item] = rating
    return user_vector

def predict_for_user(model, user_item_matrix, new_user_vector, n_neighbors, n_recommendations):
    nowe , indices = model.kneighbors([new_user_vector], n_neighbors=n_neighbors)
    similar_users = indices.flatten()
    nowe = nowe.flatten()
    
    similar_users_matrix = user_item_matrix[similar_users].toarray()
    mean_ratings = similar_users_matrix.mean(axis=0)
    
    unrated_items_mask = new_user_vector == 0
    mean_ratings[~unrated_items_mask] = -np.inf
    recommended_items = np.argsort(mean_ratings)[-n_recommendations:][::-1]
    
    books = get_books()
    recommendations = []
    
    for item, elem in enumerate( recommended_items):
        recommendations.append({'title': books['title'][elem] , 'ISBN': books['ISBN'][elem], 'distance': nowe[item] , 'image': books['image_url'][elem]})

    return recommendations




def poka(elem):
    print(' ')
    print(elem)
    print(' ')

# Create your views here.
def index(request):
    model, user_item_matrix, ratings = load_model_and_data(r'glowne_okno\program\resources\book_recommendation_dataset')
    num_items = user_item_matrix.shape[1]
    user_id = request.session.get('user_id')
    
    user_ratings = ratings[ratings['user_id'] == user_id]
    user_ratings_dict = user_ratings.set_index('book_id')['rating'].to_dict()
    user_vector = create_user_vector(user_ratings_dict,num_items)
    recommendations = predict_for_user(model, user_item_matrix, user_vector, 20, 7)
    
    # user_id_to_idx,ratings_sparse,isbn_to_idx,books,model = rekomendacja()
    # recommendations=recommend_books(user_id, model, user_id_to_idx ,ratings_sparse,isbn_to_idx, books,5)
    location = request.session.get('location')
    age = request.session.get('age')
    parts = location.split(', ')

    
    ratings = get_ratings()
    books = get_books()
    # book_id,goodreads_book_id,best_book_id,work_id,books_count,ISBN,isbn13,authors,original_publication_year,original_title,title,language_code,average_rating,ratings_count,work_ratings_count,work_text_reviews_count,ratings_1,ratings_2,ratings_3,ratings_4,ratings_5,image_url,small_image_url
    authors_dict = books.set_index('book_id')['authors'].to_dict()
    books = books.set_index('book_id')['original_title'].to_dict()
    user = str(user_id)
    # user_id,book_id,rating
    filtered_data = ratings[ratings['user_id'] == int(user)]
    count = filtered_data.shape[0]
    
    # # poka(count)
    # filtered_data.loc[:, 'book_id'] = filtered_data['book_id'].astype(str)
    # filtered_data['original_title'] = filtered_data['book_id'].map(books)
    # poka(filtered_data)
    # poka(filtered_data['original_title'])
    # Filtruj dane na podstawie user_id
    filtered_data = ratings[ratings['user_id'] == int(user)].copy()
    count = filtered_data.shape[0]

    # Mapowanie book_id na tytuły książek za pomocą słownika books_dict
    filtered_data['title'] = filtered_data['book_id'].map(books)
    filtered_data['authors'] = filtered_data['book_id'].map(authors_dict)
    poka(filtered_data['authors'])
    filtered_data_json = filtered_data.to_dict(orient='records')  # Konwertuje DataFrame do listy słowników
    
    # poka(filtered_data_json)
    
    context = {
            'user_id': user_id,
            'location': location,
            'age': age,
            'recommendations': recommendations,
            'city': parts[0].title() if len(parts) > 0 else '',
            'state': parts[1].title() if len(parts) > 1 else '',
            'country': parts[2].title() if len(parts) > 2 else '',
            'filtered_ratings': filtered_data_json,  
            'count': count,  
    }
    return render(request, 'glowne_okno/index.html', context)

