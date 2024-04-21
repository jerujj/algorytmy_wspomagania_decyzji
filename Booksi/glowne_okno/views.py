from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from .models import Book, Rating
# from .program.src.martynka import polecane_ksiazki
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import json
import csv



def get_ratings():
    # Wczytanie danych z pliku CSV
    ratings = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\Ratings.csv', usecols=['User-ID', 'ISBN', 'Book-Rating'], dtype={'User-ID': 'str', 'ISBN': 'str', 'Book-Rating': 'str'})
    
    # Zmiana nazw kolumn
    ratings.rename(columns={'User-ID': 'User_ID', 'Book-Rating': 'Book_Rating'}, inplace=True)
    return ratings

def get_books():
    # Wczytanie danych z pliku CSV
    books = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\Books.csv', usecols=['ISBN', 'Book-Title','Image-URL-L'], dtype={'ISBN': 'str', 'Book-Title': 'str', 'Image-URL-M': 'str'})
  
    # Zmiana nazw kolumn
    books.rename(columns={'Book-Title': 'Book_Title', 'Image-URL-L': 'Image_URL_L'}, inplace=True)
    return books

def rekomendacja():
   # Załadowanie danych
   books = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\Books.csv', usecols=['ISBN', 'Book-Title','Image-URL-L'], dtype={'ISBN': 'str', 'Book-Title': 'str', 'Image-URL-M': 'str'})
   ratings = pd.read_csv(r'glowne_okno\program\resources\book_recommendation_dataset\Ratings.csv', usecols=['User-ID', 'ISBN', 'Book-Rating'], dtype={'User-ID': 'int', 'ISBN': 'str', 'Book-Rating': 'float'})

   # books = list(Book.objects.all().values('ISBN', 'Book-Title', 'Image-URL-L'))
   # ratings = list(Rating.objects.all().values('User-ID', 'ISBN', 'Book-Rating'))

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
   
   return(user_id_to_idx,ratings_sparse,isbn_to_idx,books,model_knn)


def recommend_books(user_id, model, user_id_to_idx ,ratings_sparse,isbn_to_idx, books, n_recommendations=5):
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
    recommendations = []
    print(f"Rekomendacje dla użytkownika ID {user_id}:")
    if user_id not in user_id_to_idx:  # Sprawdzenie, czy ID użytkownika istnieje
        print("Nie znaleziono użytkownika.")
        return
    
    user_idx = user_id_to_idx[user_id]
    distances, indices = model.kneighbors(ratings_sparse[user_idx], n_neighbors=n_recommendations+1)
    for i, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        if i == 1: continue  # Skip the user's own entry
        isbn = list(isbn_to_idx.keys())[list(isbn_to_idx.values()).index(idx)]
        book_info = books.loc[books['ISBN'] == isbn]
        if not book_info.empty:
            book_title = book_info['Book-Title'].iloc[0]
            book_image_url = book_info['Image-URL-L'].iloc[0]  # Assuming the image URL column is named 'Image-URL-M'
            recommendations.append({'title': book_title, 'ISBN': isbn, 'distance': dist, 'image': book_image_url})
    return recommendations

def poka(elem):
    print(elem)

# Create your views here.
def index(request):
   user_id_to_idx,ratings_sparse,isbn_to_idx,books,model = rekomendacja()
   user_id = request.session.get('user_id')
   recommendations=recommend_books(user_id, model, user_id_to_idx ,ratings_sparse,isbn_to_idx, books,5)
   location = request.session.get('location')
   age = request.session.get('age')
   parts = location.split(', ')

    
   ratings = get_ratings()
   books = get_books()
   books = books.set_index('ISBN')['Book_Title'].to_dict()

   user = str(user_id)
   filtered_data = ratings[ratings['User_ID'] == user]
   count = filtered_data.shape[0]
   filtered_data.loc[:, 'ISBN'] = filtered_data['ISBN'].astype(str)
   filtered_data['Book_Title'] = filtered_data['ISBN'].map(books)
    
   filtered_data_json = filtered_data.to_dict(orient='records')  # Konwertuje DataFrame do listy słowników
   
   poka(filtered_data_json)
   
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

