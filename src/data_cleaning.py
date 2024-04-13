import pandas as pd

def clean_csv_files(min_reviews):
    """
    Cleaning chosen csv files for minimum ratings number.

    Inspecting provided csv files for empty columns and rows, removing unnecessary data and then removing Books and Users that have less ratings than the provided min_reviews.

    Parameters
    ----------
    min_reviews: `int`
        number of ratings that Book or User need to have to stay in the dataset for model training.

    Returns
    -------
    books, ratings, users
        DataFrames for books, ratings and users respectively, with the minimum ratings number rule applied
    
    books_clean, ratings_clean, users_clean
        DataFrames for books, ratings and users respectively, but only with columns and rows cleaned, without deleting Users and Books with too low number of ratings
    """
    books = pd.read_csv(r'resources\book_recommendation_dataset\Books.csv', 
                        usecols=['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-S', 'Image-URL-M', 'Image-URL-L'],
                        dtype={'ISBN': 'str', 'Book-Title': 'str', 'Book-Author': 'str', 
                               'Year-Of-Publication': 'str', 'Publisher': 'str', 'Image-URL-S': 'str', 'Image-URL-M': 'str', 'Image-URL-L': 'str'})
    ratings = pd.read_csv(r'resources\book_recommendation_dataset\Ratings.csv', 
                          usecols=['User-ID', 'ISBN', 'Book-Rating'],
                          dtype={'User-ID': 'int', 'ISBN': 'str', 'Book-Rating': 'float'})
    users = pd.read_csv(r'resources\book_recommendation_dataset\Users.csv', 
                        usecols=['User-ID', 'Location', 'Age'],
                        dtype={'User-ID': 'int', 'Location': 'str', 'Age': 'float'})

    users.drop(columns='Age', inplace=True)

    books.dropna(subset=['Year-Of-Publication', 'Book-Author', 'Publisher'], inplace=True)

    books.drop(columns=['Image-URL-S', 'Image-URL-L'], inplace=True)

    books_clean = books
    ratings_clean = ratings
    users_clean = users

    while True:
        book_counts = ratings['ISBN'].value_counts()
        user_counts = ratings['User-ID'].value_counts()

        ratings = ratings[ratings['ISBN'].isin(book_counts[book_counts >= min_reviews].index)]
        ratings = ratings[ratings['User-ID'].isin(user_counts[user_counts >= min_reviews].index)]

        new_book_counts = ratings['ISBN'].value_counts()
        new_user_counts = ratings['User-ID'].value_counts()
        
        if (new_book_counts.index.equals(book_counts.index) and new_user_counts.index.equals(user_counts.index)):
            break 

    books = books[books['ISBN'].isin(ratings['ISBN'])]
    users = users[users['User-ID'].isin(ratings['User-ID'])]

    return books, ratings, users, books_clean, ratings_clean, users_clean
