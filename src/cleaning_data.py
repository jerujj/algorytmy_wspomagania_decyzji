import pandas as pd

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

# Example usage
ratings_df = pd.read_csv('resources/archive/ratings.csv')  # Load your ratings data
cleaned_ratings = clean_data(ratings_df, 10, 10)  # Assuming 'x' is 5 for both books and users
print(f"N of original ratings: {len(ratings_df)}, n of cleaned ratings: {len(cleaned_ratings)}")