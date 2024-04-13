import unittest
import sys
import pandas as pd 
sys.path.insert(1, "src")
from data_cleaning import *

class TestCleaningData(unittest.TestCase):
    def setUp(self):
        pass

    def test_load_predict_plot(self):
        # Arrange
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
    
        # Act
        new_books, new_ratings, new_users, _, _, _ = clean_csv_files(10)

        # Assert
        print("Minimum Books and Users counts")
        print(f"Books: {ratings['ISBN'].value_counts().min()}")
        print(f"Users: {ratings['User-ID'].value_counts().min()}")
        print(" ")
        print("Checking the size of the old and new datasets")
        print(f"  Books length: old ->{len(books)},   new ->{len(new_books)} ({len(new_books)/len(books)*100:.2f}%)")
        print(f"Ratings length: old ->{len(ratings)}, new ->{len(new_ratings)} ({len(new_ratings)/len(ratings)*100:.2f}%)")
        print(f"  Users length: old ->{len(users)},   new ->{len(new_users)} ({len(new_users)/len(users)*100:.2f}%)")        

        self.assertFalse('Age' in new_users.columns)
        self.assertFalse('Image-URL-S' in new_books.columns)
        self.assertFalse('Image-URL-L' in new_books.columns)

        self.assertTrue(new_books.isna().sum().max() == 0)
        pass
    
    def tearDown(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCleaningData))

    runner = unittest.TextTestRunner()
    runner.run(suite)