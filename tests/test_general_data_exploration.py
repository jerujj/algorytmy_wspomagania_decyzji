import unittest
import sys
import os
import pandas as pd
sys.path.insert(1, "src")
from general_data_exploration import *

class TestCheckDataTypesAndMissingValues(unittest.TestCase):
    def setUp(self):
        self.resources_path = "resources/Pomiary/Zdrowe"

    def test_load_predict_plot(self):
        # Arrange
        folder_path = 'resources/book_recommendation_dataset'
        output_folder = 'resources/GeneralDataExploration'
        output_file = 'data_check_results.xlsx'
        file_path = os.path.join(output_folder, output_file)

        # Act
        check_data_types_and_missing_values_to_excel(folder_path)

        # Assert
        self.assertTrue(os.path.exists(file_path), f"The file {file_path} should exist")
    
    def tearDown(self):
        pass

class TestRatingsNumberDistribution(unittest.TestCase):
    def setUp(self):
        self.resources_path = "resources/Pomiary/Zdrowe"

    def test_load_predict_plot(self):
        # Arrange
        ratings = pd.read_csv(r'resources\book_recommendation_dataset\Ratings.csv', 
                            usecols=['User-ID', 'ISBN', 'Book-Rating'],
                            dtype={'User-ID': 'int', 'ISBN': 'str', 'Book-Rating': 'float'})

        # Act
        plot_rating_distributions(ratings)

        # Assert
        
    
    def tearDown(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestSuite()

    #suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCheckDataTypesAndMissingValues))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestRatingsNumberDistribution))

    runner = unittest.TextTestRunner()
    runner.run(suite)