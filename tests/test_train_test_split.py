import unittest
import sys
import pandas as pd 
sys.path.insert(1, "src")
from data_cleaning import *
from train_test_split import *

class TestTrainTestSplit(unittest.TestCase):
    def setUp(self):
        pass

    def test_load_predict_plot(self):
        # Arrange
        _, new_ratings, _, _, _, _ = clean_csv_files(10)
    
        # Act
        train_set, test_set = custom_train_test_split(new_ratings, 0.2)

        # Assert
        print(f"Train set size: {len(train_set)}")
        print(f" Test set size: {len(test_set)}")
        print(" ")
        self.assertEqual(len(train_set)+len(test_set), len(new_ratings), "Train and test sets are not equal to the ratings length!")
        
        train_users = set(train_set['User-ID'])
        test_users = set(test_set['User-ID'])
        self.assertTrue(train_users.intersection(test_users), "Each user should be in both train and test sets if possible.")

        expected_test_set_size = int(np.ceil(0.2 * len(new_ratings)))
        print(f"Expected test set size: {expected_test_set_size}")
        self.assertTrue(abs(len(test_set) - expected_test_set_size) <= 1, f"Test set size is not close to the expected ratio. ({abs(len(test_set) - expected_test_set_size)})")
    
    def tearDown(self):
        pass

if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestTrainTestSplit))

    runner = unittest.TextTestRunner()
    runner.run(suite)