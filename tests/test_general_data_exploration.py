import unittest
import sys
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

if __name__ == '__main__':
    suite = unittest.TestSuite()

    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestCheckDataTypesAndMissingValues))

    runner = unittest.TextTestRunner()
    runner.run(suite)