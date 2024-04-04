import pandas as pd
import os

def check_data_types_and_missing_values_to_excel(folder_path, output_folder='resources/GeneralDataExploration', output_file='data_check_results.xlsx'):
    """
    Creating summary for csv files about column datatypes and missing values.

    Inspecting every csv file in the provided folder for datatype (if it's consistent and if it is, then what's the datatype), and for missing values, if there are any, and if there are, how many of them and what percentage of entire dataset do they make. The results are stored as xlsx file for easier readibility in the 'resources/GeneralDataExploration' directory.

    Parameters
    ----------
    folder_path: string
        indicating the path to the folder with csv files to inspect

    output_fodler: string
        directory for saving the output file, set default as 'resources/GeneralDataExploration'.

    output_file: string
        name of the output file, set default as 'data_check_results.xlsx'.

    Returns
    -------
    output_file
        excel file with description of the provided csv files for datatypes and missing values in particular columns.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Directory '{output_folder}' was created.")
    
    output_path = os.path.join(output_folder, output_file)

    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(folder_path, file_name)
                df = pd.read_csv(file_path)

                rows_list = []
                for column in df.columns:
                    data_types = set(df[column].map(type))
                    if len(data_types) > 1:
                        column_type = 'mixed'
                    else:
                        column_type = list(data_types)[0].__name__

                    missing_values_count = df[column].isnull().sum()
                    if column == "Year-Of-Publication":
                        zero_year_count = (df[column] == 0).sum()
                        missing_values_count += zero_year_count
                    total_values = len(df[column])
                    missing_values_percentage = (missing_values_count / total_values) * 100

                    rows_list.append({
                        'Column': column,
                        'Type': column_type,
                        'Missing Values': missing_values_count,
                        'Percentage of Missing Data': missing_values_percentage
                    })

                results_df = pd.DataFrame(rows_list)
                results_df.to_excel(writer, sheet_name=file_name[:-4], index=False)

    print(f'Results saved to {output_path}')

