import pandas as pd
import matplotlib.pyplot as plt
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

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
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

def plot_rating_distributions(ratings):
    """
    Creating distribution plot for users and books against number of ratings.

    Using data from "Ratings.csv" file to create a plot with 2 subplots, that will present the distribution of ratings over books and users in the database. 

    Parameters
    ----------
    ratings: `DataFrame`
        DataFrame created from "Ratings.csv" file with proper dtype definitions.
    """
    book_counts = ratings['ISBN'].value_counts()

    user_counts = ratings['User-ID'].value_counts()

    plt.figure(figsize=(14, 7))

    ax1 = plt.subplot(1, 2, 1)
    book_patches = ax1.hist(book_counts, bins=200, edgecolor='black', alpha=0.7, log=True)
    ax1.set_title('Distribution of Ratings per Book (Log Scale)')
    ax1.set_xlabel('Number of Ratings')
    ax1.set_ylabel('Number of Books (Log Scale)')

    for patch in book_patches[2]:
        bin_x = patch.get_x() + patch.get_width() / 2
        bin_y = patch.get_height()
        if bin_y > 0: 
            ax1.annotate(f'{int(bin_y)}',
                         xy=(bin_x, bin_y), 
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom')

    ax2 = plt.subplot(1, 2, 2)
    user_patches = ax2.hist(user_counts, bins=200, edgecolor='black', alpha=0.7, log=True)
    ax2.set_title('Distribution of Ratings per User (Log Scale)')
    ax2.set_xlabel('Number of Ratings')
    ax2.set_ylabel('Number of Users (Log Scale)')

    for patch in user_patches[2]:
        bin_x = patch.get_x() + patch.get_width() / 2
        bin_y = patch.get_height()
        if bin_y > 0: 
            ax2.annotate(f'{int(bin_y)}', 
                         xy=(bin_x, bin_y), 
                         xytext=(0, 5),
                         textcoords="offset points",
                         ha='center', va='bottom')

    plt.tight_layout()
    plt.show()