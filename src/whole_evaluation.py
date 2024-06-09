import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

evaluation_results_path = 'src/ModelsKNN/Research2406061853KNN/knn_model_evaluation_results.csv'
evaluation_results = pd.read_csv(evaluation_results_path)

# Normalize points_ratio to range 0 to 1
scaler = MinMaxScaler()
evaluation_results['points_ratio'] = scaler.fit_transform(evaluation_results[['points_ratio']])

# Load and merge all hyperparameter files
merged_data = evaluation_results.copy()

# Here we simulate the process of merging all hyperparameter files by replicating the example file
# In practice, you would loop through all 160 files, loading them by their respective filenames
model_versions = merged_data['model_version'].unique()

# Initialize a DataFrame to store the merged data
all_hyperparameters = pd.DataFrame()

for model_version in model_versions:
    hyperparameters_path = f'src/ModelsKNN/Research2406061853KNN/knn_model_hyperparameters_{model_version}.csv'
    hyperparameters = pd.read_csv(hyperparameters_path)
    hyperparameters['model_version'] = model_version
    all_hyperparameters = pd.concat([all_hyperparameters, hyperparameters], ignore_index=True)

all_hyperparameters.rename(columns={'n_neighbors': 'n_neighbors_model'}, inplace=True)

# Merge the evaluation results with their corresponding hyperparameters
merged_data = pd.merge(merged_data, all_hyperparameters, on=['model_version'], how='right')
reordered_data = merged_data[['n_neighbors_model', 'metric', 'algorithm', 'leaf_size', 'n_recommendations', 'n_neighbors', 'points_ratio']]

# Save the merged data to a new CSV file
reordered_data.to_csv('src/ModelsKNN/Research2406061853KNN/whole.csv', index=False)

################################## Plotting

# Set the style for the plots
sns.set(style="whitegrid")

# Generate box plots for each hyperparameter against the points_ratio
def save_custom_box_plot(data, x_column, y_column, x_label, y_label, axis_fontsize, tick_fontsize, boxprops, whiskerprops, capprops, flierprops, medianprops, filename='box_plot.png'):
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=x_column, y=y_column, data=data, boxprops=boxprops, whiskerprops=whiskerprops, capprops=capprops, flierprops=flierprops, medianprops=medianprops)
    plt.xlabel(x_label, fontsize=axis_fontsize, fontname='Palatino Linotype')
    plt.ylabel(y_label, fontsize=axis_fontsize, fontname='Palatino Linotype')
    
    plt.xticks(fontsize=tick_fontsize, fontname='Palatino Linotype')
    plt.yticks(fontsize=tick_fontsize, fontname='Palatino Linotype')
    
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

# Example box properties for customization
boxprops = dict(linestyle='--', linewidth=2, color='blue', facecolor='lightblue', alpha=0.7)
whiskerprops = dict(linestyle='-', linewidth=2, color='green')
capprops = dict(linestyle='-', linewidth=2, color='red')
flierprops = dict(marker='o', markersize=5, linestyle='none', markeredgecolor='r', alpha=0.5)
medianprops = dict(linestyle='-', linewidth=2.5, color='orange')
axis_fontsize = 30
tick_fontsize = 25

# Generate and save box plots for each hyperparameter against the points_ratio
save_custom_box_plot(reordered_data, 'n_neighbors_model', 'points_ratio', 'k-sąsiadów w procesie treningu', 'Wynik ewaluacji', axis_fontsize, tick_fontsize, boxprops, whiskerprops, capprops, flierprops, medianprops, filename='src/ModelsKNN/Research2406061853KNN/box_plot_n_neighbors_model.png')
save_custom_box_plot(reordered_data, 'metric', 'points_ratio', 'Sposób obliczania odległości', 'Wynik ewaluacji', axis_fontsize, tick_fontsize, boxprops, whiskerprops, capprops, flierprops, medianprops, filename='src/ModelsKNN/Research2406061853KNN/box_plot_metric.png')
save_custom_box_plot(reordered_data, 'algorithm', 'points_ratio', 'Sposób szukania najbliższych sąsiadów', 'Wynik ewaluacji', axis_fontsize, tick_fontsize, boxprops, whiskerprops, capprops, flierprops, medianprops, filename='src/ModelsKNN/Research2406061853KNN/box_plot_algorithm.png')
save_custom_box_plot(reordered_data, 'leaf_size', 'points_ratio', 'Wielkość liścia', 'Wynik ewaluacji', axis_fontsize, tick_fontsize, boxprops, whiskerprops, capprops, flierprops, medianprops, filename='src/ModelsKNN/Research2406061853KNN/box_plot_leaf_size.png')
save_custom_box_plot(reordered_data, 'n_recommendations', 'points_ratio', 'Liczba zwracanych rekomendacji', 'Wynik ewaluacji', axis_fontsize, tick_fontsize, boxprops, whiskerprops, capprops, flierprops, medianprops, filename='src/ModelsKNN/Research2406061853KNN/box_plot_n_recommendations.png')
save_custom_box_plot(reordered_data, 'n_neighbors', 'points_ratio', 'k-sąsiadów w procesie filtracji', 'Wynik ewaluacji', axis_fontsize, tick_fontsize, boxprops, whiskerprops, capprops, flierprops, medianprops, filename='src/ModelsKNN/Research2406061853KNN/box_plot_n_neighbors.png')