import pandas as pd
import numpy as np
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns

def shapiro_test(dataset, exclude_columns=None):
    """
    Performs the Shapiro-Wilk test for normality on all columns not in the exclude list.

    :param dataset: pandas DataFrame
    :param exclude_columns: Optional, list of column names to exclude from the test. Defaults to None.
    :return: pandas DataFrame with test results (columns: 'column', 'p_value', 'normality').
    """
    if exclude_columns is None:
        exclude_columns = []

    # List to store test results
    results = []

    # Iterate through columns, excluding those in the mask
    for column in dataset.columns:
        if column not in exclude_columns and pd.api.types.is_numeric_dtype(dataset[column]):
            # Perform Shapiro-Wilk test
            stat, p_value = shapiro(dataset[column])
            # Append results: column name, p-value, normality (True/False)
            results.append({
                'column': column,
                'p_value': p_value,
                'normality': p_value > 0.05  # True if the distribution is normal
            })

    # Return results as a DataFrame
    return pd.DataFrame(results)

def skewness_correction(dataset, shapiro_results, exclude_columns=None, visualize=False, silent=False):
    """
    Automatically corrects skewness for columns identified as non-normal in the Shapiro-Wilk test.

    :param dataset: pandas DataFrame to transform.
    :param shapiro_results: Results from the Shapiro-Wilk test.
    :param exclude_columns: Optional, list of column names to exclude from transformation. Defaults to None.
    :param visualize: Optional, if set to `True`, function will visualize the distribution of each column before and after transformation.
    :param silent: Optional, if set to `True`, suppresses output logs.
    :return: pandas DataFrame with skewness-corrected columns.
    """
    if exclude_columns is None:
        exclude_columns = []
    
    # Work on a copy of the dataset to avoid modifying the original
    dataset_copy = dataset.copy()

    # Get columns identified as non-normal
    non_normal_columns = shapiro_results[shapiro_results['normality'] == False]['column']

    for column in non_normal_columns:
        if column not in exclude_columns:
            skew_value = dataset_copy[column].skew()
            if skew_value < -0.5:  # Left-skewed
                if not silent:
                    print(f"Applying reflection and log transformation to left-skewed column: {column} (skew={skew_value:.2f})")
                if visualize:
                    visualize_distribution(dataset_copy, column, stage="Before Correction")
                max_value = dataset_copy[column].max()
                dataset_copy[column] = np.log1p(max_value - dataset_copy[column] + 1e-6)
                if visualize:
                    visualize_distribution(dataset_copy, column, stage="After Correction")
            elif skew_value > 0.5:  # Right-skewed
                if not silent:
                    print(f"Applying log transformation to right-skewed column: {column} (skew={skew_value:.2f})")
                if visualize:
                    visualize_distribution(dataset_copy, column, stage="Before Correction")
                dataset_copy[column] = np.log1p(dataset_copy[column] + 1e-6)
                if visualize:
                    visualize_distribution(dataset_copy, column, stage="After Correction")
            else:
                if not silent:
                    print(f"No transformation applied to column: {column} (skew={skew_value:.2f}) - already near symmetric")

    return dataset_copy

def visualize_distribution(dataset, column, stage):
    """
    Visualizes the distribution of a specific column using a histogram and KDE.

    :param dataset: pandas DataFrame
    :param column: Column to visualize
    :param stage: Stage of processing (e.g., "Before Correction" or "After Correction")
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(dataset[column], kde=True, bins=30)
    plt.title(f"{column} Distribution - {stage}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()