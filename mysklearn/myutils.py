import numpy as np


def train_test_split(table, test_size, rand_seed=None):
    """Splits the table into training and test sets

    Args:
        table (MyPyTable): dataset to split
        test_size (int): number of rows for the test set
        rand_seed (int): the random seed

    Returns:
        tuple: (train_data, test_data)
    """
    if rand_seed is not None:
        np.random.seed(rand_seed)

    indices = list(range(len(table.data)))
    np.random.shuffle(indices)

    test_indices = indices[:test_size]
    train_indices = indices[test_size:]

    test_data = [table.data[i] for i in test_indices]
    train_data = [table.data[i] for i in train_indices]

    return train_data, test_data


# Define the DOE ratings discretizer function
def doe_mpg_discretizer(mpg_value):
    """doe mpg categorization

    Args:
        mpg_value (int):

    Returns:
        int :category
    """
    if mpg_value >= 45:
        return 10
    elif 37 <= mpg_value <= 44:
        return 9
    elif 31 <= mpg_value <= 36:
        return 8
    elif 27 <= mpg_value <= 30:
        return 7
    elif 24 <= mpg_value <= 26:
        return 6
    elif 20 <= mpg_value <= 23:
        return 5
    elif 17 <= mpg_value <= 19:
        return 4
    elif 15 <= mpg_value <= 16:
        return 3
    elif mpg_value == 14:
        return 2
    elif mpg_value <= 13:
        return 1


def normalize_data(data, min_values, max_values):
    """Normalizes data for KNN Classifier

    Args:
        data (list): data being normalized
        min_values (int): _description_
        max_values (int): _description_

    Returns:
        _type_: _description_
    """
    normalized_data = []
    for row in data:
        normalized_row = []
        for i in range(len(row)):
            if max_values[i] - min_values[i] != 0:
                normalized_value = (row[i] - min_values[i]) / (
                    max_values[i] - min_values[i]
                )
            else:
                normalized_value = 0  # Avoid division by zero if min and max are equal
            normalized_row.append(normalized_value)
        normalized_data.append(normalized_row)
    return normalized_data


def calculate_winners(table):
    """Calculates the winners of the nba dataset

    Args:
        table (MyPyTable): NBA dataset

    Returns:
        winners (list(ints)): returns winners of match, if team1 wins = 1, if team2 wins = 0
    """
    col1 = "score1"
    col2 = "score2"

    if col1 not in table.column_names or col2 not in table.column_names:
        ValueError("There are no scores in this dataset")

    col1_index = table.column_names.index(col1)
    col2_index = table.column_names.index(col2)

    winners = []

    for row in table.data:
        score1 = row[col1_index]
        score2 = row[col2_index]

        # Error check to see if any scores are empty
        if score1 is None or score2 is None:
            continue

        if score1 > score2:
            winners.append(1)
        else:
            winners.append(0)

    return winners


def preprocess_data(X, bins=5):
    """Preprocesses the dataset by discretizing numeric attributes.

    Args:
        X (list of list of obj): The dataset to preprocess.
        bins (int): The number of bins to create for discretization.

    Returns:
        list of list of str: The preprocessed dataset with categorical values.
    """
    X_transposed = list(zip(*X))
    X_discretized = []
    for col in X_transposed:
        try:
            # Attempt to convert the column to floats
            col = [float(value) for value in col]
            discretized_col = discretize_column(col, bins)
            X_discretized.append(discretized_col)
        except ValueError:
            # If conversion fails, keep the column as is
            X_discretized.append(col)
    return list(zip(*X_discretized))


def discretize_column(column, bins):
    """Discretizes a numeric column into categorical bins.

    Args:
        column (list of numeric): The numeric column to discretize.
        bins (int): The number of bins to create.

    Returns:
        list of str: The discretized column with categorical values.
    """
    min_val = min(column)
    max_val = max(column)
    bin_width = (max_val - min_val) / bins
    bin_edges = [min_val + i * bin_width for i in range(bins + 1)]

    discretized_column = []
    for value in column:
        for i in range(bins):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                discretized_column.append(f"bin_{i}")
                break
        else:
            discretized_column.append(f"bin_{bins - 1}")

    return discretized_column


from mysklearn.mypytable import MyPyTable


def filter_by_season_range(table, start_season, end_season):
    """Filters the dataset to only include instances within the specified range of seasons.

    Args:
        table (MyPyTable): The input dataset.
        start_season (int): The starting season (inclusive).
        end_season (int): The ending season (inclusive).

    Returns:
        MyPyTable: A new MyPyTable containing only the instances within the specified range of seasons.
    """
    # Find the index of the "season" column
    season_col_index = table.column_names.index("season")

    # Filter the data to include only rows within the specified range of seasons
    filtered_data = [
        row for row in table.data if start_season <= row[season_col_index] <= end_season
    ]

    # Return a new MyPyTable with the filtered data
    return MyPyTable(column_names=table.column_names, data=filtered_data)
