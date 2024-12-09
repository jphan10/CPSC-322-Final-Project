import numpy as np
import random


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


def manual_balance_dataset(X, y, method="undersample"):
    """
    Balances the dataset using the specified method without external libraries.

    Args:
        X (list of list): Feature data.
        y (list): Target labels.
        method (str): Balancing method ('undersample' or 'oversample').

    Returns:
        tuple: Balanced X and y.
    """
    # Combine features and labels into a single dataset
    data = [(x, label) for x, label in zip(X, y)]

    # Separate by class
    class_0 = [row for row in data if row[1] == 0]
    class_1 = [row for row in data if row[1] == 1]

    # Balance the dataset
    if method == "undersample":
        # Undersample the majority class
        if len(class_0) > len(class_1):
            class_0 = random.sample(class_0, len(class_1))
        else:
            class_1 = random.sample(class_1, len(class_0))
    elif method == "oversample":
        # Oversample the minority class
        if len(class_0) < len(class_1):
            class_0 = class_0 + random.choices(class_0, k=len(class_1) - len(class_0))
        else:
            class_1 = class_1 + random.choices(class_1, k=len(class_0) - len(class_1))
    else:
        raise ValueError("Invalid method. Choose 'undersample' or 'oversample'.")

    # Combine the balanced classes and shuffle
    balanced_data = class_0 + class_1
    random.shuffle(balanced_data)

    # Split back into features and labels
    X_balanced, y_balanced = zip(*balanced_data)
    return list(X_balanced), list(y_balanced)
