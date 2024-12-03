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
