"""
Programmer: Jaden Phan
Class: CPSC 322, Fall 2024
Programming Assignment #5
10/27/2024

Description: This program implements some classifier evaluation methods
"""

import numpy as np
import copy
import math
import random  # Using random b/c np.random giving me issues
from collections import defaultdict


def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    """
    # Validate test_size input
    if isinstance(test_size, float):
        test_size = int(math.ceil(test_size * len(X)))

    # Ensure test_size is valid
    if test_size >= len(X) or test_size <= 0:
        raise ValueError(
            "test_size must be a positive integer less than the number of samples"
        )

    # Create deep copies of X and y to avoid modifying the original data
    X_copy = copy.deepcopy(X)
    y_copy = copy.deepcopy(y)

    # Check if random_state is set
    if random_state is not None:
        random.seed(random_state)

    # Shuffle X and y together if shuffle is True
    if shuffle:
        indices = np.arange(len(X_copy))
        random.shuffle(indices)
        X_shuffled = [X_copy[i] for i in indices]
        y_shuffled = [y_copy[i] for i in indices]

        split_index = len(X_copy) - test_size
        X_train, X_test = X_shuffled[:split_index], X_shuffled[split_index:]
        y_train, y_test = y_shuffled[:split_index], y_shuffled[split_index:]

        return X_train, X_test, y_train, y_test

    else:
        # Split the data into training and testing sets
        split_index = len(X_copy) - test_size
        X_train, X_test = X_copy[:split_index], X_copy[split_index:]
        y_train, y_test = y_copy[:split_index], y_copy[split_index:]

        return X_train, X_test, y_train, y_test


def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """

    indices = list(range(len(X)))

    # Check random state
    if random_state is not None:
        random.seed(random_state)

    # Check if shuffle is set
    if shuffle:
        random.shuffle(indices)

    n_samples = len(X)
    n_extras = n_samples % n_splits
    fold_size = n_samples // n_splits

    folds = []
    start_index = 0
    stop_index = 0

    for i in range(n_splits):
        if n_extras > 0:
            increment = fold_size + 1
            n_extras -= 1
        else:
            increment = fold_size

        stop_index = start_index + increment

        test_indices = indices[start_index:stop_index]
        train_indices = indices[:start_index] + indices[stop_index:]

        folds.append((train_indices, test_indices))

        start_index = stop_index

    return folds


# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    # Check random state
    if random_state is not None:
        random.seed(random_state)

    # Shuffle X and y in unison if needed
    indices = list(range(len(X)))
    if shuffle:
        combined = list(zip(X, y, indices))
        random.shuffle(combined)
        X, y, indices = zip(*combined)

    # Create a dictionary to store indices of each class label
    label_to_indices = defaultdict(list)
    for idx, label in zip(indices, y):
        label_to_indices[label].append(idx)

    # Split indices for each class label into folds
    folds = [[] for _ in range(n_splits)]
    for label, indices in label_to_indices.items():
        fold_size = len(indices) // n_splits
        n_extras = len(indices) % n_splits

        start_idx = 0
        for fold_idx in range(n_splits):
            # Adjust fold size for folds with an extra sample
            size = fold_size + 1 if fold_idx < n_extras else fold_size
            folds[fold_idx].extend(indices[start_idx : start_idx + size])
            start_idx += size

    # Create train-test splits based on folds
    stratified_folds = []
    for fold_idx in range(n_splits):
        test_indices = folds[fold_idx]
        train_indices = [
            idx for i, fold in enumerate(folds) if i != fold_idx for idx in fold
        ]
        stratified_folds.append((train_indices, test_indices))

    return stratified_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if n_samples is None:
        n_samples = len(X)

    if random_state is not None:
        random.seed(random_state)

    indices = np.random.choice(len(X), size=n_samples, replace=True)
    X_sample = [X[i] for i in indices]
    X_out_of_bag = [X[i] for i in range(len(X)) if i not in indices]

    if y is not None:
        y_sample = [y[i] for i in indices]
        y_out_of_bag = [y[i] for i in range(len(y)) if i not in indices]
    else:
        y_sample = None
        y_out_of_bag = None

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    # Initialize the confusion matrix with zeros
    matrix = [[0 for _ in labels] for _ in labels]

    # Create a mapping from label to index
    label_to_index = {label: index for index, label in enumerate(labels)}

    # Populate the confusion matrix
    for true, pred in zip(y_true, y_pred):
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        matrix[true_index][pred_index] += 1

    return matrix


def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    correct_count = 0
    total_count = len(y_true)

    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct_count += 1

    if normalize:
        score = correct_count / total_count
    else:
        score = correct_count

    return score


def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if pos_label is None:
        pos_label = sorted(set(y_true))[0]

    true_positive = 0
    false_positive = 0

    for true, pred in zip(y_true, y_pred):
        if pred == pos_label:
            if true == pos_label:
                true_positive += 1
            else:
                false_positive += 1

    if true_positive + false_positive == 0:
        return 0.0

    return true_positive / (true_positive + false_positive)


def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if pos_label is None:
        pos_label = sorted(set(y_true))[0]

    true_positive = 0
    false_negative = 0

    for true, pred in zip(y_true, y_pred):
        if true == pos_label:
            if pred == pos_label:
                true_positive += 1
            else:
                false_negative += 1

    if true_positive + false_negative == 0:
        return 0.0

    return true_positive / (true_positive + false_negative)


def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels, pos_label)
    recall = binary_recall_score(y_true, y_pred, labels, pos_label)

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def evaluate_classifier(classifier, X_train, y_train, X_test, y_test, classifier_name):
    """
    Evaluates a classifier using separate training and testing datasets.

    Args:
        classifier (obj): The classifier to evaluate (must implement fit and predict).
        X_train (list of list): The training feature set.
        y_train (list): The training labels.
        X_test (list of list): The testing feature set.
        y_test (list): The testing labels.
        classifier_name (str): The name of the classifier being evaluated.

    Returns:
        None: Prints evaluation results.
    """
    print(f"\n=== {classifier_name} ===")

    # Determine the unique labels in the dataset
    unique_labels = list(set(y_train + y_test))

    # Train the classifier on the training set
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Calculate evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    error_rate = 1 - acc
    conf_matrix = confusion_matrix(y_test, y_pred, labels=unique_labels)
    precision = binary_precision_score(
        y_test, y_pred, labels=unique_labels, pos_label=unique_labels[0]
    )
    recall = binary_recall_score(
        y_test, y_pred, labels=unique_labels, pos_label=unique_labels[0]
    )
    f1 = binary_f1_score(
        y_test, y_pred, labels=unique_labels, pos_label=unique_labels[0]
    )

    # Print the evaluation metrics
    print(f"Evaluation results for {classifier_name}:")
    print(f"Accuracy: {acc:.2f}")
    print(f"Error Rate: {error_rate:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}")
