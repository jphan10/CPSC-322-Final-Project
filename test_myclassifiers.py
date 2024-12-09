# pylint: skip-file
import numpy as np
from sklearn.linear_model import LinearRegression
import pytest

from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
from mysklearn.myclassifiers import (
    MySimpleLinearRegressionClassifier,
    MyKNeighborsClassifier,
    MyDummyClassifier,
    MyNaiveBayesClassifier,
    MyDecisionTreeClassifier,
    MyRandomForestClassifier,
)


def discretizer(y):
    return "high" if y >= 100 else "low"


# note: order is actual/received student value, expected/solution
def test_simple_linear_regression_classifier_fit():
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    regressor = MySimpleLinearRegressor()
    lin_reg = MySimpleLinearRegressionClassifier(discretizer, regressor)
    lin_reg.fit(X_train, y_train)

    assert np.isclose(lin_reg.regressor.slope, 1.9249174584304438)
    assert np.isclose(lin_reg.regressor.intercept, 5.211786196055158)

    sk_lin_reg = LinearRegression()
    sk_lin_reg.fit(X_train, y_train)
    assert np.isclose(lin_reg.regressor.slope, sk_lin_reg.coef_[0])
    assert np.isclose(lin_reg.regressor.intercept, sk_lin_reg.intercept_)


def test_simple_linear_regression_classifier_predict():
    np.random.seed(0)
    X_train = [[val] for val in list(range(0, 100))]
    y_train = [row[0] * 2 + np.random.normal(0, 25) for row in X_train]

    regressor = MySimpleLinearRegressor()
    lin_reg = MySimpleLinearRegressionClassifier(discretizer, regressor)
    lin_reg.fit(X_train, y_train)

    X_test = [[150], [175]]
    y_pred_numeric = lin_reg.regressor.predict(X_test)  # Get numeric predictions
    sk_lin_reg = LinearRegression()
    sk_lin_reg.fit(X_train, y_train)
    sk_y_pred = sk_lin_reg.predict(X_test)
    assert np.allclose(y_pred_numeric, sk_y_pred)

    y_pred = lin_reg.predict(X_test)
    expected = [discretizer(y) for y in sk_y_pred]
    assert y_pred == expected


def test_kneighbors_classifier_fit():
    X_train = [[1], [2], [3], [6], [7], [8]]
    y_train = ["a", "a", "b", "b", "c", "c"]
    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    assert knn.X_train == X_train
    assert knn.y_train == y_train


"""
def test_kneighbors_classifier_kneighbors():
    # Test case 1: 4 instance training set example
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]
    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    X_test = [[0.5, 0.5]]
    distances, neighbor_indices = knn.kneighbors(X_test)

    expected_distances = [[0.707, 0.707, 0.823]]
    expected_indices = [[0, 1, 2]]

    assert np.allclose(distances, expected_distances, atol=0.01)
    assert neighbor_indices == expected_indices

    # Test case 2: 8 instance training set example
    X_train = [[3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6]]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    knn.fit(X_train, y_train)

    X_test = [[5, 5]]
    distances, neighbor_indices = knn.kneighbors(X_test)

    expected_distances = [[1.414, 2.236, 2.828]]
    expected_indices = [[1, 3, 0]]

    assert np.allclose(distances, expected_distances, atol=0.01)
    assert neighbor_indices == expected_indices

    # Test case 3: Bramer 3.6 Self-assessment exercise 2
    X_train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1],
    ]
    y_train = [
        "-",
        "-",
        "-",
        "+",
        "-",
        "+",
        "-",
        "+",
        "+",
        "+",
        "-",
        "-",
        "-",
        "-",
        "-",
        "+",
        "+",
        "+",
        "-",
        "+",
    ]
    knn.fit(X_train, y_train)

    X_test = [[10, 10]]
    distances, neighbor_indices = knn.kneighbors(X_test)

    expected_distances = [[0.806, 1.220, 1.421]]
    expected_indices = [[7, 8, 5]]

    assert np.allclose(distances, expected_distances, atol=0.01)
    assert neighbor_indices == expected_indices
"""


def test_kneighbors_classifier_predict():
    # Test case 1: 4 instance training set example
    X_train = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train = ["bad", "bad", "good", "good"]
    knn = MyKNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    X_test = [[0.5, 0.5]]
    y_pred = knn.predict(X_test)
    expected = ["bad"]
    assert y_pred == expected

    # Test case 2: 8 instance training set example
    X_train = [[3, 2], [6, 6], [4, 1], [4, 4], [1, 2], [2, 0], [0, 3], [1, 6]]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    knn.fit(X_train, y_train)

    X_test = [[5, 5]]
    y_pred = knn.predict(X_test)
    expected = ["no"]
    assert y_pred == expected

    # Test case 3: Bramer 3.6 Self-assessment exercise 2
    X_train = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1],
    ]
    y_train = [
        "-",
        "-",
        "-",
        "+",
        "-",
        "+",
        "-",
        "+",
        "+",
        "+",
        "-",
        "-",
        "-",
        "-",
        "-",
        "+",
        "+",
        "+",
        "-",
        "+",
    ]
    knn.fit(X_train, y_train)

    X_test = [[10, 10]]
    y_pred = knn.predict(X_test)
    expected = ["+"]
    assert y_pred == expected


def test_dummy_classifier_fit():
    # Test case 1
    X_train = [[1], [2], [3], [6], [7], [8]]
    y_train = ["a", "a", "b", "b", "c", "c"]
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    assert dummy.most_common_label == "a"

    # Test case 2
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy.fit(X_train, y_train)

    assert dummy.most_common_label == "yes"

    # Test case 3
    y_train = list(
        np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2])
    )
    dummy.fit(X_train, y_train)

    assert dummy.most_common_label == "no"


def test_dummy_classifier_predict():
    # Test case 1
    X_train = [[1], [2], [3], [6], [7], [8]]
    y_train = ["a", "a", "b", "b", "c", "c"]
    dummy = MyDummyClassifier()
    dummy.fit(X_train, y_train)

    X_test = [[4], [5], [9]]
    y_pred = dummy.predict(X_test)
    expected = ["a", "a", "a"]
    assert y_pred == expected

    # Test case 2
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    expected = ["yes", "yes", "yes"]
    assert y_pred == expected

    # Test case 3
    y_train = list(
        np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2])
    )
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    expected = ["no", "no", "no"]
    assert y_pred == expected


# In-Class dataset
header_inclass_example = ["att1", "att2"]
X_train_inclass_example = [
    [1, 5],  # yes
    [2, 6],  # yes
    [1, 5],  # no
    [1, 5],  # no
    [1, 6],  # yes
    [2, 6],  # no
    [1, 5],  # yes
    [1, 6],  # yes
]
y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

# iPhone dataset
header_iphone = ["standing", "job_status", "credit_rating"]
X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
]
y_train_iphone = [
    "no",
    "no",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
    "no",
    "yes",
    "yes",
    "yes",
    "yes",
    "yes",
    "no",
    "yes",
]

# Bramer 3.2 train dataset
header_train = ["day", "season", "wind", "rain"]
X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"],
]
y_train_train = [
    "on time",
    "on time",
    "on time",
    "late",
    "on time",
    "very late",
    "on time",
    "on time",
    "very late",
    "on time",
    "cancelled",
    "on time",
    "late",
    "on time",
    "very late",
    "on time",
    "on time",
    "on time",
    "on time",
    "on time",
]


def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    expected_priors_inclass = {"yes": 0.625, "no": 0.375}
    expected_posteriors_inclass = {
        "yes": {0: {1: 0.8, 2: 0.2}, 1: {5: 0.4, 6: 0.6}},
        "no": {0: {1: 0.6667, 2: 0.3333}, 1: {5: 0.6667, 6: 0.3333}},
    }

    expected_priors_iphone = {"yes": 0.6667, "no": 0.3333}
    expected_posteriors_iphone = {
        "yes": {
            0: {1: 0.2, 2: 0.8},
            1: {1: 0.3, 2: 0.4, 3: 0.3},
            2: {"fair": 0.7, "excellent": 0.3},
        },
        "no": {
            0: {1: 0.6, 2: 0.4},
            1: {1: 0.2, 2: 0.4, 3: 0.4},
            2: {"fair": 0.4, "excellent": 0.6},
        },
    }

    expected_priors_train = {
        "on time": 0.7,
        "late": 0.1,
        "very late": 0.15,
        "cancelled": 0.05,
    }
    expected_posteriors_train = {
        "on time": {
            0: {
                "weekday": 0.6428571428571429,
                "saturday": 0.14285714285714285,
                "holiday": 0.14285714285714285,
                "sunday": 0.07142857142857142,
            },
            1: {
                "spring": 0.2857142857142857,
                "winter": 0.14285714285714285,
                "summer": 0.42857142857142855,
                "autumn": 0.14285714285714285,
            },
            2: {
                "none": 0.35714285714285715,
                "normal": 0.35714285714285715,
                "high": 0.2857142857142857,
            },
            3: {
                "none": 0.35714285714285715,
                "slight": 0.5714285714285714,
                "heavy": 0.07142857142857142,
            },
        },
        "late": {
            0: {"weekday": 0.5, "saturday": 0.5},
            1: {"winter": 1.0},
            2: {"high": 0.5, "normal": 0.5},
            3: {"heavy": 0.5, "none": 0.5},
        },
        "very late": {
            0: {"weekday": 1.0},
            1: {"autumn": 0.3333333333333333, "winter": 0.6666666666666666},
            2: {"normal": 0.6666666666666666, "high": 0.3333333333333333},
            3: {"none": 0.3333333333333333, "heavy": 0.6666666666666666},
        },
        "cancelled": {
            0: {"saturday": 1.0},
            1: {"spring": 1.0},
            2: {"high": 1.0},
            3: {"heavy": 1.0},
        },
    }

    inclass_naive_bayes = MyNaiveBayesClassifier()
    iphone_naive_bayes = MyNaiveBayesClassifier()
    bramer_naive_bayes = MyNaiveBayesClassifier()

    inclass_naive_bayes.fit(X_train_inclass_example, y_train_inclass_example)
    iphone_naive_bayes.fit(X_train_iphone, y_train_iphone)
    bramer_naive_bayes.fit(X_train_train, y_train_train)

    # Asserting priors and posteriors for each dataset

    # In-Class Example
    for label, expected_prior in expected_priors_inclass.items():
        assert np.isclose(
            inclass_naive_bayes.priors[label], expected_prior, atol=0.001
        ), f"Prior for {label} does not match in in-class example"
    for label, attr_values in expected_posteriors_inclass.items():
        for attr, values in attr_values.items():
            for attr_value, expected_posterior in values.items():
                actual_posterior = inclass_naive_bayes.posteriors[label][attr].get(
                    attr_value, 0
                )
                assert np.isclose(
                    actual_posterior, expected_posterior, atol=0.001
                ), f"Posterior for {attr}={attr_value} | {label} does not match in in-class example"

    # iPhone Example
    for label, expected_prior in expected_priors_iphone.items():
        assert np.isclose(
            iphone_naive_bayes.priors[label], expected_prior, atol=0.001
        ), f"Prior for {label} does not match in iPhone example"
    for label, attr_values in expected_posteriors_iphone.items():
        for attr, values in attr_values.items():
            for attr_value, expected_posterior in values.items():
                actual_posterior = iphone_naive_bayes.posteriors[label][attr].get(
                    attr_value, 0
                )
                assert np.isclose(
                    actual_posterior, expected_posterior, atol=0.001
                ), f"Posterior for {attr}={attr_value} | {label} does not match in iPhone example"

    # Train Example
    for label, expected_prior in expected_priors_train.items():
        assert np.isclose(
            bramer_naive_bayes.priors[label], expected_prior, atol=0.001
        ), f"Prior for {label} does not match in train example"
    for label, attr_values in expected_posteriors_train.items():
        for attr, values in attr_values.items():
            for attr_value, expected_posterior in values.items():
                actual_posterior = bramer_naive_bayes.posteriors[label][attr].get(
                    attr_value, 0
                )
                assert np.isclose(
                    actual_posterior, expected_posterior, atol=0.001
                ), f"Posterior for {attr}={attr_value} | {label} does not match in train example"


"""
def test_naive_bayes_classifier_predict():
    # Testing In-Class example
    X_test_inclass_example = [[1, 5], [2, 6]]
    expected_prediction_inclass = ["yes", "yes"]

    inclass_naive_bayes = MyNaiveBayesClassifier()
    inclass_naive_bayes.fit(X_train_inclass_example, y_train_inclass_example)
    predictions_inclass = inclass_naive_bayes.predict(X_test_inclass_example)

    assert (
        predictions_inclass == expected_prediction_inclass
    ), f"In-class example predictions do not match: {predictions_inclass} vs {expected_predictions_inclass}"

    # Testing iPhone example
    X_test_iphone = [[1, 3, "fair"], [2, 1, "excellent"]]
    expected_predictions_iphone = ["no", "yes"]

    iphone_naive_bayes = MyNaiveBayesClassifier()
    iphone_naive_bayes.fit(X_train_iphone, y_train_iphone)
    predictions_iphone = iphone_naive_bayes.predict(X_test_iphone)

    assert (
        predictions_iphone == expected_predictions_iphone
    ), f"MA7 iPhone predictions do not match: {predictions_iphone} vs {expected_predictions_iphone}"

    # Testing Bramer
    X_test_bramer = [["weekday", "winter", "high", "heavy"]]
    expected_predictions_bramer = ["late"]

    bramer_naive_bayes = MyNaiveBayesClassifier()
    bramer_naive_bayes.fit(X_train_train, y_train_train)
    predictions_bramer = bramer_naive_bayes.predict(X_test_bramer)

    assert (
        predictions_bramer == expected_predictions_bramer
    ), f"Bramer 3.2 predictions do not match: {predictions_bramer} vs {expected_predictions_bramer}"

    # Bramer 3.6 self-assessment unseen instances
    X_test_bramer_exercise = [
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
    ]
    expected_predictions_bramer_exercise = ["late", "on time", "on time"]

    predictions_bramer_exercise = bramer_naive_bayes.predict(X_test_bramer_exercise)

    assert (
        predictions_bramer_exercise == expected_predictions_bramer_exercise
    ), f"Bramer 3.6 exercise predictions do not match: {predictions_bramer_exercise} vs {expected_predictions_bramer_exercise}"
"""

X_train_interview = [
    ["Senior", "Java", "no", "no"],
    ["Senior", "Java", "no", "yes"],
    ["Mid", "Python", "no", "no"],
    ["Junior", "Python", "no", "no"],
    ["Junior", "R", "yes", "no"],
    ["Junior", "R", "yes", "yes"],
    ["Mid", "R", "yes", "yes"],
    ["Senior", "Python", "no", "no"],
    ["Senior", "R", "yes", "no"],
    ["Junior", "Python", "yes", "no"],
    ["Senior", "Python", "yes", "yes"],
    ["Mid", "Python", "no", "yes"],
    ["Mid", "Java", "yes", "no"],
    ["Junior", "Python", "no", "yes"],
]
y_train_interview = [
    "False",
    "False",
    "True",
    "True",
    "True",
    "False",
    "True",
    "False",
    "True",
    "True",
    "True",
    "True",
    "True",
    "False",
]
X_test_interview = [["Senior", "Java", "no", "yes"], ["Junior", "R", "yes", "yes"]]
y_test_interview = ["False", "False"]


def test_decision_tree_classifier_predict():
    dt_classifier = MyDecisionTreeClassifier()
    dt_classifier.fit(X_train_interview, y_train_interview)

    y_pred_interview = dt_classifier.predict(X_test_interview)
    assert (
        y_pred_interview == y_test_interview
    ), f"Error in predictions: {y_pred_interview} vs {y_test_interview}"


def test_decision_tree_classifier_predict():
    dt_classifier = MyDecisionTreeClassifier()
    dt_classifier.fit(X_train_interview, y_train_interview)

    # Perform prediction
    y_pred_interview = dt_classifier.predict(X_test_interview)

    # Validate predictions
    assert (
        y_pred_interview == y_test_interview
    ), f"Error in predictions: {y_pred_interview} vs {y_test_interview}"


def test_random_forest_classifier_fit():
    rf = MyRandomForestClassifier(n=20, m=2, f=4, M=7, random_state=42)
    rf.fit(X_train_interview, y_train_interview)

    assert len(rf.trees) == 7
    assert len(rf.feature_indices) == 7
    for tree in rf.trees:
        assert isinstance(tree, MyDecisionTreeClassifier)


def test_random_forest_classifier_predict():
    rf = MyRandomForestClassifier(n=20, m=2, f=4, M=7, random_state=42)
    rf.fit(X_train_interview, y_train_interview)
    y_pred = rf.predict(X_test_interview)

    print(f"Predictions: {y_pred}")
    print(f"Expected: {y_test_interview}")

    assert len(y_pred) == len(X_test_interview)
    assert (
        y_pred == y_test_interview
    ), f"Error in predictions: {y_pred} vs {y_test_interview}"


pytest.main()
