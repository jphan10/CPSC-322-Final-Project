from graphviz import Digraph
import numpy as np
from collections import Counter
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier."""

    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closest neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []

        for test_instance in X_test:
            instance_distances = []
            for i, train_instance in enumerate(self.X_train):
                distance = np.linalg.norm(
                    np.array(test_instance) - np.array(train_instance)
                )
                instance_distances.append((distance, i))

            # Sort distances and take the k nearest neighbors
            sorted_distances = sorted(instance_distances, key=lambda x: x[0])
            k_nearest = sorted_distances[: self.n_neighbors]
            distances.append([x[0] for x in k_nearest])
            neighbor_indices.append([x[1] for x in k_nearest])

        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        _, neighbor_indices = self.kneighbors(X_test)
        y_predicted = []

        for neighbors in neighbor_indices:
            neighbor_labels = [self.y_train[i] for i in neighbors]
            most_common = Counter(neighbor_labels).most_common(1)[0][0]
            y_predicted.append(most_common)

        return y_predicted


class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy."""

    def __init__(self):
        """Initializer for DummyClassifier."""
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        self.most_common_label = Counter(y_train).most_common(1)[0][0]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for _ in X_test]


class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = (
            regressor if regressor is not None else MySimpleLinearRegressor()
        )

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_pred = self.regressor.predict(X_test)
        return [self.discretizer(y) for y in y_pred]


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        posteriors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyNaiveBayesClassifier."""
        self.priors = None
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """

        # Step 1: Calculate priors
        total_instances = len(y_train)
        class_counts = {}

        # Count occurrences of each class label in y_train
        for label in y_train:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1

        # Calculate prior probabilities (ensure floating-point division)
        self.priors = {
            label: count / float(total_instances)
            for label, count in class_counts.items()
        }

        # Initialize posteriors as a nested dictionary
        self.posteriors = {}

        # Count occurrences of each attribute value per class
        for instance, label in zip(X_train, y_train):
            if label not in self.posteriors:
                self.posteriors[label] = {}
            for attr_idx, attr_value in enumerate(instance):
                if attr_idx not in self.posteriors[label]:
                    self.posteriors[label][attr_idx] = {}
                if attr_value not in self.posteriors[label][attr_idx]:
                    self.posteriors[label][attr_idx][attr_value] = 0
                self.posteriors[label][attr_idx][attr_value] += 1

        # Convert counts to probabilities for each class
        for label, attr_dict in self.posteriors.items():
            class_count = class_counts[label]
            for attr_idx, value_dict in attr_dict.items():
                for value, count in value_dict.items():
                    self.posteriors[label][attr_idx][value] = count / float(class_count)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for instance in X_test:
            # Calculate the log-probabilities for numerical stability
            log_probabilities = {}

            for label in self.priors:
                # Start with the log of the prior probability
                log_prob = np.log(self.priors[label])

                # Add the log of the conditional probabilities for each attribute
                for attr_idx, attr_value in enumerate(instance):
                    if attr_idx in self.posteriors[label]:
                        attr_value_dict = self.posteriors[label][attr_idx]
                        if attr_value in attr_value_dict:
                            # Attribute value was seen during training
                            log_prob += np.log(attr_value_dict[attr_value])
                        else:
                            total_attr_values = len(attr_value_dict)
                            smoothed_prob = 1 / (
                                total_attr_values + sum(attr_value_dict.values()) + 1
                            )
                            log_prob += np.log(smoothed_prob)
                    else:
                        # Apply smoothing when the attribute index was not observed
                        log_prob += np.log(1 / (1 + len(self.priors)))

                log_probabilities[label] = log_prob

            # Select the class with the highest log-probability
            predicted_label = max(log_probabilities, key=log_probabilities.get)
            y_predicted.append(predicted_label)

        return y_predicted


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self):
        """Initializer for MyDecisionTreeClassifier."""
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        # Generate default attribute names
        attribute_names = [f"att{i}" for i in range(len(X_train[0]))]

        def entropy(class_counts):
            """Calculate the entropy of a distribution of class counts."""
            total = sum(class_counts.values())
            return -sum(
                (count / total) * np.log2(count / total)
                for count in class_counts.values()
                if count > 0
            )

        def select_attribute(data, labels, attributes):
            """Select the best attribute using entropy."""
            base_entropy = entropy(Counter(labels))
            best_gain = -1
            best_attr = None

            for attr_index in attributes:
                partitions = {}
                for row, label in zip(data, labels):
                    value = row[attr_index]
                    if value not in partitions:
                        partitions[value] = []
                    partitions[value].append(label)

                attr_entropy = sum(
                    (len(part) / len(labels)) * entropy(Counter(part))
                    for part in partitions.values()
                )
                gain = base_entropy - attr_entropy

                if gain > best_gain:
                    best_gain = gain
                    best_attr = attr_index

            return best_attr

        def tdidt(data, labels, attributes):
            """Recursive function for tree induction."""
            label_counts = Counter(labels)
            if len(label_counts) == 1:
                # Pure leaf
                return ["Leaf", labels[0], len(labels), len(self.y_train)]

            if not attributes:
                # Majority voting (resolve ties alphabetically)
                most_common_label = sorted(
                    label_counts.items(), key=lambda x: (-x[1], x[0])
                )[0][0]
                return ["Leaf", most_common_label, len(labels), len(self.y_train)]

            # Choose the best attribute to split
            best_attr = select_attribute(data, labels, attributes)
            tree = ["Attribute", attribute_names[best_attr]]

            # Partition data
            partitions = {}
            for row, label in zip(data, labels):
                value = row[best_attr]
                if value not in partitions:
                    partitions[value] = ([], [])
                partitions[value][0].append(row)
                partitions[value][1].append(label)

            # Recur for each partition
            remaining_attributes = [attr for attr in attributes if attr != best_attr]
            for value, (subset_data, subset_labels) in sorted(
                partitions.items(), key=lambda x: str(x[0])
            ):
                subtree = tdidt(subset_data, subset_labels, remaining_attributes)
                tree.append(["Value", value, subtree])

            return tree

        # Build the tree
        self.tree = tdidt(X_train, y_train, list(range(len(X_train[0]))))

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """

        def classify(instance, subtree):
            """Traverse the tree to classify a single instance."""
            if subtree[0] == "Leaf":
                return subtree[1]

            attr_index = int(subtree[1][3:])  # Extract attribute index from "attX"
            value = instance[attr_index]

            for child in subtree[2:]:
                if child[1] == value:
                    return classify(instance, child[2])

            # Handle unseen attribute values
            leaf_nodes = [child[2] for child in subtree[2:] if child[2][0] == "Leaf"]
            majority_leaf = max(leaf_nodes, key=lambda leaf: leaf[2])
            return majority_leaf[1]

        return [classify(instance, self.tree) for instance in X_test]

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        if attribute_names is None:
            attribute_names = [f"att{i}" for i in range(len(self.X_train[0]))]

        def traverse(subtree, rule):
            """Recursively extract rules from the tree."""
            if subtree[0] == "Leaf":
                # Print the complete rule for the leaf
                print(f"IF {' AND '.join(rule)} THEN {class_name} = {subtree[1]}")
                return

            # Extract the attribute name
            attr_index = int(subtree[1][3:])  # Extract index from "attX"
            attr_name = attribute_names[attr_index]

            # Traverse each branch of the attribute
            for child in subtree[2:]:
                value = child[1]  # The value for this branch
                traverse(child[2], rule + [f"{attr_name} == {value}"])

        # Start traversing from the root
        traverse(self.tree, [])

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
        """
        if attribute_names is None:
            attribute_names = [f"att{i}" for i in range(len(self.X_train[0]))]

        dot = Digraph(comment="Decision Tree", format="pdf")
        node_id = 0  # Unique identifier for each node

        def traverse(subtree, parent_id=None):
            nonlocal node_id
            if subtree[0] == "Leaf":
                # Add a leaf node
                leaf_label = f"Class = {subtree[1]}\nSamples = {subtree[2]}\nTotal = {subtree[3]}"
                current_id = str(node_id)
                dot.node(
                    current_id,
                    leaf_label,
                    shape="box",
                    style="filled",
                    color="lightgrey",
                )
                node_id += 1
                if parent_id is not None:
                    dot.edge(parent_id, current_id)
                return

            # Add an attribute node
            attr_index = int(subtree[1][3:])  # Extract index from "attX"
            attr_label = attribute_names[attr_index]
            current_id = str(node_id)
            dot.node(
                current_id,
                attr_label,
                shape="ellipse",
                style="filled",
                color="lightblue",
            )
            node_id += 1
            if parent_id is not None:
                dot.edge(parent_id, current_id)

            # Traverse children
            for child in subtree[2:]:
                if child[0] == "Value":
                    traverse(child[2], current_id)

        traverse(self.tree)

        try:
            dot.render(dot_fname, view=False)
            print(f".dot file saved to: {dot_fname}")
            print(f".pdf file saved to: {pdf_fname}")
        except Exception as e:
            print(f"Error generating files: {e}")


def compute_bootstrapped_sample(table):
    n = len(table)
    sampled_indexes = [np.random.randint(0, n) for _ in range(n)]
    sample = [table[index] for index in sampled_indexes]
    out_of_bag_indexes = [index for index in range(n) if index not in sampled_indexes]
    out_of_bag_sample = [table[index] for index in out_of_bag_indexes]
    return sample, out_of_bag_sample


def compute_random_subset(values, num_values):
    values_copy = values[:]  # Create a shallow copy
    np.random.shuffle(values_copy)
    return values_copy[:num_values]


class MyRandomForestClassifier:
    def __init__(self, n=20, m="sqrt", f=None, M=None, random_state=None):
        """Initializer for MyRandomForestClassifier.

        Args:
            n (int): The number of trees in the forest.
            m (str or int): The number of features to consider when looking for the best split.
            f (int): The total number of features in the dataset.
            M (int): The number of selected trees based on validation accuracy.
            random_state (int): Seed for the random number generator.
        """
        self.n = n
        self.m = m
        self.f = f
        self.M = M
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []
        self.validation_accuracies = []

    def fit(self, X_train, y_train):
        """Fits the random forest classifier to the training data.

        Args:
            X_train (list of list of obj): The list of training instances (samples).
            y_train (list of obj): The target y values (parallel to X_train).
        """
        np.random.seed(self.random_state)
        n_samples = len(X_train)
        n_features = self.f if self.f is not None else len(X_train[0])

        for _ in range(self.n):
            # Bootstrap sampling
            bootstrap_sample, oob_sample = compute_bootstrapped_sample(
                list(zip(X_train, y_train))
            )
            X_bootstrap, y_bootstrap = zip(*bootstrap_sample)
            X_oob, y_oob = zip(*oob_sample)

            # Random feature selection
            if self.m == "sqrt":
                max_features = int(np.sqrt(n_features))
            elif self.m == "log2":
                max_features = int(np.log2(n_features))
            else:
                max_features = self.m

            # Ensure max_features is not greater than n_features
            max_features = min(max_features, n_features)

            feature_indices = compute_random_subset(
                list(range(n_features)), max_features
            )
            self.feature_indices.append(feature_indices)

            # Train a decision tree on the bootstrap sample
            tree = MyDecisionTreeClassifier()
            X_bootstrap_subset = [[x[i] for i in feature_indices] for x in X_bootstrap]
            tree.fit(X_bootstrap_subset, y_bootstrap)
            self.trees.append(tree)

            # Evaluate the tree using the out-of-bag sample
            X_oob_subset = [[x[i] for i in feature_indices] for x in X_oob]
            y_pred_oob = tree.predict(X_oob_subset)
            accuracy = sum(
                1 for y_true, y_pred in zip(y_oob, y_pred_oob) if y_true == y_pred
            ) / len(y_oob)
            self.validation_accuracies.append(accuracy)

        # Select the M most accurate trees
        if self.M is not None:
            top_trees_indices = np.argsort(self.validation_accuracies)[-self.M :]
            self.trees = [self.trees[i] for i in top_trees_indices]
            self.feature_indices = [self.feature_indices[i] for i in top_trees_indices]

    def predict(self, X_test):
        """Predicts the class labels for the provided test data.

        Args:
            X_test (list of list of obj): The list of testing samples.

        Returns:
            list of obj: The predicted class labels.
        """
        predictions = []
        for tree, feature_indices in zip(self.trees, self.feature_indices):
            X_test_subset = [[x[i] for i in feature_indices] for x in X_test]
            predictions.append(tree.predict(X_test_subset))

        # Aggregate predictions (majority vote)
        y_pred = []
        for i in range(len(X_test)):
            votes = [pred[i] for pred in predictions]
            majority_vote = Counter(votes).most_common(1)[0][0]
            y_pred.append(majority_vote)

        return y_pred
