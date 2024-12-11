# some useful mysklearn package import statements and reloads
import importlib
import pickle

import mysklearn.myutils

importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils

# uncomment once you paste your mypytable.py into mysklearn package
import mysklearn.mypytable

importlib.reload(mysklearn.mypytable)
from mysklearn.mypytable import MyPyTable

# uncomment once you paste your myclassifiers.py into mysklearn package
import mysklearn.myclassifiers

importlib.reload(mysklearn.myclassifiers)
from mysklearn.myclassifiers import MyDecisionTreeClassifier

import mysklearn.myevaluation

importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation

import mysklearn.mysimplelinearregressor

importlib.reload(mysklearn.mysimplelinearregressor)

train = MyPyTable().load_from_file("input_data/nba_elo.csv")
test = MyPyTable().load_from_file("input_data/nba_elo_latest.csv")


train = MyPyTable().load_from_file("input_data/nba_elo.csv")
test = MyPyTable().load_from_file("input_data/nba_elo_latest.csv")

train.drop_duplicate_seasons(2021)
small_train = train.random_down_sample(num_rows=60000)
# First we will calculate the winners
train_winner = myutils.calculate_winners(small_train)
test_winner = myutils.calculate_winners(test)

# Next we add a new column to both test and train sets
small_train.add_column("winner", train_winner)
test.add_column("winner", test_winner)

train_attributes = [
    "elo1_pre",
    "elo2_pre",
    "elo_prob1",
    "elo_prob2",
    "team1",
    "team2",
    "score1",
    "score2",
]

attribute_indexes = [small_train.column_names.index(attr) for attr in train_attributes]

X = [[row[index] for index in attribute_indexes] for row in small_train.data]
y = small_train.get_column("winner")

X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y)

X_train_discretized = myutils.preprocess_data(X_train)
X_test_discretized = myutils.preprocess_data(X_test)

naive_bayes = MyDecisionTreeClassifier()
naive_bayes.fit(X_train_discretized, y_train)

with open("nb_model.pkl", "wb") as model_file:
    pickle.dump(naive_bayes, model_file)
