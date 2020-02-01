import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score


# >> FEATURE ENGINEERING << #
def remove_correlated_features(X):
    corr_threshold = 0.9
    corr = X.corr()
    drop_columns = np.full(corr.shape[0], False, dtype=bool)
    for i in range(corr.shape[0]):
        for j in range(i + 1, corr.shape[0]):
            if corr.iloc[i, j] >= corr_threshold:
                drop_columns[j] = True
    columns_dropped = X.columns[drop_columns]
    X.drop(columns_dropped, axis=1, inplace=True)
    return columns_dropped


def remove_less_significant_features(X, Y):
    sl = 0.05
    regression_ols = None
    columns_dropped = np.array([])
    for itr in range(0, len(X.columns)):
        regression_ols = sm.OLS(Y, X).fit()
        max_col = regression_ols.pvalues.idxmax()
        max_val = regression_ols.pvalues.max()
        if max_val > sl:
            X.drop(max_col, axis='columns', inplace=True)
            columns_dropped = np.append(columns_dropped, [max_col])
        else:
            break
    regression_ols.summary()
    return columns_dropped


##############################


# >> MODEL TRAINING << #
def compute_cost(W, X, Y):
    cost = 1 / 2 * np.dot(W, W)
    distances = 1 - Y * (np.dot(X, W))
    distances[distances < 0] = 0
    hinge_loss = np.sum(regularization_strength * distances)
    hinge_loss = hinge_loss / X.shape[0]
    cost += hinge_loss
    return cost


def calculate_cost_derivative(W, X_batch, Y_batch):
    # if only one example then convert it to array
    if type(Y_batch) == np.float64:
        Y_batch = np.array([Y_batch])
        X_batch = np.array([X_batch])

    distance = 1 - (Y_batch * np.dot(X_batch, W))
    derivative = 0

    for ind, d in enumerate(distance):
        if d > 0:
            di = W - (regularization_strength * Y_batch[ind] * X_batch[ind])
        else:
            di = W
        derivative += di

    derivative = derivative/len(Y_batch)  # average
    return derivative


def fit_svm(features, outputs):
    epochs = 500000
    rows = features.shape[0]
    weights = np.zeros(features.shape[1])
    nth = 0
    # stochastic gradient descent
    for epoch in range(1, epochs):
        rand_row = np.random.randint(rows)
        ascent = calculate_cost_derivative(weights, features[rand_row], outputs[rand_row])
        weights = weights - (learning_rate * ascent)
        if epoch == 2 ** nth:
            cost = compute_cost(weights, features, outputs)
            nth += 1
            print("Epoch is: {} and Cost is: {}".format(epoch, cost))
    return weights


########################


def init():
    print("reading dataset...")
    # read data in pandas (pd) data frame
    data = pd.read_csv('./data/data.csv')

    # drop last column (extra column added by pd) and first column (id)
    data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

    print("applying feature engineering...")
    # convert categorical labels to numbers
    diag_map = {'M': 1.0, 'B': -1.0}
    data['diagnosis'] = data['diagnosis'].map(diag_map)

    # put features & outputs in different data frames
    Y = data.loc[:, 'diagnosis']
    X = data.iloc[:, 1:]

    # filter features
    remove_correlated_features(X)
    remove_less_significant_features(X, Y)

    # normalize data for better convergence and to prevent overflow
    X_normalized = MinMaxScaler().fit_transform(X.values)
    X = pd.DataFrame(X_normalized)

    # split data into train and test set
    print("splitting dataset into train and test sets...")
    X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

    # train the model
    print("training started...")
    W = fit_svm(X_train.to_numpy(), y_train.to_numpy())
    print("training finished.")
    print("weights are: {}".format(W))

    # check accuracy
    print("checking accuracy...")
    train_prediction = np.array([])
    test_prediction = np.array([])
    for i in range(X_train.shape[0]):
        yp = np.sign(np.dot(W, X_train.iloc[i].values))
        train_prediction = np.append(train_prediction, yp)

    for i in range(X_test.shape[0]):
        yp = np.sign(np.dot(W, X_test.iloc[i].values))
        test_prediction = np.append(test_prediction, yp)

    print("accuracy on train dataset: {}".format(accuracy_score(y_train.to_numpy(), train_prediction)))
    print("accuracy on test dataset: {}".format(accuracy_score(y_test.to_numpy(), test_prediction)))


# set hyper-parameters and call init
regularization_strength = 10000
learning_rate = 0.000001
init()
