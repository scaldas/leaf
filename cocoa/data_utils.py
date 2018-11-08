import numpy as np
import os

from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize


def load_and_prepare_data(dataset, datarepo='../mocha/data/'):
    dataset_path = os.path.join(datarepo, '%s.mat' % dataset)
    data_temp = io.loadmat(dataset_path)
    X, y = data_temp['X'], data_temp['Y']

    if len(X) is not 1:
        X, y = np.transpose(X), np.transpose(y)

    num_tasks = len(X[0])

    if dataset == 'har':
        # preprocess: one-vs-all between sitting and other activities
        dynamic = [4]
        for t in range(num_tasks):
            binary_labels = [[1] if label in dynamic else [-1] for label in y[0, t]]
            y[0, t] = np.array(binary_labels)
    X = normalize_features(X)
    X = add_bias(X)
    return X, y


def normalize_features(X):
    num_tasks = len(X[0])
    for t in range(num_tasks):
        X[0, t] = normalize(X[0, t], axis=0)
    return X


def add_bias(X):
    num_tasks = len(X[0])
    for t in range(num_tasks):
        shape = X[0, t].shape
        new_design_matrix = np.ones((shape[0], shape[1] + 1))
        new_design_matrix[:, :-1] = X[0, t]
        X[0, t] = new_design_matrix
    return X


def split_data(X, y, training_perc):
    num_tasks = len(X[0])
    X_train, y_train, X_test, y_test = [], [], [], []

    for t in range(num_tasks):
        Xt_train, Xt_test, yt_train, yt_test = train_test_split(
            X[0, t], y[0, t], test_size=1 - training_perc)

        X_train.append(Xt_train)
        X_test.append(Xt_test)
        y_train.append(yt_train)
        y_test.append(yt_test)

    return np.array([X_train]), np.array([X_test]), np.array([y_train]), np.array([y_test])


def flatten_tasks(X, y):
    num_tasks = len(X[0])
    X_flat = X[0, 0]
    y_flat = y[0, 0]

    for t in range(1, num_tasks):
        X_flat = np.append(X_flat, X[0, t], 0)
        y_flat = np.append(y_flat, y[0, t], 0)

    return X_flat, y_flat