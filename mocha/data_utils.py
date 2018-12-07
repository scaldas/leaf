import json
import numpy as np
import os
import scipy

from scipy import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer


def load_and_prepare_data(dataset):
    if dataset == 'gleam':
        return load_and_prepare_data_gleam()
    elif dataset == 'femnist':
        return load_and_prepare_data_femnist()
    elif dataset == 'sent140':
        return load_and_prepare_data_sent140()

def load_and_prepare_data_gleam(datarepo='./data/'):
    dataset_path = os.path.join(datarepo, 'gleam.mat')
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

    return data_utils.split_data(X, y, 0.8)

def load_and_prepare_data_femnist():
    base_dir = '../data/femnist/data/'
    
    train_data = get_json_data(os.path.join(base_dir, 'train'))
    test_data = get_json_data(os.path.join(base_dir, 'test'))

    keys_list = [k for k in train_data.keys()]

    X_train, y_train, X_test, y_test = [], [], [], []

    for user_id in keys_list:
        X_train_u = np.array([np.array(sample) for sample in train_data[user_id]['x']])
        y_train_u = np.array([0 if i <= 9 else 1 for i in train_data[user_id]['y']])

        X_test_u = np.array([np.array(sample) for sample in test_data[user_id]['x']])
        y_test_u = np.array([0 if i <= 9 else 1 for i in test_data[user_id]['y']])
        
        X_train.append(X_train_u)
        y_train.append(y_train_u)
        
        X_test.append(X_test_u)
        y_test.append(y_test_u)

    Xtrain = np.array([X_train])
    Xtest = np.array([X_test])

    ytrain = np.array([y_train])
    ytest = np.array([y_test])

    return Xtrain, Xtest, ytrain, ytest


def load_and_prepare_data_sent140():
    base_dir = '../data/sent140/data/'

    train_data = get_json_data(os.path.join(base_dir, 'train'))
    test_data = get_json_data(os.path.join(base_dir, 'test'))

    keys_list = [k for k in train_data.keys()]
    X_train, y_train, X_test, y_test = [], [], [], []

    for user_id in keys_list:
        X_train_u = np.array([sample[4] for sample in train_data[user_id]['x']])
        y_train_u = np.array([2*i - 1 for i in train_data[user_id]['y']])

        X_test_u = np.array([sample[4] for sample in test_data[user_id]['x']])
        y_test_u = np.array([2*i - 1 for i in test_data[user_id]['y']])
        
        if X_train_u.shape[0] > 100:
            X_train.append(X_train_u)
            y_train.append(y_train_u)
            
            X_test.append(X_test_u)
            y_test.append(y_test_u)

    all_train_data = [tweet for user in X_train for tweet in user]
    count_vect = CountVectorizer(max_features=1000)
    count_vect = count_vect.fit(all_train_data)

    X_train_bow, X_test_bow = [], []

    for i in range(len(X_train)):
        X_train_bow.append(np.array(scipy.sparse.csr_matrix.todense(count_vect.transform(X_train[i]))))
        X_test_bow.append(np.array(scipy.sparse.csr_matrix.todense(count_vect.transform(X_test[i]))))

    Xtrain = np.array([X_train_bow])
    Xtest = np.array([X_test_bow])

    ytrain = np.array([y_train])
    ytest = np.array([y_test])

    return Xtrain, Xtest, ytrain, ytest


def get_json_data(path):   
    onlyfiles = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    all_data = None
    for f in onlyfiles:
        with open(f, 'r') as fp:
            data = json.load(fp)
        if all_data is None:
           all_data = data['user_data']
        else:
            all_data.update(data['user_data'])
    return all_data


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