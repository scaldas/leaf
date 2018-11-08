"""Script to run the baselines."""

import argparse
import importlib
import numpy as np
import os
import sys

import data_utils

from client import Client
from server import Server
from model import ClientModel, ServerModel


def main():

    args = parse_args()

    args.dataset
    
    print('############################## %s ##############################' % args.dataset)


    # Create clients
    clients, num_features = setup_clients(args.dataset, args.lanbda, args.local_iters_perc)
    num_clients = len(clients)

    # Create server
    server_model = ServerModel(num_clients, num_features, args.lanbda)
    server = Server(num_clients, num_features, server_model)


    print('%d Clients in Total' % num_clients)

    # Test untrained model on all clients
    acc = server.test_model(clients)
    print(acc)

    # Simulate training
    for i in range(args.num_outer_iters):
        print('--- Round %d of %d ---' % (i+1, args.num_outer_iters))

        server.orchestrate_local_training(args.num_inner_iters, clients)
        server.update_global_covariances()

        acc = server.test_model(clients)
        print(acc)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=['gleam'],
                    required=True)
    parser.add_argument('--lanbda',
                        help='lambda.; default: 1;',
                        type=float,
                        default=1.0)
    parser.add_argument('--num-outer-iters',
                        help='max outer iters.; default: 10;',
                        type=int,
                        default=10)
    parser.add_argument('--num-inner-iters',
                        help='max inner iters.; default: 50;',
                        type=int,
                        default=50)
    parser.add_argument('--local-iters-perc',
                        help='local iters perc.; default: 0.5;',
                        type=float,
                        default=0.5)

    return parser.parse_args()


def setup_clients(dataset, lanbda, local_iters_perc):
    """Instantiates clients based on the given data directory.

    Return:
        all_clients: list of Client objects.
    """
    X, y = data_utils.load_and_prepare_data(dataset)
    Xtrain, Xtest, ytrain, ytest = data_utils.split_data(X, y, 0.8)

    num_tasks = len(Xtrain[0])
    num_features = Xtrain[0, 0].shape[1]
    all_clients = []
    for t in range(num_tasks):
        c = Client(
            t,
            {'x': Xtrain[0, t], 'y': ytrain[0, t]},
            {'x': Xtest[0, t], 'y': ytest[0, t]},
            ClientModel(Xtrain[0, t].shape[0], num_features, lanbda, local_iters_perc))
        all_clients.append(c)

    return all_clients, num_features


if __name__ == '__main__':
    main()
