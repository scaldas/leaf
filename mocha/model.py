"""Implementations for ClientModel and ServerModel."""
import numpy as np
import os
import sys

from baseline_constants import ACCURACY_KEY

from utils.model_utils import batch_data
from utils.tf_utils import graph_size


class ClientModel:

    def __init__(self, num_samples, lanbda, local_iters_perc):
        self.num_samples = num_samples
        self.lanbda = lanbda
        self.local_iters_perc = local_iters_perc

        self.alpha = np.zeros((self.num_samples, 1), dtype=np.float64)
        
        #self.W = np.zeros((num_features, 1), dtype=np.float64)
        #self.deltaW = np.zeros((num_features, 1), dtype=np.float64)
        #self.deltaB = np.zeros((num_features, 1), dtype=np.float64)

    def train(self, data, W_t, sigma, rho):
        local_iters = int(local_iters_perc * self.num_samples)
        rand_perm = np.random.permutation(self.num_samples)

        deltaW = np.zeros((num_features, 1), dtype=np.float64)
        deltaB = np.zeros((num_features, 1), dtype=np.float64)

        # Run SDCA locally.
        for s in range(local_iters):
            # Select random coordinate.
            idx = rand_perm[s]
            alpha_old = self.alpha[idx]
            curr_x = data['x'][idx, :]
            curr_y = data['y'][idx]

            # Compute update.
            update = curr_y * np.dot(curr_x, (W_t + rho * deltaW))
            grad = self.lanbda * self.num_samples * (1.0 - update) / (
                    sigma * rho * np.dot(curr_x, curr_x.T)) + (alpha_old * curr_y)
            self.alpha[idx] = curr_y * max(0.0, min(1.0, grad))
            deltaW = deltaW + curr_sig * (self.alpha[idx] - alpha_old) * curr_x.T / (
                    self.lanbda * self.num_samples)
            deltaB = deltaB + (self.alpha[idx] - alpha_old) * curr_x.T / self.num_samples

        return deltaB

    def test(self, data):
        """
        Tests the current model on the given data.

        Args:
            data: dict of the form {'x': [list], 'y': [list]}
        Return:
            dict of metrics that will be recorded by the simulation.
        """
        x_vecs = self.process_x(data['x'])
        labels = self.process_y(data['y'])
        with self.graph.as_default():
            tot_acc = self.sess.run(
                self.eval_metric_ops,
                feed_dict={self.features: x_vecs, self.labels: labels}
            )
        acc = float(tot_acc) / x_vecs.shape[0]
        return {ACCURACY_KEY: acc}

    def close(self):
        self.sess.close()

    def process_x(self, raw_x_batch):
        """Pre-processes each batch of features before being fed to the model."""
        return np.asarray(raw_x_batch)

    def process_y(self, raw_y_batch):
        """Pre-processes each batch of labels before being fed to the model."""
        return np.asarray(raw_y_batch)


class ServerModel:
    def __init__(self, num_tasks, num_features, lanbda):
        self.num_tasks = num_tasks
        self.lanbda = lanbda

        self.W = np.zeros((num_features, self.num_tasks), dtype=np.float64)
        self.Sigma = np.eye(self.num_tasks, dtype=np.float64) * (1.0 / self.num_tasks)
        self.Omega = np.eye(self.num_tasks, dtype=np.float64) * (self.num_tasks)
        self.rho = 1.

    @property
    def cur_model(self):
        return self.W

    def send_to(self, clients):
        """Copies server model variables to each of the given clients

        Args:
            clients: list of Client objects
        """
        for c in clients:
            c.W_t = 

    def get_sigma(t):
        return Sigma[t, t]

    def update_weights(self, deltaB):
        """Updates server model using given client updates.

        Args:
            updates: list of (num_samples, update), where num_samples is the
                number of training samples corresponding to the update, and update
                is a list of variable weights
        """
        for t in range(self.num_tasks):
            for u in range(self.num_tasks):
                self.W[:, t] = self.W[:, t] + deltaB[:, u] * self.Sigma[t, u] * (1.0 / self.lanbda)

    def update_covariances(self):
        _, S, V = np.linalg.svd(self.W, full_matrices=False)
        S_inv = np.diag(S ** -1)
        S_trace = np.sum(S)
        S = np.diag(S)

        self.Sigma = 1.0 / S_trace * np.dot(V.T, np.dot(S, V))
        self.Omega = S_trace * np.dot(V.T, np.dot(S_inv, V))
        self.rho = max(np.sum(np.absolute(Sigma), 0) / np.diag(Sigma))

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()
