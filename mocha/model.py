"""Implementations for ClientModel and ServerModel."""
import numpy as np
import os
import sys


class ClientModel:

    def __init__(self, num_samples, num_features, lanbda, local_iters_perc):
        self.num_samples = num_samples
        self.num_features = num_features
        self.lanbda = lanbda
        self.local_iters_perc = local_iters_perc

        self.alpha = np.zeros(self.num_samples, dtype=np.float64)
        
    def train(self, data, W_t, sigma, rho):
        local_iters = int(self.local_iters_perc * self.num_samples)
        rand_perm = np.random.permutation(self.num_samples)

        deltaW = np.zeros(self.num_features, dtype=np.float64)
        deltaB = np.zeros(self.num_features, dtype=np.float64)

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
            deltaW = deltaW + sigma * (self.alpha[idx] - alpha_old) * curr_x.T / (
                    self.lanbda * self.num_samples)
            deltaB = deltaB + (self.alpha[idx] - alpha_old) * curr_x.T / self.num_samples

        return deltaB


class ServerModel:
    def __init__(self, num_clients, num_features, lanbda):
        self.num_clients = num_clients
        self.lanbda = lanbda

        self.W = np.zeros((num_features, self.num_clients), dtype=np.float64)
        self.Sigma = np.eye(self.num_clients, dtype=np.float64) * (1.0 / self.num_clients)
        self.Omega = np.eye(self.num_clients, dtype=np.float64) * (self.num_clients)
        self.rho = 1.

    @property
    def cur_model(self):
        return self.W

    def get_sigma(self, t):
        return self.Sigma[t, t]

    def update_weights(self, deltaB):
        for t in range(self.num_clients):
            for u in range(self.num_clients):
                self.W[:, t] = self.W[:, t] + deltaB[:, u] * self.Sigma[t, u] * (1.0 / self.lanbda)

    def update_covariances(self):
        _, S, V = np.linalg.svd(self.W, full_matrices=False)
        S_inv = np.diag(S ** -1)
        S_trace = np.sum(S)
        S = np.diag(S)

        self.Sigma = 1.0 / S_trace * np.dot(V.T, np.dot(S, V))
        self.Omega = S_trace * np.dot(V.T, np.dot(S_inv, V))
        self.rho = max(np.sum(np.absolute(self.Sigma), 0) / np.diag(self.Sigma))

    def save(self, path='checkpoints/model.ckpt'):
        return self.model.saver.save(self.model.sess, path)

    def close(self):
        self.model.close()
