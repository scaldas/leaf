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
        self.theta = 0.18

        
    def train(self, data, W_t, sigma, omega, rho, max_curr_iter, total_n):
        local_iters = int(self.local_iters_perc * self.num_samples)
        rand_perm = np.random.permutation(self.num_samples)

        deltaW = np.zeros(self.num_features, dtype=np.float64)
        deltaB = np.zeros(self.num_features, dtype=np.float64)
        curr_theta = 1000
        curr_iter = 1

        # Run SDCA locally.
        while (curr_theta > self.theta and curr_iter < 500):
            # Select random coordinate.
            idx = rand_perm[(curr_iter % self.num_samples)+1]
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
            
            if (curr_iter % 5 == 1 and max_curr_iter < 500):
                curr_theta = self.compute_local_gap(data['x'],
                    data['y'], W_t, omega, rho, total_n, sigma)
                break
            curr_iter += 1

        return deltaB, curr_iter

    def compute_local_gap(self, X, y, W_t, omega, rho, total_n, sigma):
        dual_loss = np.mean(-1.0 * np.dot(self.alpha.T, y))
        Xalph = np.dot(X.T, self.alpha).reshape(-1,1)
        dual_reg = (1.0 / self.num_samples * (np.dot(W_t.T, Xalph)))
        dual_reg = dual_reg + (rho / (2 * self.lanbda * self.num_samples * self.num_samples) * sigma * sigma * np.dot(Xalph.T, Xalph))
        dual_obj = - dual_reg - dual_loss
        preds = np.multiply(y.flatten(),np.dot(X, W_t))
        primal_loss = 1 / (total_n) * sum(np.maximum(np.zeros(preds.shape[0]), 1.0 - preds))
        primal_reg = self.lanbda / (rho * 2) * omega * omega * np.dot(W_t.T, W_t)
        primal_obj = primal_loss + primal_reg
        gap = primal_obj - dual_obj
        #print ('***gap***', gap)
        return gap[0][0]

class ServerModel:
    def __init__(self, num_clients, num_features, lanbda):
        self.num_clients = num_clients
        self.lanbda = lanbda

        self.W = np.zeros((num_features, self.num_clients), dtype=np.float64)
        self.Sigma = np.eye(self.num_clients, dtype=np.float64) * (1.0 / self.num_clients)
        self.Omega = np.eye(self.num_clients, dtype=np.float64) * (self.num_clients)
        self.rho = 1.
        self.max_curr_iter = 0

    @property
    def cur_model(self):
        return self.W

    def get_sigma(self, t):
        return self.Sigma[t, t]

    def get_omega(self, t):
        return self.Omega[t, t]

    def update_weights(self, deltaB):
        for t in range(self.num_clients):
            for u in range(self.num_clients):
                self.W[:, t] = self.W[:, t] + deltaB[:, u] * self.Sigma[t, u] * (1.0 / self.lanbda)

    def update_curr_iters(self, curr_iter):
        if curr_iter > self.max_curr_iter:
            self.max_curr_iter = curr_iter
        
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
