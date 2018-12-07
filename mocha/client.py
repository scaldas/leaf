import numpy as np

from sklearn.metrics import accuracy_score


class Client:
    
    def __init__(self, client_id, train_data, eval_data, model=None, group=[]):
        self._model = model
        self.id = client_id # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data

        self.W_t = None

    def train(self, sigma, rho):
        """Trains on self.model using the client's train_data."""
        deltaB_t = self.model.train(self.train_data, self.W_t, sigma, rho)
        return deltaB_t

    def test(self):
        """Tests local model on self.eval_data.
        Return:
            accuracy.
        """
        x_vecs = self.eval_data['x']
        labels = self.eval_data['y']

        ypred = np.dot(x_vecs, self.W_t)
        
        return accuracy_score(np.sign(ypred).flatten(), labels.flatten())

    @property
    def num_test_samples(self):
        return len(self.eval_data['y'])

    @property
    def model(self):
        return self._model