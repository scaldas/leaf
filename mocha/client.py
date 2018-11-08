import warnings


class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id # integer
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data

        self.W_t

    def train(self, sigma, rho):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train.
            batch_size: Size of training batches.
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        deltaB_t = self.model.train(self.train_data, self.W_t, sigma, rho)
        return deltaB_t

    def test(self, model):
        """Tests self.model on self.eval_data.

        Return:
            dict of metrics returned by the model.
        """
        return model.test(self.eval_data)

    @property
    def num_test_samples(self):
        return len(self.eval_data['y'])

    @property
    def model(self):
        return self._model
