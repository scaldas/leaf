import random


from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY


class Server:
    
    def __init__(self, num_tasks, num_features, model):
        self.num_tasks = num_tasks
        self.model = model

    def train_model(self, num_outer_iters, num_inner_iters, clients):
        """Trains global model on given clients.
        
        Each client's data is trained with the given number of epochs.

        Args:
            num_epochs: Number of epochs to train.
            clients: list of Client objects.
        Return:
            bytes_written: number of bytes written by each client to server 
                dictionary with client ids as keys and integer values.
            client computations: number of FLOPs computed by each client
                dictionary with client ids as keys and integer values.
            bytes_read: number of bytes read by each client from server
                dictionary with client ids as keys and integer values.
        """
        sys_metrics = {
            c.id: {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        '''
        deltaW = np.zeros((num_features, self.num_tasks), dtype=np.float64)
        deltaB = np.zeros((num_features, self.num_tasks), dtype=np.float64)

        for c in clients:
            self.model.send_to([c])
            deltaW_t, deltaB_t = c.train(sigma, rho, deltaW, deltaB)

            sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp

            self.updates.append((num_samples, update))
            sys_metrics[c.id][BYTES_WRITTEN_KEY] += self.model.size

        return sys_metrics
        '''

        for i in range(num_outer_iters):

            for j in range(num_inner_iters):
                # Loop over tasks (in parallel).
                deltaB = np.zeros((num_features, self.num_tasks), dtype=np.float64)

                for c in clients:
                    t = c.id
                    self.model.send_to([c])
                    curr_sig = self.model.get_sigma(t)
                    deltaB[:, t] = c.train(sigma, self.model.rho) 

                # Combine updates globally.
                self.model.update_weights(deltaB)

            # Update Omega and Sigma making sure the eigenvalues are positive.
            self.model.update_covariances()

    def update_model(self):
        self.model.update(self.updates)
        self.updates = []

    def test_model(self, clients_to_test=None):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
        """
        if clients_to_test is None:
            clients_to_test = self.selected_clients
        metrics = {}

        self.model.send_to(clients_to_test)
        
        for client in clients_to_test:
            c_metrics = client.test(self.model.cur_model)
            metrics[client.id] = c_metrics

        return metrics

    def get_clients_test_info(self, clients=None):
        """Returns the ids, hierarchies and num_test_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients
        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_test_samples for c in clients}
        return ids, groups, num_samples

    @property
    def model(self):
        return self._W

