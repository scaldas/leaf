import numpy as np


class Server:
    
    def __init__(self, num_clients, num_features, model):
        self.num_clients = num_clients
        self.num_features = num_features
        self.model = model

    def orchestrate_local_training(self, num_inner_iters, clients):
        """Trains global model on given clients."""
        def get_total_num_data_points(clients):
            count = 0
            for c in clients:
                count += c.train_data['x'].shape[0]
            return count

        total_n = get_total_num_data_points(clients)
        for _ in range(num_inner_iters):
            # Loop over clients (in parallel).
            deltaB = np.zeros((self.num_features, self.num_clients), dtype=np.float64)

            for c in clients:
                t = c.id
                self.send_model_to_clients([c])
                curr_sig = self.model.get_sigma(t)
                curr_omega = self.model.get_omega(t)
                deltaB[:, t], curr_iter = c.train(curr_sig, curr_omega, self.model.rho, 
                    self.model.max_curr_iter, total_n) 

            # Combine updates globally.
            self.model.update_weights(deltaB)
            self.model.update_curr_iters(curr_iter)

    def update_global_covariances(self):
        # Update Omega and Sigma making sure the eigenvalues are positive.
        self.model.update_covariances()

    def test_model(self, clients_to_test):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
        """
        self.send_model_to_clients(clients_to_test)
        
        accuracies = {}
        for client in clients_to_test:
            c_accuracy = client.test()
            accuracies[client.id] = c_accuracy

        return accuracies

    def get_clients_test_info(self, clients):
        """Returns the ids, hierarchies and num_test_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_test_samples for c in clients}
        return ids, groups, num_samples

    def send_model_to_clients(self, clients):
        for c in clients:
            c.W_t = self.model.cur_model[:, c.id]

