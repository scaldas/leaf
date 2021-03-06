import numpy as np


class Server:
    
    def __init__(self, num_clients, num_features, model):
        self.num_clients = num_clients
        self.num_features = num_features
        self.model = model

    def orchestrate_local_training(self, num_inner_iters, clients):
        """Trains global model on given clients."""
        for _ in range(num_inner_iters):
            # Loop over clients (in parallel).
            deltaB = np.zeros((self.num_features, self.num_clients), dtype=np.float64)

            for c in clients:
                t = c.id
                self.send_model_to_clients([c])
                curr_sig = self.model.get_sigma(t)
                deltaB[:, t] = c.train(curr_sig, self.model.rho) 

            # Combine updates globally.
            self.model.update_weights(deltaB)

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

        accuracies_list = []
        num_samples = []

        for client in clients_to_test:
            c_accuracy = client.test()
            accuracies[client.id] = c_accuracy

            accuracies_list.append(c_accuracy)
            num_samples.append(client.num_test_samples)

        return accuracies, np.average(accuracies_list, weights=num_samples)

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
