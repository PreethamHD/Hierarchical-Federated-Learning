import flwr as fl

def start_server():
    strategy = fl.server.strategy.FedAvg(min_fit_clients=3, min_evaluate_clients=3, min_available_clients=3)
    fl.server.start_server(server_address="127.0.0.1:8080",
                           config=fl.server.ServerConfig(num_rounds=5),
                           strategy=strategy)
