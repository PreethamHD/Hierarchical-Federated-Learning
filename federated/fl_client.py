import flwr as fl
import torch

class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self): return [val.cpu().numpy() for val in self.model.state_dict().values()]
    def set_parameters(self, params):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.model.state_dict().keys(), params)}
        self.model.load_state_dict(state_dict, strict=True)
    def fit(self, params, config):
        self.set_parameters(params)
        # local training logic
        return self.get_parameters(), len(self.train_loader.dataset), {}
    def evaluate(self, params, config):
        self.set_parameters(params)
        # evaluation logic
        return 0.0, len(self.test_loader.dataset), {"accuracy": 0.0}
