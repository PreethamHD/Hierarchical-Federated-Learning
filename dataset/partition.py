from torch.utils.data import Subset
import numpy as np

def iid_partition(dataset, num_clients):
    data_len = len(dataset)
    shard_size = data_len // num_clients
    return [Subset(dataset, range(i * shard_size, (i + 1) * shard_size)) for i in range(num_clients)]
