import os
import torch


# data_path = '/Users/syednabhan/Documents/Graph to Scalar/Example-1'


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    torch.manual_seed(worker_seed)
    

def load_train_data(data_path, device):
    
    train_loader = torch.load(os.path.join(data_path, 'data', 'train_dataset.pt'), weights_only=False, map_location=device)
    individual_list_of_training_graphs = []

    for batch in train_loader:
        individual_graphs = batch.to_data_list()
        individual_graphs = [graph.to(device) for graph in individual_graphs]
        individual_list_of_training_graphs.extend(individual_graphs)

    return individual_list_of_training_graphs


def load_test_data(data_path, device):
    
    test_loader = torch.load(os.path.join(data_path, 'data', 'test_dataset_2.pt'), weights_only=False, map_location=device)
    individual_list_of_test_graphs = []

    for batch in test_loader:
        individual_graphs = batch.to_data_list()
        individual_graphs = [graph.to(device) for graph in individual_graphs]
        individual_list_of_test_graphs.extend(individual_graphs)

    return individual_list_of_test_graphs

