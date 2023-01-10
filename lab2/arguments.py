import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 10,
}
