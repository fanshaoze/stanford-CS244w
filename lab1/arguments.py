import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = {
    'device': device,
    'seeds': [i for i in range(10)],
    'patience': 50,
    'lr': 0.1,
    'epochs': 10000,
}
