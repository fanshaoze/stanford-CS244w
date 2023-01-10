import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
args = {
    'device': device,
    'num_layers': 3,
    'hidden_dim': 256,
    'dropout': 0.5,
    'lr': 0.01,
    'epochs': 100,
}

GPP_args = {
  'device': device,
  'num_layers': 5,
  'hidden_dim': 256,
  'dropout': 0.5,
  'lr': 0.001,
  'epochs': 30,
}
