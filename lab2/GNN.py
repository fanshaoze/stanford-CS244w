import torch
import os
import os.path as osp

print("PyTorch has version {}".format(torch.__version__))
print([osp.dirname(__file__)])

from torch_geometric.datasets import TUDataset

import torch.nn as nn
import torch.nn.functional as F
# print(torch.__version__)

# The PyG built-in GCNConv
from torch_geometric.nn import GCNConv

import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator


def get_num_classes(pyg_dataset):
    # TODO: Implement a function that takes a PyG dataset object
    # and returns the number of classes for that dataset.

    num_classes = 0

    ############# Your code here ############
    ## (~1 line of code)
    ## Note
    ## 1. Colab autocomplete functionality might be useful.
    return pyg_dataset.num_classes

    #########################################

    # return num_classes


def get_num_features(pyg_dataset):
    # TODO: Implement a function that takes a PyG dataset object
    # and returns the number of features for that dataset.

    num_features = 0

    ############# Your code here ############
    ## (~1 line of code)
    ## Note
    ## 1. Colab autocomplete functionality might be useful.

    #########################################

    return pyg_dataset.num_features


def get_graph_class(pyg_dataset, idx):
    # TODO: Implement a function that takes a PyG dataset object,
    # an index of a graph within the dataset, and returns the class/label
    # of the graph (as an integer).

    label = -1

    ############# Your code here ############
    ## (~1 line of code)
    label = pyg_dataset[idx].y
    #########################################

    return label


def get_graph_num_edges(pyg_dataset, idx):
    # TODO: Implement a function that takes a PyG dataset object,
    # the index of a graph in the dataset, and returns the number of
    # edges in the graph (as an integer). You should not count an edge
    # twice if the graph is undirected. For example, in an undirected
    # graph G, if two nodes v and u are connected by an edge, this edge
    # should only be counted once.

    num_edges = 0

    ############# Your code here ############
    ## Note:
    ## 1. You can't return the data.num_edges directly
    ## 2. We assume the graph is undirected
    ## 3. Look at the PyG dataset built in functions
    ## (~4 lines of code)
    # print(pyg_dataset[idx].is_directed())
    num_edges = pyg_dataset[idx].num_edges / 2
    #########################################

    return num_edges


def graph_num_features(data):
    # TODO: Implement a function that takes a PyG data object,
    # and returns the number of features in the graph (as an integer).

    num_features = 0

    ############# Your code here ############
    ## (~1 line of code)
    num_features = pyg_dataset.num_classes
    #########################################

    return num_features


def test_PygNodePropPredDataset():
    dataset_name = 'ogbn-arxiv'
    # Load the dataset and transform it to sparse tensor
    dataset = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())
    print('The {} dataset has {} graph'.format(dataset_name, len(dataset)))

    # Extract the graph
    data = dataset[0]
    print(data)
    num_features = graph_num_features(data)
    print('The graph has {} features'.format(num_features))


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dropout, return_embeds=False):
        # TODO: Implement a function that initializes self.convs,
        # self.bns, and self.softmax.

        super(GCN, self).__init__()

        # A list of GCNConv layers
        self.convs = None

        # A list of 1D batch normalization layers
        self.bns = None

        # The log softmax layer
        self.softmax = None

        ############# Your code here ############
        ## Note:
        ## 1. You should use torch.nn.ModuleList for self.convs and self.bns
        ## 2. self.convs has num_layers GCNConv layers
        ## 3. self.bns has num_layers - 1 BatchNorm1d layers
        ## 4. You should use torch.nn.LogSoftmax for self.softmax
        ## 5. The parameters you can set for GCNConv include 'in_channels' and
        ## 'out_channels'. For more information please refer to the documentation:
        ## https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GCNConv
        ## 6. The only parameter you need to set for BatchNorm1d is 'num_features'
        ## For more information please refer to the documentation:
        ## https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        ## (~10 lines of code)

        # same type of layer one group

        self.convs = torch.nn.ModuleList(
            [GCNConv(in_channels=input_dim, out_channels=hidden_dim)] +
            [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim) for _ in range(num_layers-2)] +
            [GCNConv(in_channels=hidden_dim, out_channels=output_dim)])

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers-1)])

        self.softmax = nn.LogSoftmax(dim=1)


        #########################################

        # Probability of an element getting zeroed
        self.dropout = dropout

        # Skip classification layer and return node embeddings
        self.return_embeds = return_embeds

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t):
        # TODO: Implement a function that takes the feature tensor x and
        # edge_index tensor adj_t and returns the output tensor as
        # shown in the figure.

        out = None

        ############# Your code here ############
        ## Note:
        ## 1. Construct the network as shown in the figure
        ## 2. torch.nn.functional.relu and torch.nn.functional.dropout are useful
        ## For more information please refer to the documentation:
        ## https://pytorch.org/docs/stable/nn.functional.html
        ## 3. Don't forget to set F.dropout training to self.training
        ## 4. If return_embeds is True, then skip the last softmax layer
        ## (~7 lines of code)

        #########################################

        return out


def train(model, data, train_idx, optimizer, loss_fn):
    # TODO: Implement a function that trains the model by
    # using the given optimizer and loss_fn.
    model.train()
    loss = 0

    ############# Your code here ############
    ## Note:
    ## 1. Zero grad the optimizer
    ## 2. Feed the data into the model
    ## 3. Slice the model output and label by train_idx
    ## 4. Feed the sliced output and label to loss_fn
    ## (~4 lines of code)

    #########################################

    loss.backward()
    optimizer.step()

    return loss.item()


if 'IS_GRADESCOPE_ENV' not in os.environ:
    root = './enzymes'
    name = 'ENZYMES'

    # The ENZYMES dataset
    pyg_dataset = TUDataset(root, name)

    # You will find that there are 600 graphs in this dataset
    num_classes = get_num_classes(pyg_dataset)
    num_features = get_num_features(pyg_dataset)
    print(f"{name} dataset has {num_classes} classes and {num_features} features")

    graph_0 = pyg_dataset[0]
    idx = 100
    label = get_graph_class(pyg_dataset, idx)
    print('Graph with index {} has label {}'.format(idx, label))

    idx = 200
    num_edges = get_graph_num_edges(pyg_dataset, idx)
    print('Graph with index {} has {} edges'.format(idx, num_edges))

    dataset_name = 'ogbn-arxiv'
    dataset = PygNodePropPredDataset(name=dataset_name, transform=T.ToSparseTensor())
    data = dataset[0]

    # Make the adjacency matrix to symmetric
    data.adj_t = data.adj_t.to_symmetric()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # If you use GPU, the device should be cuda
    print('Device: {}'.format(device))

    data = data.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)
